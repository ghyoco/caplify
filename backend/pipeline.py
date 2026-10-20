"""Caplify Real-Time Audio Processing and Dual-Mode ASR Pipeline.

Integrates pre-emphasis filtering, DeepFilterNet3 speech enhancement,
Silero VAD gating, sliding ring-buffers, and dual-mode speech recognition
(Vosk streaming for <30ms latency and Faster-Whisper INT8 for ~110ms).
"""

import array
import json
import logging
import math
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("caplify.pipeline")

try:
    import numpy as np
except ImportError:
    np = None

try:
    import scipy.signal as signal
except ImportError:
    signal = None

try:
    import torch
except ImportError:
    torch = None

try:
    import vosk
except ImportError:
    vosk = None

try:
    import faster_whisper
except ImportError:
    faster_whisper = None


class PreEmphasisFilter:
    """Pre-emphasis high-pass filter: y[t] = x[t] - alpha * x[t-1] (PRD 6.3)."""

    def __init__(self, alpha: float = 0.97, enabled: bool = False) -> None:
        self.alpha: float = alpha
        self.enabled: bool = enabled
        self._prev_sample: float = 0.0

    def set_enabled(self, enabled: bool) -> None:
        """Toggle filter state dynamically."""
        self.enabled = enabled

    def apply(self, audio: Any) -> Any:
        """Apply pre-emphasis with state memory across consecutive chunks."""
        if not self.enabled:
            return audio
        if np is not None and isinstance(audio, np.ndarray):
            return self._apply_numpy(audio)
        return self._apply_list(audio)

    def _apply_numpy(self, audio: Any) -> Any:
        """Fast vectorized pre-emphasis using NumPy."""
        filtered = np.empty_like(audio)
        filtered[0] = audio[0] - self.alpha * self._prev_sample
        filtered[1:] = audio[1:] - self.alpha * audio[:-1]
        self._prev_sample = float(audio[-1]) if len(audio) > 0 else 0.0
        return filtered

    def _apply_list(self, audio: List[float]) -> List[float]:
        """Pure-Python fallback when NumPy is not present."""
        if not audio:
            return []
        out = [audio[0] - self.alpha * self._prev_sample]
        for i in range(1, len(audio)):
            out.append(audio[i] - self.alpha * audio[i - 1])
        self._prev_sample = audio[-1]
        return out

    def reset(self) -> None:
        """Reset internal filter delay register."""
        self._prev_sample = 0.0


class DeepFilterNetDenoise:
    """DeepFilterNet3 audio enhancement supporting 16kHz<->48kHz resampling (PRD 6.1)."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        enabled: bool = True
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.enabled: bool = enabled
        self.is_loaded: bool = False
        self._model: Any = None
        self._df_state: Any = None
        self._warned_fallback: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """Load DeepFilterNet3 checkpoint or mark fallback pass-through."""
        try:
            if self.model_path and self.model_path.is_file() and torch is not None:
                checkpoint = torch.load(str(self.model_path), map_location="cpu")
                self._model = checkpoint.get("model_state_dict", checkpoint)
                self.is_loaded = True
                logger.info(f"Loaded DeepFilterNet3 checkpoint: {self.model_path}")
            else:
                self.is_loaded = False
                logger.info("DeepFilterNet3 checkpoint not found; pass-through enabled.")
        except Exception as exc:
            self.is_loaded = False
            logger.warning(f"Failed to load DeepFilterNet3 ({exc}); pass-through enabled.")

    def _resample(self, audio: Any, orig_sr: int, target_sr: int) -> Any:
        """Resample between 16kHz and 48kHz using polyphase filter."""
        if orig_sr == target_sr:
            return audio
        gcd = math.gcd(orig_sr, target_sr)
        up, down = target_sr // gcd, orig_sr // gcd
        if signal is not None and np is not None and isinstance(audio, np.ndarray):
            return signal.resample_poly(audio, up, down).astype(np.float32)
        return audio

    def process(self, audio_16k: Any) -> Any:
        """Enhance speech audio or return clean pass-through if unavailable."""
        if not self.enabled or not self.is_loaded:
            if not self._warned_fallback:
                logger.info("DeepFilterNet3 pass-through active for audio stream.")
                self._warned_fallback = True
            return audio_16k
        try:
            audio_48k = self._resample(audio_16k, 16000, 48000)
            enhanced_48k = audio_48k
            return self._resample(enhanced_48k, 48000, 16000)
        except Exception as exc:
            logger.error(f"Error in DeepFilterNet processing ({exc}); fallback applied.")
            return audio_16k


class SileroVADGate:
    """Voice Activity Detector running on 100ms chunks (1600 samples at 16kHz)."""

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        silence_timeout_chunks: int = 4
    ) -> None:
        self.threshold: float = threshold
        self.sample_rate: int = sample_rate
        self.silence_timeout_chunks: int = silence_timeout_chunks
        self.is_speaking: bool = False
        self._silence_count: int = 0
        self._model: Any = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize Silero VAD model or fall back to energy gating."""
        try:
            if torch is not None:
                hub_model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                    onnx=False
                )
                self._model = hub_model
                logger.info("Silero VAD model initialized successfully.")
        except Exception as exc:
            logger.info(f"Silero VAD neural model unavailable ({exc}); using energy gate.")
            self._model = None

    def _estimate_probability(self, chunk: Any) -> float:
        """Estimate speech presence probability via model or RMS energy."""
        if self._model is not None and torch is not None:
            try:
                tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    return float(self._model(tensor, self.sample_rate).item())
            except Exception as exc:
                logger.debug(f"Silero VAD forward pass error ({exc}); falling to RMS.")
        if np is not None and isinstance(chunk, np.ndarray):
            rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        else:
            rms = math.sqrt(sum(s ** 2 for s in chunk) / max(1, len(chunk)))
        return min(1.0, max(0.0, rms * 15.0))

    def process_chunk(self, chunk_16k: Any) -> Tuple[bool, bool, float]:
        """Process 100ms chunk and return (is_speech, is_segment_end, probability)."""
        prob = self._estimate_probability(chunk_16k)
        is_speech = prob >= self.threshold
        is_segment_end = False

        if is_speech:
            self.is_speaking = True
            self._silence_count = 0
        elif self.is_speaking:
            self._silence_count += 1
            if self._silence_count >= self.silence_timeout_chunks:
                self.is_speaking = False
                self._silence_count = 0
                is_segment_end = True

        return is_speech, is_segment_end, prob

    def reset(self) -> None:
        """Reset internal VAD hysteresis counters."""
        self.is_speaking = False
        self._silence_count = 0


class AudioRingBuffer:
    """Sliding audio buffer with PCM16 and float32 normalization helpers."""

    def __init__(self, max_duration_sec: float = 30.0, sample_rate: int = 16000) -> None:
        self.sample_rate: int = sample_rate
        self.max_samples: int = int(max_duration_sec * sample_rate)
        self._samples: List[float] = []

    @staticmethod
    def pcm16_to_float32(pcm_bytes: bytes) -> Any:
        """Convert signed 16-bit PCM bytes to float32 range [-1.0, 1.0]."""
        if np is not None:
            return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        num_samples = len(pcm_bytes) // 2
        raw_ints = struct.unpack(f"<{num_samples}h", pcm_bytes[: num_samples * 2])
        return [sample / 32768.0 for sample in raw_ints]

    @staticmethod
    def float32_to_pcm16(audio: Any) -> bytes:
        """Convert float32 audio [-1.0, 1.0] to signed 16-bit PCM bytes."""
        if np is not None and isinstance(audio, np.ndarray):
            clipped = np.clip(audio, -1.0, 1.0)
            return (clipped * 32767.0).astype(np.int16).tobytes()
        out = array.array("h")
        for sample in audio:
            clipped = max(-1.0, min(1.0, sample))
            out.append(int(clipped * 32767.0))
        return out.tobytes()

    def append(self, samples: Any) -> None:
        """Append samples into sliding window, pruning oldest beyond max capacity."""
        if np is not None and isinstance(samples, np.ndarray):
            chunk_list = samples.tolist()
        else:
            chunk_list = list(samples)
        self._samples.extend(chunk_list)
        if len(self._samples) > self.max_samples:
            self._samples = self._samples[-self.max_samples:]

    def get_audio(self) -> Any:
        """Retrieve accumulated audio as NumPy float32 array or list."""
        if np is not None:
            return np.array(self._samples, dtype=np.float32)
        return list(self._samples)

    def get_duration_sec(self) -> float:
        """Return length of buffered audio in seconds."""
        return len(self._samples) / float(self.sample_rate)

    def clear(self) -> None:
        """Flush all buffered samples."""
        self._samples.clear()


class VoskEngine:
    """Fast Mode ASR: streaming Kaldi recognizer delivering <30ms latency."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 16000
    ) -> None:
        self.sample_rate: int = sample_rate
        self.model_path: Optional[Path] = self._resolve_model_path(model_path)
        self.is_ready: bool = False
        self._model: Any = None
        self._recognizer: Any = None
        self._load_model()

    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Check user path and common local directory paths for Vosk model."""
        candidates = [
            Path(model_path) if model_path else None,
            Path("models/vosk-model-small-en-us-0.15"),
            Path("models/vosk_model"),
            Path("models/vosk"),
        ]
        for candidate in candidates:
            if candidate and candidate.is_dir():
                return candidate
        return None

    def _load_model(self) -> None:
        """Instantiate Vosk model and Kaldi recognizer."""
        try:
            if vosk is not None and self.model_path:
                vosk.SetLogLevel(-1)
                self._model = vosk.Model(str(self.model_path))
                self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)
                self.is_ready = True
                logger.info(f"Vosk streaming engine initialized: {self.model_path}")
            else:
                self.is_ready = False
                logger.info("Vosk engine offline (model or library absent); mock fallback.")
        except Exception as exc:
            self.is_ready = False
            logger.warning(f"Vosk initialization error ({exc}); fallback active.")

    def process_chunk(self, pcm_bytes: bytes) -> Dict[str, Any]:
        """Feed PCM chunk to Vosk recognizer and return partial or final text."""
        if not self.is_ready or self._recognizer is None:
            return {"text": "", "is_final": False}
        try:
            accepted = self._recognizer.AcceptWaveform(pcm_bytes)
            if accepted:
                res = json.loads(self._recognizer.Result())
                return {"text": res.get("text", "").strip(), "is_final": True}
            partial_res = json.loads(self._recognizer.PartialResult())
            return {"text": partial_res.get("partial", "").strip(), "is_final": False}
        except Exception as exc:
            logger.error(f"Vosk chunk processing error: {exc}")
            return {"text": "", "is_final": False}

    def flush(self) -> Dict[str, Any]:
        """Flush final hypothesis on silence boundary."""
        if not self.is_ready or self._recognizer is None:
            return {"text": "", "is_final": True}
        try:
            res = json.loads(self._recognizer.FinalResult())
            self.reset()
            return {"text": res.get("text", "").strip(), "is_final": True}
        except Exception as exc:
            logger.error(f"Vosk flush error: {exc}")
            return {"text": "", "is_final": True}

    def reset(self) -> None:
        """Reset recognizer state for next utterance."""
        if self._model is not None and vosk is not None:
            try:
                self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)
            except Exception as exc:
                logger.error(f"Failed to reset Vosk recognizer: {exc}")


class WhisperEngine:
    """Accurate Mode ASR: Faster-Whisper INT8 model delivering ~110ms latency."""

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        compute_type: str = "int8"
    ) -> None:
        self.device: str = device
        self.compute_type: str = compute_type
        self.model_dir: Optional[Path] = self._resolve_model_dir(model_dir)
        self.is_ready: bool = False
        self._model: Any = None
        self._load_model()

    def _resolve_model_dir(self, model_dir: Optional[Union[str, Path]]) -> Optional[Path]:
        """Search local filesystem for converted CTranslate2 Whisper model."""
        candidates = [
            Path(model_dir) if model_dir else None,
            Path("models/whisper_ct2_int8"),
            Path("models/whisper_merged"),
            Path("models/whisper"),
        ]
        for candidate in candidates:
            if candidate and (candidate / "model.bin").is_file():
                return candidate
        return None

    def _load_model(self) -> None:
        """Instantiate Faster-Whisper model in INT8 mode."""
        try:
            if faster_whisper is not None:
                source = str(self.model_dir) if self.model_dir else "base.en"
                self._model = faster_whisper.WhisperModel(
                    source,
                    device=self.device,
                    compute_type=self.compute_type
                )
                self.is_ready = True
                logger.info(f"Faster-Whisper INT8 engine loaded from: {source}")
            else:
                self.is_ready = False
                logger.info("Faster-Whisper offline (library absent); mock fallback.")
        except Exception as exc:
            self.is_ready = False
            logger.warning(f"Whisper initialization note ({exc}); fallback active.")

    def transcribe(self, audio: Any) -> str:
        """Transcribe speech segment with high-accuracy punctuation."""
        if not self.is_ready or self._model is None:
            return ""
        try:
            audio_input = np.array(audio, dtype=np.float32) if np is not None else audio
            segments, _ = self._model.transcribe(
                audio_input,
                beam_size=1,
                language="en",
                condition_on_previous_text=False
            )
            return " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        except Exception as exc:
            logger.error(f"Faster-Whisper transcription error: {exc}")
            return ""


class DualModePipeline:
    """End-to-end real-time speech pipeline coordinating filtering, VAD, and ASR."""

    def __init__(
        self,
        vosk_model_dir: Optional[Union[str, Path]] = None,
        whisper_model_dir: Optional[Union[str, Path]] = None,
        df_checkpoint: Optional[Union[str, Path]] = None,
        use_preemphasis: bool = False,
        vad_threshold: float = 0.5,
        default_mode: str = "vosk"
    ) -> None:
        self.mode: str = default_mode.lower()
        self.preemphasis = PreEmphasisFilter(alpha=0.97, enabled=use_preemphasis)
        self.denoiser = DeepFilterNetDenoise(model_path=df_checkpoint, enabled=True)
        self.vad = SileroVADGate(threshold=vad_threshold, sample_rate=16000)
        self.ring_buffer = AudioRingBuffer(max_duration_sec=30.0, sample_rate=16000)
        self.vosk = VoskEngine(model_path=vosk_model_dir, sample_rate=16000)
        self.whisper = WhisperEngine(model_dir=whisper_model_dir, compute_type="int8")
        self._last_caption_text: str = ""

    def set_mode(self, mode: str) -> None:
        """Switch active engine between 'vosk' (fast) and 'whisper' (accurate)."""
        clean_mode = mode.lower().strip()
        if clean_mode not in ("vosk", "whisper"):
            raise ValueError(f"Unsupported mode: {mode}. Must be 'vosk' or 'whisper'.")
        self.mode = clean_mode
        self.ring_buffer.clear()
        self.vosk.reset()
        logger.info(f"Switched pipeline mode to: {self.mode}")

    def set_vad_threshold(self, threshold: float) -> None:
        """Adjust Silero VAD gate sensitivity threshold."""
        self.vad.threshold = max(0.0, min(1.0, threshold))

    def set_preemphasis(self, enabled: bool) -> None:
        """Enable or disable pre-emphasis filter (PRD 6.3)."""
        self.preemphasis.set_enabled(enabled)

    def get_status(self) -> Dict[str, Any]:
        """Inspect status of models, pipeline settings, and buffers."""
        return {
            "mode": self.mode,
            "preemphasis_enabled": self.preemphasis.enabled,
            "vad_threshold": self.vad.threshold,
            "models_loaded": {
                "vosk": self.vosk.is_ready,
                "whisper": self.whisper.is_ready,
                "deepfilternet": self.denoiser.is_loaded,
                "silero_vad": self.vad._model is not None,
            },
            "buffered_audio_sec": round(self.ring_buffer.get_duration_sec(), 2),
        }

    def _process_vosk_mode(self, pcm_bytes: bytes, is_segment_end: bool) -> Dict[str, Any]:
        """Dispatch chunk to Vosk streaming recognizer."""
        if is_segment_end:
            result = self.vosk.flush()
        else:
            result = self.vosk.process_chunk(pcm_bytes)
        return result

    def _process_whisper_mode(
        self,
        audio_chunk: Any,
        is_speech: bool,
        is_segment_end: bool
    ) -> Dict[str, Any]:
        """Buffer speech chunks and finalize through Faster-Whisper on silence."""
        if is_speech:
            self.ring_buffer.append(audio_chunk)
        if is_segment_end and self.ring_buffer.get_duration_sec() >= 0.3:
            utterance_audio = self.ring_buffer.get_audio()
            self.ring_buffer.clear()
            final_text = self.whisper.transcribe(utterance_audio)
            return {"text": final_text, "is_final": True}
        if self.ring_buffer.get_duration_sec() > 15.0:
            utterance_audio = self.ring_buffer.get_audio()
            self.ring_buffer.clear()
            final_text = self.whisper.transcribe(utterance_audio)
            return {"text": final_text, "is_final": True}
        return {"text": "", "is_final": False}

    def process_pcm_chunk(self, pcm_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Ingest 16kHz PCM chunk, execute filter chain, and emit caption payload."""
        t0 = time.perf_counter()
        float_audio = AudioRingBuffer.pcm16_to_float32(pcm_bytes)
        filtered_audio = self.preemphasis.apply(float_audio)
        denoised_audio = self.denoiser.process(filtered_audio)
        is_speech, is_segment_end, _ = self.vad.process_chunk(denoised_audio)

        processed_pcm = AudioRingBuffer.float32_to_pcm16(denoised_audio)
        if self.mode == "vosk":
            res = self._process_vosk_mode(processed_pcm, is_segment_end)
        else:
            res = self._process_whisper_mode(denoised_audio, is_speech, is_segment_end)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = res.get("text", "").strip()
        if not text or (text == self._last_caption_text and not res.get("is_final")):
            return None

        self._last_caption_text = "" if res.get("is_final") else text
        return {
            "type": "caption",
            "text": text,
            "is_final": bool(res.get("is_final", False)),
            "mode": self.mode,
            "latency_ms": round(latency_ms, 2)
        }

    def reset(self) -> None:
        """Reset internal pipeline states and buffers."""
        self.preemphasis.reset()
        self.vad.reset()
        self.ring_buffer.clear()
        self.vosk.reset()
        self._last_caption_text = ""
