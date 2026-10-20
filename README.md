# Caplify: Real-Time Live Captioning & Noise Suppression Pipeline

Caplify is a high-performance, real-time live captioning pipeline featuring integrated DeepFilterNet3 noise suppression, Silero Voice Activity Detection (VAD), and a dual-mode English ASR engine (Vosk for ultra-low latency <50ms, Faster-Whisper INT8 for high accuracy). Captured via a lightweight Chrome Manifest V3 extension, audio is streamed over WebSockets and rendered natively into HTML5 `<video>` elements on streaming platforms.

---

## 1. System Requirements & Hardware Target

This guide is tailored specifically for:
- **OS:** Windows 10 (64-bit)
- **CPU:** Intel Core i7-3770K (AVX supported)
- **GPU:** NVIDIA GeForce GTX 1060 (6GB VRAM, Pascal Architecture `sm_61`)
- **System Memory:** 16GB DDR3 RAM
- **CUDA Target:** CUDA 11.8 (cu118)

---

## 2. Manual Environment Setup (Windows 10 + CUDA 11.8)

Do not run automated installers or package setups on the remote headless server. Perform the following steps inside a Miniconda or Anaconda prompt on your Windows 10 PC:

### Step 2.1: Create & Activate Conda Environment

```cmd
conda create -n caplify python=3.10 -y
conda activate caplify
```

### Step 2.2: Install PyTorch with CUDA 11.8 (Pascal sm_61 Safe)

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify your GPU is recognized:
```cmd
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Step 2.3: Install ASR & Quantization Engines

```cmd
pip install ctranslate2 faster-whisper vosk
```

### Step 2.4: Install Noise Suppression & Voice Activity Detection

```cmd
pip install deepfilternet onnxruntime
```

### Step 2.5: Install Hugging Face & Audio Processing Stack

```cmd
pip install transformers datasets accelerate peft evaluate jiwer soundfile librosa scipy
```

### Step 2.6: Install FastAPI Backend & WebSocket Server

```cmd
pip install fastapi "uvicorn[standard]" websockets pydantic
```

### Step 2.7: Install Jupyter Notebook Kernel

```cmd
pip install ipykernel ipywidgets matplotlib
python -m ipykernel install --user --name caplify --display-name "Python (caplify)"
```

---

## 3. Offline Model & Dataset Setup

### 3.1 Vosk English Model (Fast Mode)

1. Download the lightweight English model (`vosk-model-small-en-us-0.15.zip`, ~40MB):
   - URL: [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
   - Direct link: [https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)
2. Extract the archive into `models/vosk-model-small-en-us-0.15/` so that `models/vosk-model-small-en-us-0.15/am/` and configuration files exist.

### 3.2 Datasets for Notebooks

Store all training datasets under `training/dataset/`:

#### A. ASR Domain Adaptation (Whisper LoRA)
- **GigaSpeech (`training/dataset/gigaspeech/`)**:
  - Contains real-world YouTube and podcast audio with natural conversational bleed and acoustic variation.
  - Setup on Windows 10:
    ```cmd
    huggingface-cli login
    python -c "from datasets import load_dataset; ds = load_dataset('speechcolab/gigaspeech', 'xs', split='train'); print('GigaSpeech xs cached successfully')"
    ```
    *(Subset `xs` is ~10 hours, ideal for fast fine-tuning on a GTX 1060; subset `s` is ~250 hours).*
- **LibriSpeech SLR12 (`training/dataset/librispeech/`)**:
  - Clean English speech baseline (`train-clean-100.tar.gz` from OpenSLR).
- **TED-LIUM 3 SLR51 (`training/dataset/tedlium/`)**:
  - Spontaneous speech recordings to enhance vocabulary and conversational flow.

#### B. Noise Suppression Multi-Tier Noise Pool (DeepFilterNet3)
- **Microsoft DNS Challenge 5 (`training/dataset/dns_noise/`)**:
  - Real-world non-stationary noise bank (ambient, crowd, domestic, technical).
  - Clone repo or download noise files: [https://github.com/microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)
- **MUSAN SLR17 (`training/dataset/musan/`)**:
  - Background music scores, speech babble, and room noise.
- **Movie SFX / Freesound (`training/dataset/movie_sfx/`)**:
  - Transient cinematic sound effects (explosions, vehicle dynamics, weather, impacts).
- **RIRS_NOISES SLR28 (`training/dataset/rirs/`)**:
  - Room impulse response filters for realistic movie acoustic emulation.

---

## 4. Hardware Safety Guidelines (GTX 1060 6GB & 16GB DDR3)

To prevent CUDA Out-Of-Memory (OOM) errors and Windows process deadlock:
1. **Multiprocessing Workers:** Always set `dataloader_num_workers=0` in PyTorch `DataLoader`. Windows uses `spawn` instead of `fork`, which duplicates memory and can cause RAM exhaustion on 16GB systems.
2. **Batch Sizing & Gradient Accumulation:**
   - `batch_size = 2`
   - `gradient_accumulation_steps = 4` (effective batch size: 8)
   - `fp16 = True` (NVIDIA Pascal hardware supports FP16 compute via PyTorch AMP).
3. **Inference Quantization:**
   - Always convert fine-tuned Whisper weights to **CTranslate2 INT8**. This reduces VRAM footprint from ~1.5GB to <400MB and speeds up decoding to ~110ms on Pascal GPUs.

---

## 5. Educational Notebooks

The repository includes two self-contained, heavily documented notebooks:
1. `notebooks/1_model_ns_finetune.ipynb`:
   - DeepFilterNet3 architecture and ERB filterbank concepts.
   - 48 kHz vs 16 kHz sample rate handling (PRD Section 6.1).
   - Synthetic SNR mixing pipeline (clean speech + MUSAN noise at 0–20 dB).
   - PyTorch fine-tuning loop with gradient accumulation and AMP FP16.
   - Spectral loss, SNR improvement evaluation, and export.
2. `notebooks/2_model_asr_finetune_and_dualmode.ipynb`:
   - Dual-Mode ASR theory: Vosk (WFST Kaldi) vs Whisper (Transformer).
   - Hugging Face datasets and Log-Mel Spectrogram preprocessing.
   - Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) on attention projections (`q_proj`, `v_proj`).
   - CTranslate2 INT8 model conversion.
   - Comparative WER & Latency benchmarking.

---

## 6. Running the Real-Time Pipeline

### Step 6.1: Start the Backend Server

```cmd
conda activate caplify
python backend/server.py
```
The server will start a FastAPI WebSocket service listening on:
`ws://localhost:8000/ws`

### Step 6.2: Install the Chrome Extension

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (top right toggle).
3. Click **Load unpacked** and select the `extension/` directory.
4. Open any video stream (e.g. YouTube, Twitch, local video file in browser).
5. Click the **Caplify** extension icon:
   - Select your mode: **Fast Mode (Vosk)** or **Accurate Mode (Whisper)**.
   - Click **Start Captioning**.
   - Captions will automatically inject into the active HTML5 `<video>` container.
