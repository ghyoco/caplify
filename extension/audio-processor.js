/**
 * AudioWorkletProcessor for Caplify.
 * Captures tab audio, downsamples to 16,000 Hz mono Float32,
 * and emits 100ms frames (1600 samples) to the service worker.
 */

class CaplifyAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.chunkSize = 1600; // 100ms @ 16kHz
    this.accumulationBuffer = new Float32Array(this.chunkSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // Mixdown stereo channels to mono
    const channel0 = input[0];
    const channel1 = input[1] || channel0;
    const inputLen = channel0.length;
    const mono = new Float32Array(inputLen);
    for (let i = 0; i < inputLen; i++) {
      mono[i] = (channel0[i] + channel1[i]) * 0.5;
    }

    // Downsample to 16kHz using linear interpolation
    const ratio = sampleRate / this.targetSampleRate;
    const outputLen = Math.floor(inputLen / ratio);

    for (let i = 0; i < outputLen; i++) {
      const srcIdx = i * ratio;
      const idx = Math.floor(srcIdx);
      const frac = srcIdx - idx;
      const s0 = mono[idx] || 0;
      const s1 = mono[idx + 1] || s0;
      const sample = s0 + frac * (s1 - s0);

      this.accumulationBuffer[this.bufferIndex++] = sample;

      if (this.bufferIndex >= this.chunkSize) {
        const frame = this.accumulationBuffer.slice(0, this.chunkSize);
        this.port.postMessage(frame.buffer, [frame.buffer]);
        this.bufferIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor("caplify-audio-processor", CaplifyAudioProcessor);
