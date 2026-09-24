class AudioRecorder {
  constructor(onChunkCallback, onVolumeCallback) {
    this.onChunkCallback = onChunkCallback;
    this.onVolumeCallback = onVolumeCallback;
    this.audioContext = null;
    this.mediaStream = null;
    this.workletNode = null;
    this.scriptProcessor = null;
    this.isRecording = false;
    this.targetSampleRate = 16000;
  }

  async start() {
    if (this.isRecording) return;

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
          sampleRate: this.targetSampleRate
        }
      });

      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.targetSampleRate
      });

      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      
      // Fallback ScriptProcessorNode for maximum browser compatibility
      const bufferSize = 4096;
      this.scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

      this.scriptProcessor.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Calculate volume meter level
        if (this.onVolumeCallback) {
          let sum = 0;
          for (let i = 0; i < inputData.length; i++) {
            sum += inputData[i] * inputData[i];
          }
          const rms = Math.sqrt(sum / inputData.length);
          this.onVolumeCallback(rms);
        }

        // Resample to targetSampleRate (16000 Hz) if browser audio context runs at 44.1k/48k
        const currentRate = (this.audioContext && this.audioContext.sampleRate) ? this.audioContext.sampleRate : this.targetSampleRate;
        const processedData = (currentRate !== this.targetSampleRate)
          ? this.resample(inputData, currentRate, this.targetSampleRate)
          : inputData;

        // Convert Float32Array [-1.0, 1.0] to PCM Int16 ArrayBuffer
        const pcm16Buffer = this.floatTo16BitPCM(processedData);
        if (this.onChunkCallback) {
          this.onChunkCallback(pcm16Buffer, processedData);
        }
      };

      source.connect(this.scriptProcessor);
      this.scriptProcessor.connect(this.audioContext.destination);

      this.isRecording = true;
      console.log("Microphone recording started.");
    } catch (err) {
      console.error("Microphone permission denied or unavailable:", err);
      throw err;
    }
  }

  stop() {
    if (!this.isRecording) return;
    this.isRecording = false;

    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    console.log("Microphone recording stopped.");
  }

  floatTo16BitPCM(output) {
    const buffer = new ArrayBuffer(output.length * 2);
    const view = new DataView(buffer);
    let offset = 0;
    for (let i = 0; i < output.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, output[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return buffer;
  }

  resample(buffer, fromSampleRate, toSampleRate) {
    if (!fromSampleRate || !toSampleRate || fromSampleRate === toSampleRate) return buffer;
    const ratio = fromSampleRate / toSampleRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const originIndex = i * ratio;
      const indexPrev = Math.floor(originIndex);
      const indexNext = Math.min(indexPrev + 1, buffer.length - 1);
      const fraction = originIndex - indexPrev;
      result[i] = buffer[indexPrev] * (1 - fraction) + buffer[indexNext] * fraction;
    }
    return result;
  }
}
