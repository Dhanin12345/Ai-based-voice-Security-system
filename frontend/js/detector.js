class WebSocketDetector {
  constructor(onResultCallback, onErrorCallback) {
    this.onResultCallback = onResultCallback;
    this.onErrorCallback = onErrorCallback;
    this.ws = null;
    this.isConnected = false;
  }

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/detect`;

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
          this.isConnected = true;
          console.log("Real-time detection WebSocket connected.");
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          try {
            const result = JSON.parse(event.data);
            if (this.onResultCallback) {
              this.onResultCallback(result);
            }
          } catch (e) {
            console.error("Failed to parse WebSocket JSON payload:", e);
          }
        };

        this.ws.onerror = (err) => {
          console.error("WebSocket error:", err);
          if (this.onErrorCallback) this.onErrorCallback(err);
          reject(err);
        };

        this.ws.onclose = () => {
          this.isConnected = false;
          console.log("WebSocket disconnected.");
        };
      } catch (e) {
        reject(e);
      }
    });
  }

  sendAudioChunk(pcmArrayBuffer) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(pcmArrayBuffer);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      this.isConnected = false;
    }
  }
}
