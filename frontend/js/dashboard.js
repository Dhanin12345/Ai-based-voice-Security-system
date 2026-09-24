document.addEventListener("DOMContentLoaded", () => {
  if (!document.getElementById("syntheticChart")) return;

  // DOM Elements
  const btnStart = document.getElementById("btnStartMonitoring");
  const btnStop = document.getElementById("btnStopMonitoring");
  const fileInput = document.getElementById("fileInput");
  const btnUpload = document.getElementById("btnUpload");
  const statusBadge = document.getElementById("statusBadge");
  const riskValue = document.getElementById("riskValue");
  const confidenceValue = document.getElementById("confidenceValue");
  const classificationBadge = document.getElementById("classificationBadge");
  const recommendationBox = document.getElementById("recommendationBox");
  const alertBanner = document.getElementById("alertBanner");
  const alertBannerText = document.getElementById("alertBannerText");
  
  // Stat counters
  const statTotalWindows = document.getElementById("statTotalWindows");
  const statSuspiciousWindows = document.getElementById("statSuspiciousWindows");
  const statTotalAlerts = document.getElementById("statTotalAlerts");
  const statAvgConfidence = document.getElementById("statAvgConfidence");

  // Chart Setup
  const ctx = document.getElementById("syntheticChart").getContext("2d");
  const chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Synthetic Probability (%)",
          data: [],
          borderColor: "#ec4899",
          backgroundColor: "rgba(236, 72, 153, 0.15)",
          borderWidth: 2,
          fill: true,
          tension: 0.3
        },
        {
          label: "Risk Score Threshold (70%)",
          data: [],
          borderColor: "#ef4444",
          borderDash: [5, 5],
          borderWidth: 1,
          fill: false,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 100,
          grid: { color: "#1f293d" },
          ticks: { color: "#9ca3af" }
        },
        x: {
          grid: { color: "#1f293d" },
          ticks: { color: "#9ca3af" }
        }
      },
      plugins: {
        legend: { labels: { color: "#f3f4f6" } }
      }
    }
  });

  // Visualizer Canvas
  const canvas = document.getElementById("visualizerCanvas");
  const canvasCtx = canvas.getContext("2d");

  let recorder = null;
  let detector = null;

  let totalWindowsCount = 0;
  let suspiciousWindowsCount = 0;
  let alertCount = 0;
  let confidenceHistory = [];

  function updateVisualizer(volume) {
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    canvasCtx.fillStyle = "#00f2fe";
    const barHeight = Math.min(canvas.height, volume * canvas.height * 8);
    canvasCtx.fillRect(canvas.width / 2 - 20, (canvas.height - barHeight) / 2, 40, barHeight);
  }

  function handleDetectionResult(res) {
    totalWindowsCount++;
    if (res.synthetic_probability >= 0.6) {
      suspiciousWindowsCount++;
    }

    confidenceHistory.push(res.confidence);
    const avgConf = (confidenceHistory.reduce((a, b) => a + b, 0) / confidenceHistory.length * 100).toFixed(1);

    // Update Counter Stats
    if (statTotalWindows) statTotalWindows.innerText = totalWindowsCount;
    if (statSuspiciousWindows) statSuspiciousWindows.innerText = suspiciousWindowsCount;
    if (statAvgConfidence) statAvgConfidence.innerText = `${avgConf}%`;

    // Update Risk & Confidence Displays
    const riskScore = res.risk_score;
    if (riskValue) riskValue.innerText = `${riskScore}/100`;
    if (confidenceValue) confidenceValue.innerText = `${(res.confidence * 100).toFixed(1)}%`;

    // Update Classification Badge
    const rawHuman = (res.human_probability !== undefined) ? res.human_probability : (1 - (res.synthetic_probability || 0));
    const isHuman = (rawHuman <= 1.0 ? rawHuman * 100 : rawHuman) >= 65.0;
    if (classificationBadge) {
      classificationBadge.innerText = isHuman ? "HUMAN VOICE" : "ROBOTIC VOICE / AI VOICE";
      classificationBadge.className = "badge " + (isHuman ? "badge-safe" : "badge-danger");
    }

    // Recommendation
    if (recommendationBox) {
      recommendationBox.innerText = res.recommendation;
    }

    // Chart Update
    const timeLabel = new Date().toLocaleTimeString();
    chart.data.labels.push(timeLabel);
    chart.data.datasets[0].data.push((res.synthetic_probability * 100).toFixed(1));
    chart.data.datasets[1].data.push(70);

    if (chart.data.labels.length > 20) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
      chart.data.datasets[1].data.shift();
    }
    chart.update();

    // Alert Handling
    if (res.alert_triggered) {
      alertCount++;
      if (statTotalAlerts) statTotalAlerts.innerText = alertCount;
      if (alertBanner) alertBanner.style.display = "flex";
      if (alertBannerText) {
        alertBannerText.innerText = `Potential synthetic speech detected with Risk Score ${riskScore}/100. ${res.recommendation}`;
      }
      App.playAlertSound();
      App.showToast(`SECURITY ALERT: High Voice Cloning Risk (${riskScore}/100)!`, "danger");
    }
  }

  // Start Real-Time Monitoring
  if (btnStart) {
    btnStart.addEventListener("click", async () => {
      try {
        detector = new WebSocketDetector(handleDetectionResult, (err) => {
          App.showToast("WebSocket connection failed.", "danger");
        });

        await detector.connect();

        recorder = new AudioRecorder(
          (pcmBuffer) => {
            if (detector) detector.sendAudioChunk(pcmBuffer);
          },
          (vol) => updateVisualizer(vol)
        );

        await recorder.start();

        btnStart.style.display = "none";
        btnStop.style.display = "inline-flex";
        if (statusBadge) {
          statusBadge.innerText = "● LIVE MONITORING";
          statusBadge.className = "badge badge-danger";
        }
        App.showToast("Live microphone monitoring started.", "info");

      } catch (err) {
        App.showToast("Could not access microphone: " + err.message, "danger");
      }
    });
  }

  // Stop Real-Time Monitoring
  if (btnStop) {
    btnStop.addEventListener("click", () => {
      if (recorder) recorder.stop();
      if (detector) detector.disconnect();

      btnStop.style.display = "none";
      btnStart.style.display = "inline-flex";
      if (statusBadge) {
        statusBadge.innerText = "● IDLE";
        statusBadge.className = "badge badge-safe";
      }
      App.showToast("Monitoring stopped.", "info");
    });
  }

  // Handle Audio File Upload
  if (btnUpload && fileInput) {
    btnUpload.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", async () => {
      const file = fileInput.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      App.showToast(`Uploading and analyzing ${file.name}...`, "info");

      try {
        const apiUrl = (window.location.protocol === "file:" || !window.location.hostname) ? "http://localhost:8000/api/detect/upload" : "/api/detect/upload";
        const resp = await fetch(apiUrl, {
          method: "POST",
          body: formData
        });

        if (!resp.ok) {
          const errData = await resp.json().catch(() => ({}));
          throw new Error(errData.detail || "AUDIO INPUT ERROR");
        }

        const data = await resp.json();
        handleDetectionResult(data);

        // Process individual windows returned from file analysis
        if (data.windows && data.windows.length > 0) {
          data.windows.forEach(w => {
            chart.data.labels.push(`Window ${w.window_index}`);
            chart.data.datasets[0].data.push((w.synthetic_probability * 100).toFixed(1));
            chart.data.datasets[1].data.push(70);
          });
          chart.update();
        }

        App.showToast("Audio analysis complete!", "info");
      } catch (err) {
        const errMsg = err.message && err.message.includes("AUDIO INPUT ERROR") ? err.message : `AUDIO INPUT ERROR: ${err.message}`;
        if (classificationBadge) {
          classificationBadge.innerText = "AUDIO INPUT ERROR";
          classificationBadge.className = "badge badge-danger";
        }
        App.showToast(errMsg, "danger");
      }
    });
  }
});
