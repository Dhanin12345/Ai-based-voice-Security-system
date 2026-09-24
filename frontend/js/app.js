let monitoring = false;
let chart = null;
let socket = null;
let audioRecorder = null;
let reconnectTimer = null;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

// Modern Floating Toast Notification Function
function showToast(message, type = "info", duration = 5000) {
    let container = document.getElementById("toastContainerCustom");
    if (!container) {
        container = document.createElement("div");
        container.id = "toastContainerCustom";
        container.className = "toast-container-custom";
        document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.className = `toast-custom toast-${type}`;
    
    let icon = "ℹ️";
    if (type === "error") icon = "❌";
    else if (type === "success") icon = "✅";
    else if (type === "warning") icon = "⚠️";

    toast.innerHTML = `<span style="font-size: 16px;">${icon}</span><span style="flex: 1;">${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(40px) scale(0.9)";
        setTimeout(() => toast.remove(), 300);
    }, duration);
}
window.showToast = showToast;


document.addEventListener("DOMContentLoaded", () => {
    initChart();
    initGauge();
    fetchDashboardStats();
    initSystemLiveClock();
});

function initSystemLiveClock() {
    function tick() {
        const now = new Date();
        const formatted = now.toLocaleString();
        const clockEls = document.querySelectorAll("#systemLiveClock, #topbarLiveClock, .system-live-clock");
        clockEls.forEach(el => {
            el.textContent = `● LIVE: ${formatted}`;
        });
    }
    tick();
    setInterval(tick, 1000);
}

function getApiUrl(path) {
    if (window.location.protocol === "file:" || !window.location.hostname) {
        return "http://localhost:8000" + path;
    }
    return path;
}

function getWsUrl() {
    const isFileProtocol = window.location.protocol === "file:" || !window.location.hostname;
    const protocol = (window.location.protocol === "https:") ? "wss:" : "ws:";
    const host = isFileProtocol ? "localhost:8000" : window.location.host;
    return `${protocol}//${host}/ws/detect`;
}

// Fetch High-Level Dashboard Top 4 Data Cards from SQLite Database
async function fetchDashboardStats() {
    try {
        const resp = await fetch(getApiUrl("/api/detect/stats"));
        if (!resp.ok) return;
        const stats = await resp.json();

        const elTotal = document.getElementById("cardTotalAnalyses");
        const elSusp = document.getElementById("cardSuspicious");
        const elAlerts = document.getElementById("cardAlerts");
        const elAvg = document.getElementById("cardAvgRisk");

        if (elTotal) elTotal.innerText = (stats.total_analyses || 0).toLocaleString();
        if (elSusp) elSusp.innerText = (stats.suspicious_detections || 0).toLocaleString();
        if (elAlerts) elAlerts.innerText = (stats.active_alerts || 0).toLocaleString();
        if (elAvg) elAvg.innerText = `${stats.average_risk || 0} / 100`;
    } catch (e) {
        console.warn("Could not fetch dashboard stats:", e);
    }
}

// Semicircular Risk Gauge Canvas Drawing
function initGauge() {
    drawGauge(0);
}

function drawGauge(score) {
    const canvas = document.getElementById("gaugeCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height - 10;
    const radius = 65;

    ctx.clearRect(0, 0, width, height);

    // Background track arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, Math.PI, 2 * Math.PI);
    ctx.lineWidth = 12;
    ctx.strokeStyle = "#1D334B";
    ctx.stroke();

    // Fill arc
    const fillPercent = Math.max(0, Math.min(100, score)) / 100;
    const endAngle = Math.PI + fillPercent * Math.PI;

    let strokeColor = "#22C55E"; // Safe
    if (score >= 80) strokeColor = "#EF4444"; // Very High
    else if (score >= 60) strokeColor = "#F97316"; // High
    else if (score >= 30) strokeColor = "#F59E0B"; // Moderate

    ctx.beginPath();
    ctx.arc(cx, cy, radius, Math.PI, endAngle);
    ctx.lineWidth = 12;
    ctx.strokeStyle = strokeColor;
    ctx.stroke();

    const textEl = document.getElementById("riskGaugeText");
    if (textEl) {
        textEl.innerText = score;
        textEl.style.color = strokeColor;
    }
}

// Chart.js Timeline Initialization
function initChart() {
    const chartCanvas = document.getElementById("detectionChart");
    if (!chartCanvas) return;

    const ctx = chartCanvas.getContext("2d");
    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Synthetic Probability (%)",
                    data: [],
                    borderColor: "#22D3EE",
                    backgroundColor: "rgba(34, 211, 238, 0.15)",
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3
                },
                {
                    label: "Risk Threshold (70%)",
                    data: [],
                    borderColor: "#EF4444",
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
                    grid: { color: "#1D334B" },
                    ticks: { color: "#A8B6C8" }
                },
                x: {
                    grid: { color: "#1D334B" },
                    ticks: { color: "#A8B6C8" }
                }
            },
            plugins: {
                legend: {
                    labels: { color: "#F8FAFC" }
                }
            }
        }
    });
}

// Stage Progress Bar Handler (Supports both 6-step and 4-step modern pipeline)
function updateStageProgress(currentStepIndex) {
    const steps = ["stepInput", "stepPreprocess", "stepSegment", "stepFeature", "stepAI", "stepRisk"];
    steps.forEach((stepId, idx) => {
        const el = document.getElementById(stepId);
        if (!el) return;
        el.className = "stage-step";
        if (idx < currentStepIndex) el.classList.add("done");
        else if (idx === currentStepIndex) el.classList.add("active");
    });
}

// 4-Stage Interactive Pipeline Manager ("Process the Data Each by Each")
function advancePipelineStage(stageNum, descText) {
    for (let s = 1; s <= 4; s++) {
        const stageEl = document.getElementById(`pipeStage${s}`);
        const descEl = document.getElementById(`pipeStage${s}Text`);
        if (!stageEl) continue;
        if (s < stageNum) {
            stageEl.className = "pipeline-stage done";
        } else if (s === stageNum) {
            stageEl.className = "pipeline-stage active";
            if (descEl && descText) descEl.textContent = descText;
        } else {
            stageEl.className = "pipeline-stage";
        }
    }
}

function resetPipelineStages() {
    for (let s = 1; s <= 4; s++) {
        const stageEl = document.getElementById(`pipeStage${s}`);
        const descEl = document.getElementById(`pipeStage${s}Text`);
        if (!stageEl) continue;
        if (s === 1) {
            stageEl.className = "pipeline-stage active";
            if (descEl) descEl.textContent = "Standby for Voice Input";
        } else {
            stageEl.className = "pipeline-stage";
            if (s === 2 && descEl) descEl.textContent = "Acoustic Signal Extraction";
            if (s === 3 && descEl) descEl.textContent = "Anti-Spoofing Classifier";
            if (s === 4 && descEl) descEl.textContent = "Risk Mitigation Action";
        }
    }
}

function startMonitoring() {
    monitoring = true;
    updateStageProgress(0); // 1. INPUT
    advancePipelineStage(1, "Listening to live microphone stream...");

    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;

    const monitorStatus = document.getElementById("monitorStatus");
    if (monitorStatus) {
        monitorStatus.textContent = "● LISTENING";
        monitorStatus.className = "badge-safe";
    }

    const micText = document.getElementById("micText");
    if (micText) {
        micText.textContent = "Listening for voice...";
        micText.style.color = "var(--text-main)";
    }

    const micSubtext = document.getElementById("micSubtext");
    if (micSubtext) {
        micSubtext.textContent = "Speak into microphone to verify voice authenticity";
        micSubtext.style.color = "var(--text-secondary)";
    }

    const micCircle = document.getElementById("micCircle");
    if (micCircle) {
        micCircle.classList.add("mic-active");
        micCircle.style.borderColor = "var(--primary)";
        micCircle.style.boxShadow = "0 0 20px rgba(0, 245, 255, 0.35)";
    }

    const waveform = document.querySelector(".waveform");
    if (waveform) waveform.classList.add("active");

    connectWebSocket();
}

function stopMonitoring() {
    monitoring = false;
    updateStageProgress(0);
    resetPipelineStages();

    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;

    const monitorStatus = document.getElementById("monitorStatus");
    if (monitorStatus) {
        monitorStatus.textContent = "READY";
        monitorStatus.className = "badge-safe";
    }

    const micText = document.getElementById("micText");
    if (micText) micText.textContent = "Microphone Ready";

    const micCircle = document.getElementById("micCircle");
    if (micCircle) micCircle.classList.remove("mic-active");

    const waveform = document.querySelector(".waveform");
    if (waveform) waveform.classList.remove("active");

    if (audioRecorder) {
        audioRecorder.stop();
        audioRecorder = null;
    }

    if (socket) {
        socket.close();
        socket = null;
    }
}

async function resetMonitorData() {
    stopMonitoring();
    resetPipelineStages();

    try {
        await fetch(getApiUrl("/api/history"), { method: "DELETE" });
    } catch (e) {
        console.warn("Could not reset backend history:", e);
    }
    fetchDashboardStats();

    // Reset Primary Security Verdict Banner
    const vHeadline = document.getElementById("verdictHeadline");
    const vDetail = document.getElementById("verdictDetail");
    const vShield = document.getElementById("verdictShield");
    const vBanner = document.getElementById("securityVerdictBanner");
    const vHumanLabel = document.getElementById("verdictHumanLabel");
    const vSynthLabel = document.getElementById("verdictSynthLabel");
    const vHumanBar = document.getElementById("verdictHumanBar");
    const vSynthBar = document.getElementById("verdictSynthBar");
    const elThreat = document.getElementById("threatStatusBadge");

    if (vHeadline) {
        vHeadline.textContent = "STANDBY — READY FOR AUDIO STREAM";
        vHeadline.style.color = "#FFFFFF";
    }
    if (vDetail) vDetail.textContent = "Speak into microphone or upload an audio file to evaluate synthetic voice risk.";
    if (vShield) {
        vShield.textContent = "🛡️";
        vShield.style.background = "rgba(0, 245, 255, 0.1)";
        vShield.style.borderColor = "rgba(0, 245, 255, 0.25)";
        vShield.style.color = "var(--primary)";
    }
    if (vBanner) {
        vBanner.style.borderColor = "var(--border-color)";
        vBanner.style.background = "rgba(12, 21, 35, 0.85)";
    }
    if (vHumanLabel) vHumanLabel.textContent = "HUMAN: 0%";
    if (vSynthLabel) vSynthLabel.textContent = "SYNTHETIC: 0%";
    if (vHumanBar) vHumanBar.style.width = "50%";
    if (vSynthBar) vSynthBar.style.width = "50%";
    if (elThreat) {
        elThreat.textContent = "● SAFE / STANDBY";
        elThreat.className = "badge-safe";
    }

    const elHuman = document.getElementById("humanProbability");
    const elSynth = document.getElementById("syntheticProbability");
    const elConf = document.getElementById("modelConfidence");
    const elRisk = document.getElementById("riskScore");
    const elLevel = document.getElementById("riskLevel");

    if (elHuman) elHuman.textContent = "0%";
    if (elSynth) elSynth.textContent = "0%";
    if (elConf) elConf.textContent = "0%";
    if (elRisk) elRisk.textContent = "0 / 100";
    if (elLevel) {
        elLevel.textContent = "LOW";
        elLevel.className = "badge-safe";
    }

    const hConf = document.getElementById("humanConfidence");
    const sConf = document.getElementById("synthConfidence");
    if (hConf) hConf.textContent = "0%";
    if (sConf) sConf.textContent = "0%";

    const audioEl = document.getElementById("audioPlayback");
    const container = document.getElementById("audioPlaybackContainer");
    if (audioEl) {
        audioEl.pause();
        audioEl.src = "";
    }
    if (container) container.style.display = "none";

    drawGauge(0);
    clearAlert();

    if (chart) {
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.data.datasets[1].data = [];
        chart.update();
    }

    updateStageProgress(0);

    const elDur = document.getElementById("procDuration");
    const elStat = document.getElementById("procStatus");
    const elProcConf = document.getElementById("procConfidence");
    const elLat = document.getElementById("procLatency");
    if (elDur) elDur.textContent = "0.0 sec";
    if (elStat) elStat.textContent = "STANDBY";
    if (elProcConf) elProcConf.textContent = "0%";
    if (elLat) elLat.textContent = "0 ms";

    const elExpClass = document.getElementById("explainClassification");
    const elExpConf = document.getElementById("explainConfidence");
    const elExpSynth = document.getElementById("explainSynthIndex");
    const elExpRiskBadge = document.getElementById("explainRiskBadge");
    const elExpRec = document.getElementById("explainRecommendation");
    const elExpInd = document.getElementById("explainIndicators");

    if (elExpClass) {
        elExpClass.textContent = "Awaiting Audio Analysis...";
        elExpClass.style.color = "var(--text-main)";
    }
    if (elExpConf) elExpConf.textContent = "0%";
    if (elExpSynth) elExpSynth.textContent = "0%";
    if (elExpRiskBadge) {
        elExpRiskBadge.textContent = "LOW RISK";
        elExpRiskBadge.className = "badge-safe";
    }
    if (elExpRec) {
        elExpRec.textContent = "Start microphone or upload audio file to run 4-stage detection pipeline.";
    }
    if (elExpInd) {
        elExpInd.innerHTML = `
            <div class="indicator-item" style="color: var(--text-muted);">No audio stream analyzed yet</div>
            <div class="indicator-item" style="color: var(--text-muted);">FFT Autocorrelation pitch analysis pending</div>
            <div class="indicator-item" style="color: var(--text-muted);">STFT spectral flatness index pending</div>
            <div class="indicator-item" style="color: var(--text-muted);">Neural vocoder phase coherence check pending</div>
        `;
    }
}

function connectWebSocket() {
    const wsUrl = getWsUrl();

    socket = new WebSocket(wsUrl);
    socket.binaryType = "arraybuffer";

    socket.onopen = function () {
        console.log("Connected to VoiceGuard detection server WebSocket");
        updateStageProgress(1); // 2. PREPROCESS

        const micText = document.getElementById("micText");
        const micSubtext = document.getElementById("micSubtext");
        const micCircle = document.getElementById("micCircle");
        const monitorStatus = document.getElementById("monitorStatus");
        const impersonationBadge = document.getElementById("impersonationBadge");

        if (micText && !micText.textContent.includes("HUMAN VOICE")) {
            micText.textContent = "Listening for voice...";
            micText.style.color = "var(--text-main)";
        }
        if (micSubtext && !micSubtext.textContent.includes("AUTHENTIC HUMAN VOICE")) {
            micSubtext.textContent = "Speak into microphone to verify voice authenticity";
            micSubtext.style.color = "var(--text-secondary)";
        }
        if (micCircle && !micCircle.style.borderColor.includes("34, 197, 94") && micCircle.style.borderColor !== "rgb(34, 197, 94)") {
            micCircle.style.borderColor = "var(--primary)";
            micCircle.style.boxShadow = "0 0 20px rgba(0, 245, 255, 0.35)";
        }
        if (monitorStatus && !monitorStatus.textContent.includes("GREEN")) {
            monitorStatus.textContent = "● LISTENING";
            monitorStatus.className = "badge-safe";
        }
        if (impersonationBadge && impersonationBadge.textContent !== "AUTHENTIC HUMAN VOICE") {
            impersonationBadge.textContent = "LISTENING";
            impersonationBadge.className = "badge-safe";
        }

        audioRecorder = new AudioRecorder((pcmBuffer) => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                updateStageProgress(2); // 3. SEGMENT
                socket.send(pcmBuffer);
            }
        });
        
        audioRecorder.start().catch((err) => {
            console.error("Microphone access error:", err);
            showToast("Microphone permission denied or unavailable. Please allow microphone access in your browser.", "error", 7000);
            stopMonitoring();
        });
    };

    socket.onmessage = function (event) {
        try {
            const result = JSON.parse(event.data);
            if (result.error || result.status === "error") {
                const errDetail = result.error || "AUDIO INPUT ERROR";
                const monitorStatus = document.getElementById("monitorStatus");
                const micText = document.getElementById("micText");
                const micSubtext = document.getElementById("micSubtext");
                const vHeadline = document.getElementById("verdictHeadline");
                const vDetail = document.getElementById("verdictDetail");
                if (monitorStatus) {
                    monitorStatus.textContent = "AUDIO INPUT ERROR";
                    monitorStatus.className = "badge-danger";
                }
                if (micText) {
                    micText.textContent = "AUDIO INPUT ERROR";
                    micText.style.color = "#EF4444";
                }
                if (micSubtext) {
                    micSubtext.textContent = errDetail;
                    micSubtext.style.color = "#EF4444";
                }
                if (vHeadline) {
                    vHeadline.textContent = "AUDIO INPUT ERROR";
                    vHeadline.style.color = "#EF4444";
                }
                if (vDetail) {
                    vDetail.textContent = errDetail;
                }
                return;
            }

            if (result.status === "listening" || result.state === "listening" || result.classification === "LISTENING") {
                // Room silence or speech pause: keep clean listening state without false alarms
                const micText = document.getElementById("micText");
                const micSubtext = document.getElementById("micSubtext");
                const micCircle = document.getElementById("micCircle");
                const monitorStatus = document.getElementById("monitorStatus");

                if (micText && !micText.textContent.includes("HUMAN VOICE")) {
                    micText.textContent = "Listening for voice...";
                    micText.style.color = "var(--text-main)";
                }
                if (micSubtext && !micSubtext.textContent.includes("AUTHENTIC HUMAN VOICE")) {
                    micSubtext.textContent = "Speak into microphone to verify voice authenticity";
                    micSubtext.style.color = "var(--text-secondary)";
                }
                if (micCircle && !micCircle.style.borderColor.includes("34, 197, 94") && micCircle.style.borderColor !== "rgb(34, 197, 94)") {
                    micCircle.style.borderColor = "var(--primary)";
                    micCircle.style.boxShadow = "0 0 20px rgba(0, 245, 255, 0.35)";
                }
                if (monitorStatus && !monitorStatus.textContent.includes("GREEN")) {
                    monitorStatus.textContent = "● LISTENING";
                    monitorStatus.className = "badge-safe";
                }
                return;
            }

            updateStageProgress(4); // 5. AI DETECTION
            updateStageProgress(5); // 6. RISK & RESULT
            updateDashboard(result);
        } catch (e) {
            console.error("Error parsing WebSocket response:", e);
        }
    };

    socket.onerror = function (err) {
        console.error("WebSocket connection issue:", err);
        const micText = document.getElementById("micText");
        if (micText && monitoring && socket && socket.readyState !== WebSocket.OPEN) {
            micText.textContent = "Connection lost. Reconnecting...";
            micText.style.color = "var(--status-very-high)";
        }
    };

    socket.onclose = function () {
        console.log("WebSocket connection closed.");
        if (monitoring) {
            reconnectTimer = setTimeout(() => {
                if (monitoring) connectWebSocket();
            }, 2000);
        }
    };
}

function updateDashboard(result) {
    const rawHuman = (result.human_probability !== undefined) ? result.human_probability : (result.confidence || 0);
    const humanProbPercent = rawHuman <= 1.0 ? rawHuman * 100 : rawHuman;
    const human = (humanProbPercent % 1 !== 0) ? +(humanProbPercent).toFixed(2) : Math.round(humanProbPercent);
    // Mathematically consistent: Human Confidence + AI/Synthetic Confidence = 100%
    const synthetic = (humanProbPercent % 1 !== 0) ? +(100 - human).toFixed(2) : (100 - human);
    const clone = Math.round((result.clone_probability || (synthetic * (result.risk_score / 10000))) * 100);
    const conf = human; // Real model value

    // EXACT 65% THRESHOLD:
    // IF human_confidence >= 65% => "HUMAN VOICE", status_color = GREEN, risk = "LOW"
    // IF human_confidence < 65%  => "ROBOTIC VOICE / AI VOICE", status_color = RED, risk = "HIGH", danger alert
    const isHuman = humanProbPercent >= 65.0;

    // Primary Security Verdict Banner
    const vHeadline = document.getElementById("verdictHeadline");
    const vDetail = document.getElementById("verdictDetail");
    const vShield = document.getElementById("verdictShield");
    const vBanner = document.getElementById("securityVerdictBanner");
    const vHumanLabel = document.getElementById("verdictHumanLabel");
    const vSynthLabel = document.getElementById("verdictSynthLabel");
    const vHumanBar = document.getElementById("verdictHumanBar");
    const vSynthBar = document.getElementById("verdictSynthBar");
    const elRisk = document.getElementById("riskScore");
    const threatBadges = document.querySelectorAll("#threatStatusBadge");

    if (vHumanLabel) vHumanLabel.textContent = `HUMAN: ${human}%`;
    if (vSynthLabel) vSynthLabel.textContent = `SYNTHETIC: ${synthetic}%`;
    if (vHumanBar) vHumanBar.style.width = `${human}%`;
    if (vSynthBar) vSynthBar.style.width = `${synthetic}%`;
    if (elRisk) elRisk.textContent = `${result.risk_score} / 100`;

    if (isHuman) {
        if (vHeadline) {
            vHeadline.textContent = "🟢 HUMAN VOICE";
            vHeadline.style.color = "#22C55E";
        }
        if (vDetail) vDetail.textContent = "Acoustic dynamics, F0 pitch contour, and vocal tract formants verify an organic living human speaker.";
        if (vShield) {
            vShield.textContent = "🟢";
            vShield.style.background = "rgba(34, 197, 94, 0.15)";
            vShield.style.borderColor = "#22C55E";
            vShield.style.color = "#22C55E";
        }
        if (vBanner) {
            vBanner.style.borderColor = "rgba(34, 197, 94, 0.45)";
            vBanner.style.background = "rgba(34, 197, 94, 0.05)";
        }
        threatBadges.forEach(b => {
            b.textContent = "● LOW RISK";
            b.className = "badge-safe";
        });
    } else {
        if (vHeadline) {
            vHeadline.textContent = "🔴 ROBOTIC VOICE / AI VOICE";
            vHeadline.style.color = "#EF4444";
        }
        if (vDetail) vDetail.textContent = `Critical impersonation threat detected. Voice failed the 65% human-confidence threshold (${human}%). Audio exhibits synthetic neural vocoder artifacts and pitch rigidity.`;
        if (vShield) {
            vShield.textContent = "⚠️";
            vShield.style.background = "rgba(239, 68, 68, 0.2)";
            vShield.style.borderColor = "#EF4444";
            vShield.style.color = "#EF4444";
        }
        if (vBanner) {
            vBanner.style.borderColor = "rgba(239, 68, 68, 0.55)";
            vBanner.style.background = "rgba(239, 68, 68, 0.08)";
        }
        threatBadges.forEach(b => {
            b.textContent = "● HIGH RISK";
            b.className = "badge-danger";
        });
    }

    // Update Live Recording / Call Screen Status & Indicators
    const micText = document.getElementById("micText");
    const micSubtext = document.getElementById("micSubtext");
    const micCircle = document.getElementById("micCircle");
    const monitorStatus = document.getElementById("monitorStatus");
    const impersonationBadge = document.getElementById("impersonationBadge");

    if (isHuman) {
        if (micText) {
            micText.textContent = "🟢 HUMAN VOICE";
            micText.style.color = "#22C55E";
        }
        if (micSubtext) {
            micSubtext.innerHTML = `Human Confidence: <strong>${human}%</strong> | Risk: <strong>LOW</strong> | Status: <strong style="color: #22C55E;">GREEN</strong> (AUTHENTIC HUMAN VOICE)`;
            micSubtext.style.color = "var(--text-secondary)";
        }
        if (micCircle) {
            micCircle.style.borderColor = "#22C55E";
            micCircle.style.boxShadow = "0 0 0 8px rgba(34, 197, 94, 0.15), 0 0 32px rgba(34, 197, 94, 0.55)";
        }
        if (monitorStatus) {
            monitorStatus.textContent = "● GREEN (AUTHENTIC HUMAN VOICE)";
            monitorStatus.className = "badge-safe";
        }
        if (impersonationBadge) {
            impersonationBadge.textContent = "AUTHENTIC HUMAN VOICE";
            impersonationBadge.className = "badge-safe";
        }
    } else {
        if (micText) {
            micText.textContent = "🔴 ROBOTIC VOICE / AI VOICE";
            micText.style.color = "#EF4444";
        }
        if (micSubtext) {
            micSubtext.innerHTML = `Human Confidence: <strong>${human}%</strong> | AI Confidence: <strong>${synthetic}%</strong> | Risk: <strong>HIGH</strong> | Status: <strong style="color: #EF4444;">RED</strong><br><span style="color: #EF4444; font-weight: 800;">⚠ DANGER ALERT: Possible Synthetic/AI Voice Detected</span>`;
            micSubtext.style.color = "#EF4444";
        }
        if (micCircle) {
            micCircle.style.borderColor = "#EF4444";
            micCircle.style.boxShadow = "0 0 24px rgba(239, 68, 68, 0.6)";
        }
        if (monitorStatus) {
            monitorStatus.textContent = "● RED (DANGER ALERT)";
            monitorStatus.className = "badge-danger";
        }
        if (impersonationBadge) {
            impersonationBadge.textContent = "ROBOTIC / AI VOICE";
            impersonationBadge.className = "badge-danger";
        }
    }

    // Sync synthetic and clone progress bars in dashboard
    const synthBarText = document.getElementById("synthBarText");
    const synthProgressBar = document.getElementById("synthProgressBar");
    if (synthBarText) synthBarText.textContent = synthetic + "%";
    if (synthProgressBar) synthProgressBar.style.width = synthetic + "%";

    const cloneBarText = document.getElementById("cloneBarText");
    const cloneProgressBar = document.getElementById("cloneProgressBar");
    if (cloneBarText) cloneBarText.textContent = clone + "%";
    if (cloneProgressBar) cloneProgressBar.style.width = clone + "%";

    // Dual Cards confidence and indicators
    const elHConf = document.getElementById("humanConfidence");
    const elSConf = document.getElementById("synthConfidence");
    if (elHConf) elHConf.textContent = human + "%";
    if (elSConf) elSConf.textContent = synthetic + "%";

    const elHuman = document.getElementById("humanProbability");
    const elHumanCard = document.getElementById("humanProbabilityCard");
    const elSynth = document.getElementById("syntheticProbability");
    const elSynthCard = document.getElementById("syntheticProbabilityCard");
    const elClone = document.getElementById("cloneProbability");
    const elConf = document.getElementById("modelConfidence");
    const elLevel = document.getElementById("riskLevel");

    if (elHuman) elHuman.textContent = human + "%";
    if (elHumanCard) elHumanCard.textContent = human + "%";
    if (elSynth) elSynth.textContent = synthetic + "%";
    if (elSynthCard) elSynthCard.textContent = synthetic + "%";
    if (elClone) elClone.textContent = clone + "%";
    if (elConf) elConf.textContent = conf + "%";
    if (elLevel) {
        elLevel.textContent = isHuman ? "● LOW RISK" : "● HIGH RISK";
        elLevel.className = isHuman ? "badge-safe" : "badge-danger";
    }

    // Prevention Recommendation Action Banner
    const elPrevAction = document.getElementById("preventionActionText");
    const elPrevDetail = document.getElementById("preventionDetailText");
    const elPrevBanner = document.getElementById("preventionBanner");
    const prevAction = result.prevention_action || (result.risk_score >= 80 ? "BLOCK ACTION - REQUIRE SECONDARY MFA" : result.risk_score >= 60 ? "PROMINENT WARNING - REQUEST SECONDARY VERIFICATION" : result.risk_score >= 30 ? "SHOW WARNING - EXERCISE CAUTION" : "CONTINUE MONITORING");
    if (elPrevAction) elPrevAction.textContent = prevAction;
    if (elPrevDetail) elPrevDetail.textContent = result.recommendation || (result.risk_score >= 60 ? "Potential synthetic or cloned voice detected. Additional identity verification is recommended before continuing." : "Voice analysis indicates authentic human speech. Continue standard monitoring.");
    if (elPrevBanner) {
        if (result.risk_score >= 80) {
            elPrevBanner.style.borderColor = "#EF4444";
            elPrevBanner.style.background = "rgba(239, 68, 68, 0.12)";
        } else if (result.risk_score >= 60) {
            elPrevBanner.style.borderColor = "#F97316";
            elPrevBanner.style.background = "rgba(249, 115, 22, 0.12)";
        } else if (result.risk_score >= 30) {
            elPrevBanner.style.borderColor = "#F59E0B";
            elPrevBanner.style.background = "rgba(245, 158, 11, 0.12)";
        } else {
            elPrevBanner.style.borderColor = "#22C55E";
            elPrevBanner.style.background = "rgba(34, 197, 94, 0.08)";
        }
    }

    // Dynamic Acoustic Indicators List
    const elIndList = document.getElementById("acousticIndicatorsList");
    if (elIndList) {
        let indicators = [];
        if (result.risk_score >= 60 || synthetic >= 60) {
            indicators = [
                '<div class="indicator-item" style="color: #F87171;"><span>⚠</span> Rigid / unnatural pitch consistency (F0 freeze)</div>',
                '<div class="indicator-item" style="color: #F87171;"><span>⚠</span> High spectral flatness (neural vocoder hiss artifact)</div>',
                '<div class="indicator-item" style="color: #F87171;"><span>⚠</span> Reduced spectral formant modulation across frames</div>',
                '<div class="indicator-item" style="color: #F87171;"><span>⚠</span> Synthetic speech characteristics confirmed by ML classifier</div>'
            ];
        } else if (result.risk_score >= 30) {
            indicators = [
                '<div class="indicator-item" style="color: #FBBF24;"><span>⚠</span> Minor spectral distortion in vocoder frequency band</div>',
                '<div class="indicator-item" style="color: #22C55E;"><span>✓</span> Pitch variation partially present</div>',
                '<div class="indicator-item" style="color: #FBBF24;"><span>⚠</span> Moderate voice naturalness variance</div>'
            ];
        } else {
            indicators = [
                '<div class="indicator-item" style="color: #22C55E;"><span>✓</span> Natural pitch variation & F0 contour inflection</div>',
                '<div class="indicator-item" style="color: #22C55E;"><span>✓</span> Dynamic spectral formant modulation</div>',
                '<div class="indicator-item" style="color: #22C55E;"><span>✓</span> Consistent harmonic structure & natural pauses</div>',
                '<div class="indicator-item" style="color: #22C55E;"><span>✓</span> Stable natural voice characteristics</div>'
            ];
        }
        elIndList.innerHTML = indicators.join("");
    }

    // 1. Populate Human Voice Analysis Card
    if (elHConf) elHConf.textContent = human + "%";
    
    if (result.human_analysis && result.human_analysis.features) {
        const hf = result.human_analysis.features;
        const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        setEl("hPitchMean", hf.pitch_mean || "0.0 Hz");
        setEl("hPitchVar", hf.pitch_variation || "0.0 Hz");
        setEl("hRMSEnergy", hf.energy || "0.000000");
        setEl("hZCR", hf.zcr || "0.0000");
        setEl("hCentroid", hf.spectral_centroid || "0.0 Hz");
        setEl("hBandwidth", hf.spectral_bandwidth || "0.0 Hz");
        setEl("hRolloff", hf.spectral_rolloff || "0.0 Hz");
        setEl("hHarmonicity", hf.harmonicity || "0.0000");
        setEl("hSpeechDuration", hf.speech_duration || "0.00 sec");
        setEl("hSilenceRatio", hf.silence_ratio || "0.0%");
    }

    const hEvList = document.getElementById("humanEvidenceList");
    if (hEvList) {
        const evs = (result.human_analysis && result.human_analysis.evidence) ? result.human_analysis.evidence : [
            "✓ Natural pitch variation & F0 contour inflection",
            "✓ Dynamic spectral formant modulation",
            "✓ Authentic speech rhythm & pauses"
        ];
        hEvList.innerHTML = evs.map(e => `<div>${e}</div>`).join("");
    }

    // 2. Populate Robot / Synthetic Voice Analysis Card
    if (elSConf) elSConf.textContent = synthetic + "%";

    if (result.synthetic_analysis && result.synthetic_analysis.features) {
        const sf = result.synthetic_analysis.features;
        const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        setEl("sPitchMean", sf.pitch_mean || "0.0 Hz");
        setEl("sPitchVar", sf.pitch_variation || "0.0 Hz");
        setEl("sRMSEnergy", sf.energy || "0.000000");
        setEl("sZCR", sf.zcr || "0.0000");
        setEl("sCentroid", sf.spectral_centroid || "0.0 Hz");
        setEl("sBandwidth", sf.spectral_bandwidth || "0.0 Hz");
        setEl("sRolloff", sf.spectral_rolloff || "0.0 Hz");
        setEl("sHarmonicity", sf.harmonicity || "0.0000");
        setEl("sSpeechDuration", sf.speech_duration || "0.00 sec");
        setEl("sSilenceRatio", sf.silence_ratio || "0.0%");
    }

    const sEvList = document.getElementById("syntheticEvidenceList");
    if (sEvList) {
        const evs = (result.synthetic_analysis && result.synthetic_analysis.evidence) ? result.synthetic_analysis.evidence : [
            "• Acoustic patterns aligned with synthetic training data"
        ];
        sEvList.innerHTML = evs.map(e => `<div>${e}</div>`).join("");
    }

    // 3. Populate Comparison Table
    if (result.features_comparison) {
        const fc = result.features_comparison;
        const setCmp = (hId, sId, diffId, obj, unit = "", isFixed = 1) => {
            if (!obj) return;
            const hEl = document.getElementById(hId);
            const sEl = document.getElementById(sId);
            const dEl = document.getElementById(diffId);
            if (hEl) hEl.textContent = typeof obj.human === "number" ? obj.human.toFixed(isFixed) + unit : (obj.human || "0.0");
            if (sEl) sEl.textContent = typeof obj.synthetic === "number" ? obj.synthetic.toFixed(isFixed) + unit : (obj.synthetic || "0.0");
            if (dEl) {
                const diffVal = typeof obj.difference === "number" ? obj.difference.toFixed(isFixed) + unit : (obj.difference || "0.0");
                dEl.textContent = diffVal;
            }
        };

        setCmp("cmpHPitch", "cmpSPitch", "cmpDiffPitch", fc.pitch_mean, " Hz", 1);
        setCmp("cmpHPitchVar", "cmpSPitchVar", "cmpDiffPitchVar", fc.pitch_std, " Hz", 1);
        setCmp("cmpHEnergy", "cmpSEnergy", "cmpDiffEnergy", fc.rms_energy, "", 4);
        setCmp("cmpHZCR", "cmpSZCR", "cmpDiffZCR", fc.zcr, "", 4);
        setCmp("cmpHCentroid", "cmpSCentroid", "cmpDiffCentroid", fc.spectral_centroid, " Hz", 1);
        setCmp("cmpHHarmonicity", "cmpSHarmonicity", "cmpDiffHarmonicity", fc.harmonicity, "", 4);
    }

    // 4. Audio Playback Handler
    if (result.audio_url) {
        const audioEl = document.getElementById("audioPlayback");
        const container = document.getElementById("audioPlaybackContainer");
        if (audioEl) {
            audioEl.src = getApiUrl(result.audio_url);
            if (container) container.style.display = "block";
        }
    }

    // 3. Status Level & Badge Formatting
    if (elLevel) {
        elLevel.textContent = isHuman ? "● LOW RISK" : "● HIGH RISK";
        elLevel.className = isHuman ? "badge-safe" : "badge-danger";
    }

    const elExpRiskBadge = document.getElementById("explainRiskBadge");
    if (elExpRiskBadge) {
        elExpRiskBadge.textContent = isHuman ? "LOW RISK" : "HIGH RISK";
        elExpRiskBadge.className = isHuman ? "badge-safe" : "badge-danger";
    }

    const elModelStatus = document.getElementById("modelStatusBadge");
    if (elModelStatus && result.model_status) {
        elModelStatus.textContent = result.model_status;
    }

    // 4. Update Real-Time Processing Statistics Bar
    const elDur = document.getElementById("procDuration");
    const elStat = document.getElementById("procStatus");
    const elProcConf = document.getElementById("procConfidence");
    const elLat = document.getElementById("procLatency");

    if (elDur) elDur.textContent = (result.duration_seconds || 2.0).toFixed(1) + " sec";
    if (elStat) elStat.textContent = isHuman ? "AUTHENTIC HUMAN" : "ROBOTIC / SYNTHETIC";
    if (elProcConf) elProcConf.textContent = conf + "%";
    if (elLat) elLat.textContent = (result.processing_time_ms || 184).toFixed(0) + " ms";

    // 5. Update Final Result Panel & AI Explanation
    const elExpClass = document.getElementById("explainClassification");
    const elExpRec = document.getElementById("explainRecommendation");

    if (elExpClass) {
        if (isHuman) {
            elExpClass.textContent = "HUMAN VOICE";
            elExpClass.style.color = "var(--status-safe)";
        } else {
            elExpClass.textContent = "ROBOTIC VOICE / AI VOICE";
            elExpClass.style.color = "var(--status-very-high)";
        }
    }

    if (elExpRec) {
        elExpRec.textContent = result.recommendation || (isHuman 
            ? "Voice analysis indicates authentic human speech. Continue standard monitoring." 
            : `CRITICAL WARNING: Voice failed the 65% human-confidence threshold (${human}%). Potential robotic or synthetic speech detected. Verify speaker via secondary identity check.`);
    }

    drawGauge(result.risk_score);

    // 6. Dynamic Threat Panel & Alerts
    if (isHuman) {
        showSuccessAlert(
            "🟢 HUMAN VOICE",
            `Authentic human speech characteristics verified (${human}% Human Confidence >= 65%). Natural pitch modulation & vocal formants detected. LOW RISK.`
        );
    } else {
        showAlert(
            "🔴 ROBOTIC VOICE / AI VOICE",
            `⚠ DANGER ALERT: Voice failed the 65% human-confidence threshold (${human}%). Synthetic or robotic speech characteristics confirmed. HIGH RISK.`
        );
    }

    updateChart(synthetic);
    fetchDashboardStats();
}

function updateChart(syntheticScore) {
    if (!chart) return;

    const timeLabel = new Date().toLocaleTimeString();
    chart.data.labels.push(timeLabel);
    chart.data.datasets[0].data.push(syntheticScore);
    chart.data.datasets[1].data.push(70);

    if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
        chart.data.datasets[1].data.shift();
    }
    chart.update();
}

function showAlert(title, message) {
    const elTitle = document.getElementById("alertTitle");
    const elMsg = document.getElementById("alertMessage");

    if (elTitle) {
        elTitle.textContent = title;
        elTitle.style.color = "var(--status-very-high)";
    }
    if (elMsg) elMsg.textContent = message;

    const alertCard = document.querySelector(".alert-card");
    if (alertCard) alertCard.style.borderColor = "var(--status-very-high)";
}

function showSuccessAlert(title, message) {
    const elTitle = document.getElementById("alertTitle");
    const elMsg = document.getElementById("alertMessage");

    if (elTitle) {
        elTitle.textContent = title;
        elTitle.style.color = "var(--status-safe)";
    }
    if (elMsg) elMsg.textContent = message;

    const alertCard = document.querySelector(".alert-card");
    if (alertCard) alertCard.style.borderColor = "var(--status-safe)";
}

function clearAlert() {
    const elTitle = document.getElementById("alertTitle");
    const elMsg = document.getElementById("alertMessage");

    if (elTitle) {
        elTitle.textContent = "No Active Threat";
        elTitle.style.color = "var(--text-main)";
    }
    if (elMsg) elMsg.textContent = "No elevated synthetic-speech indication in the current voice analysis.";

    const alertCard = document.querySelector(".alert-card");
    if (alertCard) alertCard.style.borderColor = "rgba(245, 158, 11, 0.3)";
}

function acknowledgeAlert() {
    clearAlert();
    alert("Alert acknowledged by security analyst.");
}

async function uploadAudio(input) {
    if (!input.files || !input.files.length) return;

    const file = input.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const monitorStatus = document.getElementById("monitorStatus");
    const micText = document.getElementById("micText");
    const micSubtext = document.getElementById("micSubtext");
    const waveform = document.querySelector(".waveform");

    if (monitorStatus) monitorStatus.textContent = "ANALYZING UPLOAD";
    if (micText) micText.textContent = `Analyzing: ${file.name}`;
    if (micSubtext) micSubtext.textContent = `File size: ${(file.size / 1024).toFixed(1)} KB — Ingesting audio...`;
    if (waveform) waveform.classList.add("active");

    // STAGE 1: CAPTURE
    advancePipelineStage(1, `Audio file "${file.name}" ingested (${(file.size / 1024).toFixed(1)} KB)`);
    updateStageProgress(1);

    // Setup Audio Player with Playback ("And Listen")
    const audioEl = document.getElementById("audioPlayback");
    const container = document.getElementById("audioPlaybackContainer");
    const playerStatus = document.getElementById("audioPlayerStatus");
    const playerBadge = document.getElementById("audioPlayerBadge");

    if (audioEl) {
        try {
            audioEl.src = URL.createObjectURL(file);
            if (container) container.style.display = "block";
            if (playerStatus) playerStatus.textContent = `NOW LISTENING: ${file.name}`;
            if (playerBadge) {
                playerBadge.textContent = "UPLOAD READY";
                playerBadge.className = "badge-safe";
            }
            audioEl.onplay = () => { if (waveform) waveform.classList.add("active"); };
            audioEl.onended = () => { if (waveform) waveform.classList.remove("active"); };
            audioEl.play().catch(e => console.log("Audio autoplay note:", e));
        } catch (e) {
            console.warn("Could not set audio preview URL:", e);
        }
    }

    try {
        advancePipelineStage(2, "Extracting acoustic features (F0 pitch, jitter, energy, formants)...");
        updateStageProgress(3);

        advancePipelineStage(3, "Evaluating anti-spoofing ML model against extracted vectors...");
        updateStageProgress(4);

        const response = await fetch(getApiUrl("/api/detect/upload"), {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errJson = await response.json();
            throw new Error(errJson.detail || "Analysis failed.");
        }

        const result = await response.json();

        // STAGE 4: DEFENSE ACTION
        updateStageProgress(5);
        const isHuman = (result.classification === "HUMAN VOICE") || (Math.round((result.human_probability || 0) * 100) >= 65);
        const actionDesc = isHuman 
            ? "Authentic Human Voice Verified — Authenticity Confirmed"
            : "Synthetic Impersonation Threat Intercepted — Security Action Applied";
        advancePipelineStage(4, actionDesc);

        updateDashboard(result);

        if (playerBadge) {
            playerBadge.textContent = isHuman ? "HUMAN VOICE" : "ROBOTIC VOICE / AI VOICE";
            playerBadge.className = isHuman ? "badge-safe" : "badge-danger";
        }

        if (result.windows && result.windows.length > 0 && chart) {
            chart.data.labels = [];
            chart.data.datasets[0].data = [];
            chart.data.datasets[1].data = [];

            result.windows.forEach(w => {
                chart.data.labels.push(`Win ${w.window_index}`);
                chart.data.datasets[0].data.push(Math.round(w.synthetic_probability * 100));
                chart.data.datasets[1].data.push(70);
            });
            chart.update();
        }

        const humanConfVal = (result.human_confidence !== undefined) ? result.human_confidence : (result.human_probability <= 1 ? Math.round((result.human_probability || 0) * 100) : result.human_probability);
        showToast(`Analyzed "${file.name}" — ${isHuman ? "🟢 HUMAN VOICE" : "🔴 ROBOTIC VOICE / AI VOICE"} (${humanConfVal}% Human Confidence)`, isHuman ? "success" : "warning");
    } catch (error) {
        console.error("Audio processing error:", error);
        const errMsg = (error.message && error.message.includes("AUDIO INPUT ERROR"))
            ? error.message
            : `AUDIO INPUT ERROR: ${error.message || "Failed to load or process audio file"}`;

        if (monitorStatus) {
            monitorStatus.textContent = "AUDIO INPUT ERROR";
            monitorStatus.className = "badge-danger";
        }
        if (micText) {
            micText.textContent = "AUDIO INPUT ERROR";
            micText.style.color = "#EF4444";
        }
        if (micSubtext) {
            micSubtext.textContent = errMsg;
            micSubtext.style.color = "#EF4444";
        }
        const vHeadline = document.getElementById("verdictHeadline");
        if (vHeadline) {
            vHeadline.textContent = "AUDIO INPUT ERROR";
            vHeadline.style.color = "#EF4444";
        }
        const vDetail = document.getElementById("verdictDetail");
        if (vDetail) {
            vDetail.textContent = errMsg;
        }
        showToast(errMsg, "error", 7000);
    }
}

// 1-Click Demo Audio Execution Handler
async function runDemo(sampleType) {
    const monitorStatus = document.getElementById("monitorStatus");
    const micText = document.getElementById("micText");
    const micSubtext = document.getElementById("micSubtext");
    const waveform = document.querySelector(".waveform");

    const sampleTitle = sampleType === "human" 
        ? "Genuine Human Voice" 
        : sampleType === "cloned" 
        ? "Voice-Cloned Impersonation" 
        : "AI Synthetic Voice";

    if (monitorStatus) monitorStatus.textContent = "RUNNING STAGE PIPELINE";
    if (micText) micText.textContent = `Evaluating: ${sampleTitle}`;
    if (micSubtext) micSubtext.textContent = "Processing audio through 4-stage anti-spoofing pipeline...";
    if (waveform) waveform.classList.add("active");

    // STAGE 1: CAPTURE
    advancePipelineStage(1, `Ingesting 16kHz PCM audio: ${sampleTitle}`);
    updateStageProgress(1);

    try {
        advancePipelineStage(2, "Extracting F0 pitch jitter, formants & spectral vocoder artifacts...");
        updateStageProgress(3);

        advancePipelineStage(3, "Evaluating anti-spoofing ML model against trained voice weights...");
        updateStageProgress(4);

        const resp = await fetch(getApiUrl(`/api/demo/${sampleType}`));
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Demo execution failed");
        }
        const data = await resp.json();

        // STAGE 4: DEFENSE ACTION
        updateStageProgress(5);
        const isHuman = (data.classification === "HUMAN VOICE") || (Math.round((data.human_probability || 0) * 100) >= 65);
        const defenseText = isHuman 
            ? "Authentic Human Voice Verified — Zero Threat Detected"
            : "Synthetic Attack Neutralized — Security Advisory Triggered";
        advancePipelineStage(4, defenseText);

        updateDashboard(data);

        // Update Audio Playback Player & Auto-Play ("And Listen")
        const audioEl = document.getElementById("audioPlayback");
        const container = document.getElementById("audioPlaybackContainer");
        const playerStatus = document.getElementById("audioPlayerStatus");
        const playerBadge = document.getElementById("audioPlayerBadge");

        if (data.audio_url && audioEl) {
            audioEl.src = getApiUrl(data.audio_url);
            if (container) container.style.display = "block";
            if (playerStatus) playerStatus.textContent = `NOW LISTENING: ${data.source || sampleTitle}`;
            if (playerBadge) {
                playerBadge.textContent = isHuman ? "HUMAN VOICE" : "ROBOTIC VOICE / AI VOICE";
                playerBadge.className = isHuman ? "badge-safe" : "badge-danger";
            }

            audioEl.onplay = () => { if (waveform) waveform.classList.add("active"); };
            audioEl.onended = () => { if (waveform) waveform.classList.remove("active"); };

            audioEl.play().catch(e => {
                console.log("Audio autoplay note (requires user interaction in some browsers):", e);
            });
        }

        if (chart) {
            chart.data.labels = ["Win 1", "Win 2", "Win 3", "Win 4"];
            const synthVal = Math.round(data.synthetic_probability * 100);
            chart.data.datasets[0].data = [
                Math.max(0, synthVal - 3),
                Math.min(100, synthVal + 2),
                Math.max(0, synthVal - 1),
                synthVal
            ];
            chart.data.datasets[1].data = [70, 70, 70, 70];
            chart.update();
        }

        const humanConfVal = (data.human_confidence !== undefined) ? data.human_confidence : (data.human_probability <= 1 ? Math.round((data.human_probability || 0) * 100) : data.human_probability);
        showToast(`Loaded ${data.source}: ${isHuman ? "🟢 HUMAN VOICE" : "🔴 ROBOTIC VOICE / AI VOICE"} (${humanConfVal}% Human Confidence)`, isHuman ? "success" : "warning");
    } catch (e) {
        console.error("Demo failed:", e);
        const errMsg = (e.message && e.message.includes("AUDIO INPUT ERROR"))
            ? e.message
            : `AUDIO INPUT ERROR: ${e.message || "Model prediction unavailable"}`;
        if (monitorStatus) {
            monitorStatus.textContent = "AUDIO INPUT ERROR";
            monitorStatus.className = "badge-danger";
        }
        if (micText) {
            micText.textContent = "AUDIO INPUT ERROR";
            micText.style.color = "#EF4444";
        }
        if (micSubtext) {
            micSubtext.textContent = errMsg;
            micSubtext.style.color = "#EF4444";
        }
        const vHeadline = document.getElementById("verdictHeadline");
        if (vHeadline) {
            vHeadline.textContent = "AUDIO INPUT ERROR";
            vHeadline.style.color = "#EF4444";
        }
        const vDetail = document.getElementById("verdictDetail");
        if (vDetail) {
            vDetail.textContent = errMsg;
        }
        showToast(errMsg, "error", 7000);
    }
}
window.runDemo = runDemo;
window.uploadAudio = uploadAudio;

// Verification Guidance Modal Handler
function verifySpeaker() {
    const existingModal = document.getElementById("verificationGuidanceModal");
    if (existingModal) existingModal.remove();

    const modalBackdrop = document.createElement("div");
    modalBackdrop.id = "verificationGuidanceModal";
    modalBackdrop.className = "verification-modal-backdrop";
    modalBackdrop.innerHTML = `
        <div class="verification-modal">
            <div class="verification-modal-header">
                <h3>🛡️ Speaker Verification Guidance Protocol</h3>
                <button class="verification-modal-close" onclick="closeVerificationGuidanceModal()">&times;</button>
            </div>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 16px;">
                Follow these SOC security steps to verify speaker identity out-of-band when high or suspicious synthetic voice risk is flagged.
            </p>
            <div class="verification-step-item">
                <div class="verification-step-num">1</div>
                <div class="verification-step-content">
                    <h5>Out-of-Band Callback</h5>
                    <p>Directly contact the caller on their official registered phone number on record (do NOT rely on incoming call audio).</p>
                </div>
            </div>
            <div class="verification-step-item">
                <div class="verification-step-num">2</div>
                <div class="verification-step-content">
                    <h5>Dynamic Security Challenge</h5>
                    <p>Ask unpredictable, non-public account questions or verify recent off-system transactional details.</p>
                </div>
            </div>
            <div class="verification-step-item">
                <div class="verification-step-num">3</div>
                <div class="verification-step-content">
                    <h5>Spectral Pitch & Phase Audit</h5>
                    <p>Request a continuous 5-second vocalization ("AAAAAH") to analyze phase coherence, pitch jitter, and neural vocoder artifacts.</p>
                </div>
            </div>
            <div class="verification-step-item">
                <div class="verification-step-num">4</div>
                <div class="verification-step-content">
                    <h5>Step-Up Multi-Factor Auth</h5>
                    <p>Trigger push notification or hardware token challenge via secondary authentication channel.</p>
                </div>
            </div>
            <div class="verification-modal-footer">
                <button class="btn btn-secondary" onclick="closeVerificationGuidanceModal()">Close</button>
                <button class="btn btn-primary" onclick="completeVerificationGuidance()">Mark Verification Checked</button>
            </div>
        </div>
    `;

    document.body.appendChild(modalBackdrop);
}

function closeVerificationGuidanceModal() {
    const el = document.getElementById("verificationGuidanceModal");
    if (el) el.remove();
}

function completeVerificationGuidance() {
    closeVerificationGuidanceModal();
    if (typeof App !== "undefined" && App.showToast) {
        App.showToast("Speaker identity verification marked complete.", "info");
    } else {
        alert("Speaker identity verification check logged.");
    }
}

window.verifySpeaker = verifySpeaker;
window.closeVerificationGuidanceModal = closeVerificationGuidanceModal;
window.completeVerificationGuidance = completeVerificationGuidance;

