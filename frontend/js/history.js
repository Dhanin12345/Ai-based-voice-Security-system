document.addEventListener("DOMContentLoaded", () => {
  const historyTableBody = document.getElementById("historyTableBody");
  if (!historyTableBody) return;

  const searchInput = document.getElementById("searchInput");
  const filterClassification = document.getElementById("filterClassification");
  const filterRiskLevel = document.getElementById("filterRiskLevel");
  const btnExportCSV = document.getElementById("btnExportCSV");

  let historyData = [];

  async function loadHistory() {
    try {
      const search = searchInput ? searchInput.value : "";
      const classification = filterClassification ? filterClassification.value : "";
      const riskLevel = filterRiskLevel ? filterRiskLevel.value : "";

      const baseUrl = (window.location.protocol === "file:" || !window.location.hostname) ? "http://localhost:8000" : "";
      let url = `${baseUrl}/api/history?limit=100`;
      if (search) url += `&search=${encodeURIComponent(search)}`;
      if (classification) url += `&classification=${encodeURIComponent(classification)}`;
      if (riskLevel) url += `&risk_level=${encodeURIComponent(riskLevel)}`;

      const resp = await fetch(url);
      if (!resp.ok) throw new Error("Failed to load history.");

      const result = await resp.json();
      historyData = result.items || [];
      renderTable(historyData);
    } catch (err) {
      console.error("Error loading history:", err);
      historyTableBody.innerHTML = `<tr><td colspan="9" style="text-align:center; color: var(--status-critical);">Error loading history: ${err.message}</td></tr>`;
    }
  }

  function renderTable(items) {
    if (items.length === 0) {
      historyTableBody.innerHTML = `<tr><td colspan="11" style="text-align:center; color: var(--text-muted);">No detection history records found.</td></tr>`;
      return;
    }

    historyTableBody.innerHTML = items.map(item => {
      const rawHuman = (item.human_probability !== undefined) ? item.human_probability : (item.confidence || 0);
      const humanProbPercent = rawHuman <= 1.0 ? rawHuman * 100 : rawHuman;
      const hProb = (humanProbPercent % 1 !== 0) ? +(humanProbPercent).toFixed(2) : Math.round(humanProbPercent);
      const sProb = Math.round((item.synthetic_probability || (1 - rawHuman)) * 100);

      // Exact 65% threshold
      const isHuman = humanProbPercent >= 65.0;
      const badgeClass = isHuman ? 'badge-safe' : 'badge-danger';
      const levelLabel = isHuman ? 'LOW' : 'HIGH';
      const classText = isHuman ? 'HUMAN VOICE' : 'ROBOTIC VOICE / AI VOICE';
      const classColor = isHuman ? 'var(--status-safe)' : 'var(--status-critical)';

      return `
        <tr>
          <td style="font-family: 'JetBrains Mono', monospace; font-weight: 700;">#${item.analysis_id || item.id}</td>
          <td style="font-family: 'JetBrains Mono', monospace; font-size: 12px;">${formatHistoryTimestamp(item.timestamp)}</td>
          <td><span class="badge-safe">${item.source}</span></td>
          <td><strong style="color: ${classColor};">${classText}</strong></td>
          <td style="font-family: 'JetBrains Mono', monospace; color: var(--status-safe); font-weight: 600;">${hProb}%</td>
          <td style="font-family: 'JetBrains Mono', monospace; color: var(--status-critical); font-weight: 600;">${sProb}%</td>
          <td style="font-family: 'JetBrains Mono', monospace;">${(item.confidence * 100).toFixed(1)}%</td>
          <td style="font-family: 'JetBrains Mono', monospace;"><strong>${item.risk_score} / 100</strong></td>
          <td><span class="${badgeClass}">${levelLabel}</span></td>
          <td><span class="badge-safe">${item.status || 'COMPLETED'}</span></td>
          <td style="white-space: nowrap;">
            <button class="btn-start" style="padding: 4px 10px; font-size: 12px;" onclick="viewReport(${item.analysis_id || item.id})">Report</button>
            <button class="btn-stop" style="padding: 4px 10px; font-size: 12px;" onclick="deleteRecord(${item.analysis_id || item.id})">Delete</button>
          </td>
        </tr>
      `;
    }).join('');
  }

  window.viewReport = function(id) {
    window.location.href = `report.html?id=${id}`;
  };

  window.deleteRecord = async function(id) {
    if (!confirm(`Are you sure you want to delete detection record #${id}?`)) return;

    try {
      const baseUrl = (window.location.protocol === "file:" || !window.location.hostname) ? "http://localhost:8000" : "";
      const resp = await fetch(`${baseUrl}/api/history/${id}`, { method: "DELETE" });
      if (!resp.ok) throw new Error("Delete failed.");
      alert(`Record #${id} deleted successfully.`);
      loadHistory();
    } catch (e) {
      alert("Error: " + e.message);
    }
  };

  window.clearAllHistoryData = async function() {
    if (!confirm("Are you sure you want to clear all detection history logs and security alerts?")) return;

    try {
      const baseUrl = (window.location.protocol === "file:" || !window.location.hostname) ? "http://localhost:8000" : "";
      const resp = await fetch(`${baseUrl}/api/history`, { method: "DELETE" });
      if (!resp.ok) throw new Error("Clear history failed.");
      alert("All detection history and security alerts cleared successfully.");
      loadHistory();
    } catch (e) {
      alert("Error: " + e.message);
    }
  };


  if (searchInput) searchInput.addEventListener("input", loadHistory);
  if (filterClassification) filterClassification.addEventListener("change", loadHistory);
  if (filterRiskLevel) filterRiskLevel.addEventListener("change", loadHistory);

  if (btnExportCSV) {
    btnExportCSV.addEventListener("click", () => {
      if (historyData.length === 0) {
        alert("No data to export.");
        return;
      }
      let csv = "AnalysisID,Timestamp,Source,Filename,Classification,HumanProbability,SyntheticProbability,Confidence,RiskScore,RiskLevel,Status\n";
      historyData.forEach(item => {
        csv += `${item.analysis_id || item.id},"${item.timestamp}",${item.source},"${item.filename || ''}",${item.classification},${item.human_probability || 0},${item.synthetic_probability || 0},${item.confidence},${item.risk_score},${item.risk_level},${item.status || 'COMPLETED'}\n`;
      });
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `VoiceGuard_History_${Date.now()}.csv`;
      a.click();
    });
  }

  function formatHistoryTimestamp(ts) {
    if (!ts) return "---";
    let d = (typeof ts === "string" && !ts.endsWith("Z") && !ts.includes("+"))
      ? new Date(ts + "Z")
      : new Date(ts);
    return isNaN(d.getTime()) ? ts : d.toLocaleString();
  }

  loadHistory();
});
