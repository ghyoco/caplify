/**
 * UI controller for Caplify extension popup.
 */

document.addEventListener("DOMContentLoaded", () => {
  const toggleBtn = document.getElementById("toggleBtn");
  const statusBadge = document.getElementById("statusBadge");
  const latencyDisplay = document.getElementById("latencyDisplay");
  const modeRadios = document.querySelectorAll("input[name='asrMode']");

  chrome.runtime.sendMessage({ type: "GET_STATUS" }, (state) => {
    if (!state) return;
    updateUI(state.isCapturing, state.currentMode);
  });

  toggleBtn.addEventListener("click", async () => {
    chrome.runtime.sendMessage({ type: "GET_STATUS" }, async (state) => {
      if (state && state.isCapturing) {
        chrome.runtime.sendMessage({ type: "STOP_CAPTURE" }, () => updateUI(false));
      } else {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab || !tab.id) return;
        chrome.runtime.sendMessage({ type: "START_CAPTURE", tabId: tab.id }, () => updateUI(true));
      }
    });
  });

  modeRadios.forEach((radio) => {
    radio.addEventListener("change", (e) => {
      chrome.runtime.sendMessage({ type: "SET_MODE", mode: e.target.value });
    });
  });

  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "LATENCY_UPDATE" && message.latency !== undefined) {
      latencyDisplay.textContent = `${message.latency} ms`;
      latencyDisplay.style.color = message.latency < 200 ? "#22c55e" : "#f59e0b";
    }
  });

  function updateUI(isCapturing, activeMode) {
    if (isCapturing) {
      statusBadge.textContent = "Capturing";
      statusBadge.className = "status-badge status-active";
      toggleBtn.textContent = "Stop Captioning";
      toggleBtn.className = "btn-toggle btn-stop";
    } else {
      statusBadge.textContent = "Idle";
      statusBadge.className = "status-badge";
      toggleBtn.textContent = "Start Captioning";
      toggleBtn.className = "btn-toggle";
    }
    if (activeMode) {
      const targetRadio = document.querySelector(`input[name='asrMode'][value='${activeMode}']`);
      if (targetRadio) targetRadio.checked = true;
    }
  }
});
