/**
 * Background Service Worker for Caplify Chrome Extension.
 * Manages tab audio capture, WebSocket connection, and caption distribution.
 */

const WS_URL = "ws://localhost:8000/ws";

let isCapturing = false;
let currentMode = "vosk";
let socket = null;
let audioCtx = null;
let streamSource = null;
let workletNode = null;
let activeTabId = null;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "START_CAPTURE") {
    handleStartCapture(message.tabId)
      .then(() => sendResponse({ success: true }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }
  if (message.type === "STOP_CAPTURE") {
    handleStopCapture();
    sendResponse({ success: true });
    return true;
  }
  if (message.type === "SET_MODE") {
    currentMode = message.mode;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "config", mode: currentMode }));
    }
    sendResponse({ success: true, mode: currentMode });
    return true;
  }
  if (message.type === "GET_STATUS") {
    sendResponse({
      isCapturing,
      currentMode,
      wsConnected: socket && socket.readyState === WebSocket.OPEN,
    });
    return true;
  }
});

async function handleStartCapture(tabId) {
  if (isCapturing) return;
  activeTabId = tabId;

  chrome.tabCapture.capture({ audio: true, video: false }, async (stream) => {
    if (!stream) {
      console.error("Failed to capture tab audio:", chrome.runtime.lastError);
      return;
    }
    await setupAudioPipeline(stream);
    setupWebSocket();
    isCapturing = true;
  });
}

async function setupAudioPipeline(stream) {
  audioCtx = new AudioContext();
  streamSource = audioCtx.createMediaStreamSource(stream);

  // Play tab audio to local destination so speaker audio is not muted
  streamSource.connect(audioCtx.destination);

  await audioCtx.audioWorklet.addModule(chrome.runtime.getURL("audio-processor.js"));
  workletNode = new AudioWorkletNode(audioCtx, "caplify-audio-processor");

  workletNode.port.onmessage = (event) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(event.data);
    }
  };

  streamSource.connect(workletNode);
}

function setupWebSocket() {
  socket = new WebSocket(WS_URL);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    console.log("WebSocket connected to Caplify backend");
    socket.send(JSON.stringify({ type: "config", mode: currentMode }));
  };

  socket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "caption" && activeTabId) {
        chrome.tabs.sendMessage(activeTabId, { type: "CAPTION_UPDATE", payload }).catch(() => {});
        chrome.runtime.sendMessage({ type: "LATENCY_UPDATE", latency: payload.latency_ms }).catch(() => {});
      }
    } catch (err) {
      console.error("Error parsing WebSocket caption payload:", err);
    }
  };

  socket.onclose = () => {
    console.log("WebSocket connection closed");
  };
}

function handleStopCapture() {
  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
  if (streamSource) {
    streamSource.disconnect();
    streamSource = null;
  }
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  if (socket) {
    socket.close();
    socket = null;
  }
  isCapturing = false;
  if (activeTabId) {
    chrome.tabs.sendMessage(activeTabId, { type: "CAPTION_STOP" }).catch(() => {});
  }
}
