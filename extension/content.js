/**
 * Content Script for Caplify.
 * Injects native HTML5 <video> caption overlays into movie streaming pages.
 */

let captionContainer = null;
let fadeTimeout = null;

function getActiveVideo() {
  const videos = Array.from(document.querySelectorAll("video"));
  if (videos.length === 0) return null;
  // Return playing video or largest visible video
  return videos.find((v) => !v.paused && v.currentTime > 0) || videos[0];
}

function ensureCaptionElement(video) {
  if (captionContainer && document.body.contains(captionContainer)) {
    return captionContainer;
  }
  captionContainer = document.createElement("div");
  captionContainer.className = "caplify-native-caption caplify-hidden";

  const parent = video.parentElement || document.body;
  if (window.getComputedStyle(parent).position === "static") {
    parent.style.position = "relative";
  }
  parent.appendChild(captionContainer);
  return captionContainer;
}

function updateCaption(text, isFinal) {
  const video = getActiveVideo();
  if (!video) return;

  const el = ensureCaptionElement(video);
  el.textContent = text;
  el.classList.remove("caplify-hidden");
  el.classList.toggle("caplify-partial", !isFinal);

  clearTimeout(fadeTimeout);
  fadeTimeout = setTimeout(() => {
    el.classList.add("caplify-hidden");
  }, 3500);
}

function removeCaption() {
  clearTimeout(fadeTimeout);
  if (captionContainer && captionContainer.parentElement) {
    captionContainer.parentElement.removeChild(captionContainer);
    captionContainer = null;
  }
}

// Adapt to full-screen mode on movie streaming platforms
document.addEventListener("fullscreenchange", () => {
  const fsElem = document.fullscreenElement;
  if (fsElem && captionContainer) {
    fsElem.appendChild(captionContainer);
  }
});

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "CAPTION_UPDATE" && message.payload) {
    updateCaption(message.payload.text, message.payload.is_final);
  } else if (message.type === "CAPTION_STOP") {
    removeCaption();
  }
});
