"""Caplify FastAPI WebSocket Backend Server.

Provides a real-time streaming WebSocket endpoint (ws://localhost:8000/ws)
for ingesting PCM 16kHz audio from the Chrome extension and returning
instant captions with latency tracking, plus a health-check endpoint.
"""

import json
import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.pipeline import DualModePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("caplify.server")

app = FastAPI(
    title="Caplify Real-Time Captioning Backend",
    version="1.0.0",
    description="WebSocket server for streaming live captions and noise suppression."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()
DEFAULT_PIPELINE = DualModePipeline()


@app.get("/")
def read_root() -> Dict[str, str]:
    """Root endpoint verifying server availability."""
    return {"service": "Caplify Live Captioning", "status": "running"}


@app.get("/health")
def read_health() -> JSONResponse:
    """System health inspection returning loaded models and active configurations."""
    pipeline_status = DEFAULT_PIPELINE.get_status()
    payload = {
        "status": "healthy",
        "uptime_sec": round(time.time() - START_TIME, 2),
        "active_mode": pipeline_status["mode"],
        "preemphasis": pipeline_status["preemphasis_enabled"],
        "vad_threshold": pipeline_status["vad_threshold"],
        "models": pipeline_status["models_loaded"],
    }
    return JSONResponse(content=payload)


async def handle_control_message(
    websocket: WebSocket,
    pipeline: DualModePipeline,
    raw_text: str
) -> None:
    """Parse and apply JSON control messages to configure the pipeline session."""
    try:
        data = json.loads(raw_text)
        msg_type = data.get("type", "")
        if msg_type == "config":
            if "mode" in data:
                pipeline.set_mode(data["mode"])
            if "vad_threshold" in data:
                pipeline.set_vad_threshold(float(data["vad_threshold"]))
            if "preemphasis" in data:
                pipeline.set_preemphasis(bool(data["preemphasis"]))
            ack = {"type": "ack", "status": pipeline.get_status()}
            await websocket.send_text(json.dumps(ack))
        elif msg_type == "reset":
            pipeline.reset()
            await websocket.send_text(json.dumps({"type": "ack", "status": "reset"}))
    except Exception as exc:
        logger.error(f"Error handling control message: {exc}")


async def handle_binary_audio(
    websocket: WebSocket,
    pipeline: DualModePipeline,
    audio_bytes: bytes
) -> None:
    """Process incoming PCM audio chunk through pipeline and stream caption."""
    try:
        caption = pipeline.process_pcm_chunk(audio_bytes)
        if caption is not None:
            await websocket.send_text(json.dumps(caption))
    except Exception as exc:
        logger.error(f"Error processing audio chunk: {exc}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Bidirectional WebSocket endpoint for streaming PCM audio and control."""
    await websocket.accept()
    session_pipeline = DualModePipeline()
    logger.info("New WebSocket client connected to Caplify captioning stream.")
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                await handle_binary_audio(websocket, session_pipeline, message["bytes"])
            elif "text" in message and message["text"]:
                await handle_control_message(websocket, session_pipeline, message["text"])
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally.")
    except Exception as exc:
        logger.error(f"Unexpected WebSocket session termination: {exc}")


def main() -> None:
    """Run Caplify FastAPI server with Uvicorn when executed directly."""
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
