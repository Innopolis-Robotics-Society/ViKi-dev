"""
viki.server.routes.skeleton
--------------------------
Endpoints for controlling skeleton estimation and recording, 
and a WebSocket for streaming the latest skeleton frame.
"""

from __future__ import annotations

import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from viki.server.deps import get_worker
from viki.server.skeleton_worker import SkeletonWorker

router = APIRouter(prefix="/api/skeleton", tags=["skeleton"])

class ToggleRequest(BaseModel):
    enabled: bool

@router.post("/toggle")
async def toggle_estimation(req: ToggleRequest, worker: SkeletonWorker = Depends(get_worker)):
    worker.set_enabled(req.enabled)
    return {"status": "updated", "enabled": worker.is_enabled}

@router.post("/record")
async def toggle_recording(req: ToggleRequest, worker: SkeletonWorker = Depends(get_worker)):
    worker.set_recording(req.enabled)
    return {"status": "updated", "recording": worker.is_recording}

@router.get("/status")
async def get_status(worker: SkeletonWorker = Depends(get_worker)):
    return {
        "enabled": worker.is_enabled,
        "recording": worker.is_recording,
    }

@router.websocket("/stream")
async def skeleton_stream(websocket: WebSocket):
    await websocket.accept()
    worker: SkeletonWorker = websocket.app.state.skeleton_worker
    try:
        while True:
            frame = worker.get_latest_frame()
            if frame:
                # Serialize SkeletonFrame to dict
                data = {
                    "ts": frame.timestamp_us,
                    "landmarks": frame.landmarks.tolist(),
                    "source": [str(s) for s in frame.source],
                    "confidence": frame.confidence.tolist(),
                    "origin": [str(o) for o in frame.origin],
                }
                await websocket.send_json(data)
            
            # Stream at ~20 fps
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
