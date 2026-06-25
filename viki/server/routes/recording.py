from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from viki.capture.manager import CameraManager
from viki.capture.recorder import RGBDRecorder
from viki.server.deps import get_manager
import os

router = APIRouter(prefix="/api/record", tags=["recording"])

class RecordRequest(BaseModel):
    duration: float = 10.0
    fps: int = 15
    output_dir: str = "data/videos"

def run_recording(manager: CameraManager, req: RecordRequest):
    try:
        recorder = RGBDRecorder(manager, output_base_dir=req.output_dir)
        recorder.record(duration_s=req.duration, sync_fps=req.fps)
        print(f"Background recording finished: {recorder.current_recording_dir}")
    except Exception as e:
        print(f"Background recording error: {e}")

@router.post("/start")
async def start_recording(
    req: RecordRequest,
    background_tasks: BackgroundTasks,
    mgr: CameraManager = Depends(get_manager),
):
    if not mgr.active_device_ids():
        raise HTTPException(status_code=400, detail="No cameras are currently active. Start cameras first.")
    
    background_tasks.add_task(run_recording, mgr, req)
    return {"status": "recording started in background", "duration": req.duration, "fps": req.fps}
