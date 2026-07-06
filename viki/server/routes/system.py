from fastapi import APIRouter, HTTPException
import json
import os
import shutil
import logging
from viki.config import USER_CONFIG_PATH, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["system"])

# Default upload directory (same as SkeletonRecorder)
DEFAULT_UPLOAD_DIR = Path("data/skeleton_recs")

def get_upload_dir() -> Path:
    """Return the upload directory from config if set, else default."""
    try:
        if USER_CONFIG_PATH.exists():
            with open(USER_CONFIG_PATH) as f:
                cfg = json.load(f)
            custom = cfg.get("upload_dir")
            if custom:
                return Path(custom)
    except Exception:
        pass
    return DEFAULT_UPLOAD_DIR


@router.get("/config")
async def get_config():
    if not os.path.exists(USER_CONFIG_PATH):
        raise HTTPException(status_code=404, detail="User configuration file not found")
    with open(USER_CONFIG_PATH, "r") as f:
        return json.load(f)

@router.post("/config")
async def save_config(config: dict):
    try:
        with open(USER_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config/reset")
async def reset_config():
    try:
        if not os.path.exists(DEFAULT_CONFIG_PATH):
            raise HTTPException(status_code=404, detail="Default configuration file not found")
        shutil.copy(DEFAULT_CONFIG_PATH, USER_CONFIG_PATH)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to reset config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restart")
async def restart_server():
    logger.info("Restarting server via API request...")
    # os._exit(1) is used to kill the python process immediately.
    # Since the container is set to restart: unless-stopped, Docker will restart it.
    os._exit(1)


@router.post("/upload-h5")
async def upload_h5(file: UploadFile = File(...)):
    """
    Upload an HDF5 recording file (.h5 or .hdf5) and save it as rec_{timestamp}.h5.
    """
    # Validate file extension
    if not (file.filename.endswith('.h5') or file.filename.endswith('.hdf5')):
        raise HTTPException(400, "Only HDF5 files (.h5 or .hdf5) are allowed")

    upload_dir = get_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename with current timestamp
    timestamp = int(time.time())
    new_filename = f"rec_{timestamp}.h5"
    file_path = upload_dir / new_filename

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(500, f"Failed to save file: {str(e)}")

    return {
        "status": "success",
        "filename": new_filename,
        "path": str(file_path.relative_to(Path.cwd()))
    }


@router.get("/download-latest-h5")
async def download_latest_h5():
    """
    Download the most recent HDF5 recording (by timestamp) from the upload directory.
    """
    upload_dir = get_upload_dir()
    if not upload_dir.exists():
        raise HTTPException(404, "No recordings directory found")

    files = list(upload_dir.glob("rec_*.h5"))
    if not files:
        raise HTTPException(404, "No HDF5 recording files found")

    # Extract timestamp from filename (rec_123456.h5)
    def extract_ts(path: Path) -> int:
        try:
            return int(path.stem.split('_')[1])
        except (IndexError, ValueError):
            return 0

    latest = max(files, key=extract_ts)
    return FileResponse(latest, filename=latest.name)
