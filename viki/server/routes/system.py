from fastapi import APIRouter, HTTPException
import json
import os
import shutil
import logging
from viki.config import USER_CONFIG_PATH, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["system"])

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
