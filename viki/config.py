"""
viki.config
-----------
Centralised tunables for the ViKi capture server.
Loaded from data/user_configuration.json.
"""

import json
import os
import shutil

DEFAULT_CONFIG_PATH = "data/default_configuration.json"
USER_CONFIG_PATH = "data/user_configuration.json"

def _load_config():
    if not os.path.exists(USER_CONFIG_PATH):
        if os.path.exists(DEFAULT_CONFIG_PATH):
            shutil.copy(DEFAULT_CONFIG_PATH, USER_CONFIG_PATH)
        else:
            # Fallback if even default is missing (shouldn't happen with our setup)
            return {}

    with open(USER_CONFIG_PATH, "r") as f:
        return json.load(f)

_config = _load_config()

# We assign these to globals so that 'from viki.config import CONSTANT' still works
globals().update(_config)

# Keep a reference to the paths for the API
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH
USER_CONFIG_PATH = USER_CONFIG_PATH

