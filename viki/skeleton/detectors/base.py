

from dataclasses import dataclass
import numpy as np


@dataclass
class PartialDetection2D:
    '''Describes a partial Detection. Can be used for modular detection.'''
    indices: tuple[int, ...]
    px: np.ndarray                  # (k, 2) float32
    lm_z_rel: np.ndarray          # (k,)  float32 
    per_index_confidence: np.ndarray  # (k,) float32
    timestamp_us: int
    device_id: str

