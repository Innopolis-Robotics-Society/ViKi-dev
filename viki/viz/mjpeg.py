"""
viki.viz.mjpeg
--------------
MJPEG transport helpers: JPEG encoding, multipart chunk framing, and a
text placeholder image. No camera or web-framework dependencies.
"""
from __future__ import annotations

import cv2
import numpy as np

from viki.config import JPEG_QUALITY


def encode_jpeg(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode a BGR image as JPEG bytes."""
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()


def mjpeg_chunk(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Frame a single image as one multipart/x-mixed-replace MJPEG chunk."""
    data = encode_jpeg(img, quality)
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(data)).encode()
        + b"\r\n\r\n" + data + b"\r\n"
    )


def placeholder(w: int, h: int, text: str) -> np.ndarray:
    """A black image with centred grey text — shown when no frame is available."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (80, 80, 80), 2, cv2.LINE_AA)
    return img
