import pytest
import numpy as np
from viki.viz.mjpeg import encode_jpeg, mjpeg_chunk, placeholder


def test_encode_jpeg():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    data = encode_jpeg(img)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_mjpeg_chunk():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    chunk = mjpeg_chunk(img)
    assert isinstance(chunk, bytes)
    assert b"--frame" in chunk
    assert b"Content-Type: image/jpeg" in chunk
    assert b"Content-Length: " in chunk


def test_placeholder():
    w, h = 640, 480
    text = "Test Placeholder"
    img = placeholder(w, h, text)
    assert img.shape == (h, w, 3)
    assert img.dtype == np.uint8
    # The image should be mostly black (0)
    assert np.mean(img) < 10
