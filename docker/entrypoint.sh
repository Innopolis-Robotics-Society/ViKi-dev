#!/usr/bin/env bash
# docker/entrypoint.sh
# Runs inside the container before starting the server.
# Sets up permissions that cannot be baked into the image.

set -e

# DRI access for Kinect depth engine (OpenGL)
chmod a+rw /dev/dri/renderD* 2>/dev/null || true
chmod a+rw /dev/dri/card*    2>/dev/null || true

# USB access for cameras
chmod a+rw /dev/bus/usb/*/*  2>/dev/null || true

exec "$@"