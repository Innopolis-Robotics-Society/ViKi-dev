# docker/Dockerfile.frontend
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget ca-certificates gnupg2 apt-transport-https \
    libglib2.0-0 \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir pip-tools && \
    pip-compile --extra frontend pyproject.toml -o requirements-frontend.txt && \
    pip install --no-cache-dir -r requirements-frontend.txt

RUN mkdir -p viki && touch viki/__init__.py
RUN pip install --no-deps --no-cache-dir -e .

# Copy real source
COPY ./viki/ ./viki/


CMD ["/bin/bash"]
