# -------- AcuityCheck Dockerfile (CPU) --------
ARG PY_VER=3.11
FROM python:${PY_VER}-slim AS app

# System deps (minimal). `libglib2.0-0` for OpenCV; `curl` for model download.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libgl1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from writing .pyc + make logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install Python deps  
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Fetch ONNX models at build time 
RUN chmod +x scripts/download_models.sh && \
    ./scripts/download_models.sh || true

EXPOSE 8501

# Lightweight healthcheck 
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
