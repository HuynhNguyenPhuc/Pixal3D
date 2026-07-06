FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL name="pixal3d-api" \
      maintainer="pixal3d-api" \
      description="Pixal3D API Server with CUDA support"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0" \
    PYOPENGL_PLATFORM=egl \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    CUDA_MODULE_LOADING=LAZY \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Compilers & Build Tools
    build-essential \
    cmake \
    ninja-build \
    # Python
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    # Graphics & OpenGL
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-utils-extra \
    # X11 & rendering
    libxrender1 \
    libxrender-dev \
    libxi6 \
    libxext6 \
    libsm6 \
    libxkbcommon-x11-0 \
    libgconf-2-4 \
    # Additional libraries
    libglib2.0-0 \
    libeigen3-dev \
    libjpeg-dev \
    libwebp-dev \
    # Utilities
    git \
    git-lfs \
    wget \
    curl \
    unzip \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built requirements first to optimize Docker layer caching
COPY requirements-hfdemo.txt /app/

# Install Python packages using pre-built binary wheels (including o_voxel, cumesh, etc.)
RUN pip install --no-cache-dir -r requirements-hfdemo.txt

# Build and install natten directly (optimized for A100 by targeting CC 8.0)
ARG NATTEN_CUDA_ARCH="8.0"
RUN NATTEN_CUDA_ARCH="${NATTEN_CUDA_ARCH}" NATTEN_N_WORKERS=$(nproc) pip install natten==0.21.0 --no-build-isolation

# Install utils3d as specified in the project documentation
RUN pip install --no-cache-dir https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# Copy only the necessary folders and files for API and Worker to run
COPY app.py app_state.py config.py constants.py inference.py /app/
COPY pixal3d /app/pixal3d
COPY broker /app/broker
COPY controllers /app/controllers
COPY keys /app/keys
COPY routes /app/routes
COPY schemas /app/schemas
COPY utilities /app/utilities
COPY worker /app/worker

# Create cache directory
RUN mkdir -p /app/gradio_cache

# Expose API port
EXPOSE 8081

# Runtime environment variables (non-sensitive defaults)
ENV MAX_IMAGE_SIZE=8388608 \
    REQUIRE_REDIS=true \
    STATUS_TTL=7200 \
    GENERATION_TIMEOUT_SECONDS=3600 \
    REDIS_HOST=localhost \
    REDIS_PORT=6379 \
    REDIS_DB=0 \
    ATTN_BACKEND=flash_attn_3 \
    DISABLE_TQDM=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Default command
CMD ["python", "app.py", "--host=0.0.0.0", "--port=8081", "--cache-path=/app/gradio_cache"]
