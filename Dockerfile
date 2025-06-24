# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support (specific version for compatibility)
RUN pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Clone and install Hunyuan3D-2
RUN git clone https://github.com/Tencent/Hunyuan3D-2.git /tmp/Hunyuan3D-2
WORKDIR /tmp/Hunyuan3D-2
RUN python setup.py install

# Clone and install ComfyUI-Hunyuan3DWrapper
RUN git clone https://github.com/kijai/ComfyUI-Hunyuan3DWrapper.git /tmp/ComfyUI-Hunyuan3DWrapper
WORKDIR /tmp/ComfyUI-Hunyuan3DWrapper
RUN pip install -r requirements.txt

# Build and install custom rasterizer
WORKDIR /tmp/ComfyUI-Hunyuan3DWrapper/hy3dgen/texgen/custom_rasterizer/
RUN python setup.py bdist_wheel
RUN pip install dist/custom_rasterizer*.whl

# Return to app directory
WORKDIR /app

# Copy application files
COPY app.py .
COPY model_utils.py .
COPY config.py .

# Create necessary directories
RUN mkdir -p /tmp/outputs
RUN mkdir -p /app/models
RUN mkdir -p /app/cache

# Set permissions
RUN chmod +x /app/app.py

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]