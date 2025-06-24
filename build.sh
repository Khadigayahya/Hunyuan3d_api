#!/bin/bash

# Hunyuan3D-2 API Build Script
# This script builds the Docker image and prepares the environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="hunyuan3d-api"
IMAGE_TAG="latest"
CONTAINER_NAME="hunyuan3d-container"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log_info "Docker is installed"
}

# Check if NVIDIA Docker is available (for GPU support)
check_nvidia_docker() {
    if command -v nvidia-docker &> /dev/null; then
        log_info "NVIDIA Docker is available"
        return 0
    elif docker info | grep -q nvidia; then
        log_info "NVIDIA Docker runtime is available"
        return 0
    else
        log_warn "NVIDIA Docker is not available. GPU acceleration will not be available."
        return 1
    fi
}

# Build Docker image
build_image() {
    log_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    
    # Build the image
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} . \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Test the built image
test_image() {
    log_info "Testing the built image..."
    
    # Run a quick test
    docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"
    
    if [ $? -eq 0 ]; then
        log_info "Image test passed"
    else
        log_error "Image test failed"
        exit 1
    fi
}

# Create run script
create_run_script() {
    log_info "Creating run script..."
    
    cat > run_container.sh << 'EOF'
#!/bin/bash

# Run script for Hunyuan3D API container

IMAGE_NAME="hunyuan3d-api"
IMAGE_TAG="latest"
CONTAINER_NAME="hunyuan3d-container"

# Check if container is already running
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Stopping existing container..."
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

# Remove existing container if exists
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Removing existing container..."
    docker rm ${CONTAINER_NAME}
fi

# Check for GPU support
if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
    echo "Running with GPU support..."
    docker run -d \
        --name ${CONTAINER_NAME} \
        --gpus all \
        -p 8080:8080 \
        -v $(pwd)/outputs:/tmp/outputs \
        -v $(pwd)/models:/app/models \
        --restart unless-stopped \
        --memory=8g \
        --memory-swap=16g \
        --shm-size=2g \
        ${IMAGE_NAME}:${IMAGE_TAG}
else
    echo "Running without GPU support..."
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p 8080:8080 \
        -v $(pwd)/outputs:/tmp/outputs \
        -v $(pwd)/models:/app/models \
        --restart unless-stopped \
        --memory=4g \
        --memory-swap=8g \
        --shm-size=1g \
        ${IMAGE_NAME}:${IMAGE_TAG}
fi

# Check if container started successfully
sleep 5
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container ${CONTAINER_NAME} started successfully"
    echo "API is available at: http://localhost:8080"
    echo "Health check: http://localhost:8080/health"
    echo "API docs: http://localhost:8080/docs"
else
    echo "Failed to start container. Check logs with: docker logs ${CONTAINER_NAME}"
    exit 1
fi
EOF

    chmod +x run_container.sh
    log_info "Run script created: run_container.sh"
}

# Create stop script
create_stop_script() {
    log_info "Creating stop script..."
    
    cat > stop_container.sh << 'EOF'
#!/bin/bash

CONTAINER_NAME="hunyuan3d-container"

echo "Stopping container ${CONTAINER_NAME}..."
docker stop ${CONTAINER_NAME}

echo "Removing container ${CONTAINER_NAME}..."
docker rm ${CONTAINER_NAME}

echo "Container stopped and removed"
EOF

    chmod +x stop_container.sh
    log_info "Stop script created: stop_container.sh"
}

# Create monitoring script
create_monitoring_script() {
    log_info "Creating monitoring script..."
    
    cat > monitor.sh << 'EOF'
#!/bin/bash

CONTAINER_NAME="hunyuan3d-container"

# Function to check container health
check_health() {
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo "Container Status: Running"
        
        # Check API health
        if curl -f http://localhost:8080/health > /dev/null 2>&1; then
            echo "API Status: Healthy"
        else
            echo "API Status: Unhealthy"
        fi
        
        # Show resource usage
        echo "Resource Usage:"
        docker stats ${CONTAINER_NAME} --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"
    else
        echo "Container Status: Not Running"
    fi
}

# Function to show logs
show_logs() {
    echo "Recent logs:"
    docker logs --tail 50 ${CONTAINER_NAME}
}

case "$1" in
    "health")
        check_health
        ;;
    "logs")
        show_logs
        ;;
    "follow")
        docker logs -f ${CONTAINER_NAME}
        ;;
    *)
        echo "Usage: $0 {health|logs|follow}"
        echo "  health  - Check container and API health"
        echo "  logs    - Show recent logs"
        echo "  follow  - Follow logs in real-time"
        ;;
esac
EOF

    chmod +x monitor.sh
    log_info "Monitoring script created: monitor.sh"
}

# Main build process
main() {
    log_info "Starting Hunyuan3D API build process..."
    
    # Check prerequisites
    check_docker
    check_nvidia_docker
    
    # Build image
    build_image
    
    # Test image
    test_image
    
    # Create utility scripts
    create_run_script
    create_stop_script
    create_monitoring_script
    
    # Create output directories
    mkdir -p outputs models
    
    log_info "Build process completed successfully!"
    log_info ""
    log_info "Next steps:"
    log_info "1. Run the container: ./run_container.sh"
    log_info "2. Check health: ./monitor.sh health"
    log_info "3. View logs: ./monitor.sh logs"
    log_info "4. Stop container: ./stop_container.sh"
    log_info ""
    log_info "API will be available at: http://localhost:8080"
    log_info "API documentation: http://localhost:8080/docs"
}

#