#!/bin/bash

# Build and run script for hosting services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_status "Docker is available"
}

# Check if NVIDIA Docker is available
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker not available. The service will run on CPU only."
        print_warning "For GPU support, install NVIDIA Container Toolkit:"
        print_warning "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        return 1
    else
        print_status "NVIDIA Docker is available"
        return 0
    fi
}

# Build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t hosting-service .
    print_status "Docker image built successfully"
}

# Run with Docker Compose
run_compose() {
    if check_nvidia_docker; then
        print_status "Starting services with GPU support..."
        docker-compose --profile gpu up hosting-service-gpu --build
    else
        print_status "Starting services with CPU only..."
        docker-compose up hosting-service --build
    fi
}

# Run with Docker Compose in background
run_compose_daemon() {
    if check_nvidia_docker; then
        print_status "Starting services with GPU support in background..."
        docker-compose --profile gpu up -d hosting-service-gpu --build
    else
        print_status "Starting services with CPU only in background..."
        docker-compose up -d hosting-service --build
    fi
    print_status "Services started. Check logs with: docker-compose logs -f"
}

# Stop services
stop_services() {
    print_status "Stopping services..."
    docker-compose down
    print_status "Services stopped"
}

# Show logs
show_logs() {
    print_status "Showing logs..."
    docker-compose logs -f
}

# Test the service
test_service() {
    print_status "Testing service health..."
    sleep 10  # Wait for service to start
    
    if curl -f http://localhost:8001/health &> /dev/null; then
        print_status "Service is healthy!"
        print_status "API available at: http://localhost:8001"
        print_status "API documentation at: http://localhost:8001/docs"
    else
        print_error "Service is not responding. Check logs with: docker-compose logs"
    fi
}

# Clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    print_status "Cleanup completed"
}

# Run CPU-only mode
run_cpu_only() {
    print_status "Starting services with CPU only (forced)..."
    docker-compose up hosting-service --build
}

# Run CPU-only daemon mode
run_cpu_daemon() {
    print_status "Starting services with CPU only in background (forced)..."
    docker-compose up -d hosting-service --build
    print_status "Services started. Check logs with: docker-compose logs -f"
}

# Run GPU mode (force)
run_gpu_only() {
    print_status "Starting services with GPU support (forced)..."
    docker-compose --profile gpu up hosting-service-gpu --build
}

# Run GPU daemon mode (force)
run_gpu_daemon() {
    print_status "Starting services with GPU support in background (forced)..."
    docker-compose --profile gpu up -d hosting-service-gpu --build
    print_status "Services started. Check logs with: docker-compose logs -f"
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       - Build the Docker image"
    echo "  run         - Build and run with Docker Compose (auto-detect GPU)"
    echo "  daemon      - Build and run in background (auto-detect GPU)"
    echo "  run-cpu     - Force CPU-only mode"
    echo "  daemon-cpu  - Force CPU-only daemon mode"
    echo "  run-gpu     - Force GPU mode"
    echo "  daemon-gpu  - Force GPU daemon mode"
    echo "  stop        - Stop all services"
    echo "  logs        - Show service logs"
    echo "  test        - Test service health"
    echo "  cleanup     - Clean up Docker resources"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 run-cpu      # Force CPU mode"
    echo "  $0 daemon-gpu   # Force GPU mode in background"
    echo "  $0 daemon && $0 test"
}

# Main script logic
case "${1:-help}" in
    build)
        check_docker
        build_image
        ;;
    run)
        check_docker
        run_compose
        ;;
    daemon)
        check_docker
        run_compose_daemon
        ;;
    run-cpu)
        check_docker
        run_cpu_only
        ;;
    daemon-cpu)
        check_docker
        run_cpu_daemon
        ;;
    run-gpu)
        check_docker
        run_gpu_only
        ;;
    daemon-gpu)
        check_docker
        run_gpu_daemon
        ;;
    stop)
        stop_services
        ;;
    logs)
        show_logs
        ;;
    test)
        test_service
        ;;
    cleanup)
        cleanup
        ;;
    help|*)
        show_usage
        ;;
esac 