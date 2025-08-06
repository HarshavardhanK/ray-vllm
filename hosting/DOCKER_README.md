# Docker Setup for Hosting Services

This directory contains Docker configuration for the hosting services that provide:
- Cross-Encoder Reranking
- Prompt Guard (Malicious Content Detection)
- Text Embeddings (Sentence Transformers & vLLM optimized)

## Prerequisites

### System Requirements
- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (recommended 16GB+ VRAM)
- NVIDIA Container Toolkit installed
- At least 32GB system RAM
- Linux OS (recommended)

### Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and run the service
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t hosting-service .

# Run with GPU support
docker run --gpus all -p 8001:8001 hosting-service

# Run with specific GPU
docker run --gpus '"device=0"' -p 8001:8001 hosting-service
```

## Configuration

### Environment Variables

You can customize the service behavior using environment variables:

```yaml
# In docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Specify GPU device
  - USE_FLASH_ATTENTION_2=false  # Disable flash attention
  - TRANSFORMERS_ATTENTION_IMPLEMENTATION=sdpa  # Use PyTorch SDPA
```

### GPU Configuration

The service automatically detects available GPUs. You can specify which GPU to use:

```bash
# Use specific GPU
docker run --gpus '"device=0"' -p 8001:8001 hosting-service

# Use multiple GPUs
docker run --gpus '"device=0,1"' -p 8001:8001 hosting-service
```

### Model Caching

To cache downloaded models and avoid re-downloading:

```bash
# Create a models directory
mkdir -p models

# Mount it in docker-compose.yml (already configured)
volumes:
  - ./models:/app/models
```

## API Endpoints

Once running, the service will be available at `http://localhost:8001`

### Available Endpoints:

- `GET /health` - Health check
- `POST /rerank` - Document reranking
- `POST /validate/input` - Prompt validation
- `POST /embed` - Text embeddings (Sentence Transformers)
- `POST /vllm/embed` - Text embeddings (vLLM optimized)
- `POST /vllm/embed/documents` - Document embeddings
- `POST /vllm/embed/queries` - Query embeddings
- `POST /embed/queries-documents` - Query-document similarity
- `POST /vllm/embed/queries-documents` - vLLM query-document similarity
- `GET /embed/info` - Model information

## Testing

### Test the service is running:

```bash
curl http://localhost:8001/health
```

### Test embedding service:

```bash
curl -X POST "http://localhost:8001/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world"],
    "task_description": "General text embedding",
    "max_length": 8192,
    "normalize": true
  }'
```

### Test reranking:

```bash
curl -X POST "http://localhost:8001/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "passages": [
      "Machine learning is a subset of AI",
      "Python is a programming language",
      "Deep learning uses neural networks"
    ]
  }'
```

### Test prompt guard:

```bash
curl -X POST "http://localhost:8001/validate/input" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is the capital of France?"
  }'
```

## Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   # Check if nvidia-docker is working
   docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
   ```

2. **Out of memory**
   - Reduce batch sizes in the API calls
   - Use CPU-only mode by setting `CUDA_VISIBLE_DEVICES=""`
   - Increase system RAM or use a GPU with more VRAM

3. **Model download issues**
   - Check internet connection
   - Ensure sufficient disk space
   - Try running with `--network host` for better network access

4. **Service not starting**
   ```bash
   # Check logs
   docker-compose logs hosting-service
   
   # Check if port is available
   netstat -tulpn | grep 8001
   ```

### Performance Optimization

1. **For production use:**
   - Use vLLM endpoints (`/vllm/embed/*`) for better performance
   - Ensure GPU has sufficient VRAM (16GB+ recommended)
   - Use model caching to avoid re-downloading

2. **For development:**
   - Use CPU mode for testing without GPU
   - Reduce model sizes in config.py for faster loading

## Development

### Building for different architectures:

```bash
# Build for specific platform
docker build --platform linux/amd64 -t hosting-service .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t hosting-service .
```

### Adding new models:

1. Update `config.py` with new model configurations
2. Add corresponding service files
3. Update `main.py` with new endpoints
4. Rebuild the Docker image

## Security Considerations

- The container runs as a non-root user (`appuser`)
- Health checks are configured for monitoring
- Environment variables are used for configuration
- Model files are cached in a separate volume

## Monitoring

The service includes health checks that can be monitored:

```bash
# Check container health
docker inspect hosting-service | grep Health -A 10

# Monitor resource usage
docker stats hosting-service
```

## Support

For issues related to:
- Docker setup: Check this README and Docker logs
- Model performance: Refer to the main README.md
- API usage: Check the API documentation in README.md 