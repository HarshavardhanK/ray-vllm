#!/bin/bash

#Check if model name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 meta-llama/Llama-2-7b-chat-hf"
    exit 1
fi

MODEL_NAME=$1

#Check if huggingface-cli is logged in
if ! huggingface-cli whoami &>/dev/null; then
    echo "HuggingFace login required. Please login:"
    huggingface-cli login
fi

#Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vllm-env

#Start vLLM server
echo "Starting vLLM server with model: $MODEL_NAME"
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype auto

#Handle script interruption
trap 'echo "Shutting down vLLM server..."; exit' INT TERM 