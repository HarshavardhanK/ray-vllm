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

#Function to get available GPU memory in MB
get_gpu_memory() {
    local gpu_id=$1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id | head -1
}

#Function to get total GPU memory in MB
get_total_gpu_memory() {
    local gpu_id=$1
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu_id | head -1
}

#Function to get GPU utilization percentage
get_gpu_utilization() {
    local gpu_id=$1
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id | head -1
}

#Function to check if GPU is available (not too busy)
is_gpu_available() {
    local gpu_id=$1
    local utilization=$(get_gpu_utilization $gpu_id)
    #Consider GPU available if utilization is less than 80%
    [ $utilization -lt 80 ]
}

#Function to estimate model memory requirements
estimate_model_memory() {
    local model_name=$1
    local context_length=32768  #Default context length
    
    #Extract model size from name
    local model_size=0
    if [[ $model_name == *"17b"* ]] || [[ $model_name == *"17B"* ]]; then
        model_size=17
    elif [[ $model_name == *"13b"* ]] || [[ $model_name == *"13B"* ]]; then
        model_size=13
    elif [[ $model_name == *"7b"* ]] || [[ $model_name == *"7B"* ]]; then
        model_size=7
    elif [[ $model_name == *"3b"* ]] || [[ $model_name == *"3B"* ]]; then
        model_size=3
    else
        model_size=7  #Default assumption
    fi
    
    #Calculate memory using OSC formula: 2x model parameters + 1x context length (in GB)
    local model_weights=$((model_size * 2))
    local context_overhead=$((context_length / 1000))
    local total_gb=$((model_weights + context_overhead))
    local total_mb=$((total_gb * 1024))
    
    echo $total_mb
}

#Function to estimate memory with custom context length
estimate_model_memory_with_context() {
    local model_name=$1
    local context_length=$2
    
    #Extract model size from name
    local model_size=0
    if [[ $model_name == *"17b"* ]] || [[ $model_name == *"17B"* ]]; then
        model_size=17
    elif [[ $model_name == *"13b"* ]] || [[ $model_name == *"13B"* ]]; then
        model_size=13
    elif [[ $model_name == *"7b"* ]] || [[ $model_name == *"7B"* ]]; then
        model_size=7
    elif [[ $model_name == *"3b"* ]] || [[ $model_name == *"3B"* ]]; then
        model_size=3
    else
        model_size=7
    fi
    
    #Calculate memory using OSC formula
    local model_weights=$((model_size * 2))
    local context_overhead=$((context_length / 1000))
    local total_gb=$((model_weights + context_overhead))
    local total_mb=$((total_gb * 1024))
    
    echo $total_mb
}

#Function to get max_position_embeddings from HuggingFace
get_max_position_embeddings() {
    local model_name=$1
    #Download config.json from HuggingFace
    local config_url="https://huggingface.co/${model_name}/resolve/main/config.json"
    local config_file="/tmp/${model_name//\//_}_config.json"
    curl -s -L "$config_url" -o "$config_file"
    if [ -f "$config_file" ]; then
        local max_pos=$(jq '.max_position_embeddings' "$config_file")
        if [ "$max_pos" != "null" ] && [ -n "$max_pos" ]; then
            echo $max_pos
            return
        fi
    fi
    #Fallback
    echo 8192
}

#Check available GPUs and their memory
echo "=== GPU Detection and Analysis ==="

#Get number of GPUs on the node
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Total GPUs detected on node: $NUM_GPUS"

if [ $NUM_GPUS -eq 0 ]; then
    echo "Error: No GPUs detected on this node!"
    exit 1
fi

#Get model's max context length
echo "Fetching model configuration..."
MAX_MODEL_LEN=$(get_max_position_embeddings $MODEL_NAME)
echo "Model max position embeddings: $MAX_MODEL_LEN"

#Use MAX_MODEL_LEN in memory estimation and for --max-model-len
REQUIRED_MEMORY_MB=$(estimate_model_memory_with_context $MODEL_NAME $MAX_MODEL_LEN)
echo "Estimated memory requirement per GPU: ${REQUIRED_MEMORY_MB}MB (~$((REQUIRED_MEMORY_MB/1024))GB)"

#Check each GPU for availability and memory
AVAILABLE_GPUS=""
TENSOR_PARALLEL_SIZE=0
SUFFICIENT_MEMORY_GPUS=""

echo ""
echo "=== GPU Availability Analysis ==="

for ((i=0; i<NUM_GPUS; i++)); do
    free_memory=$(get_gpu_memory $i)
    total_memory=$(get_total_gpu_memory $i)
    used_memory=$((total_memory - free_memory))
    utilization=$(get_gpu_utilization $i)
    
    echo "GPU $i: ${free_memory}MB free / ${total_memory}MB total (${used_memory}MB used, ${utilization}% utilization)"
    
    #Check if GPU has enough free memory
    if [ $free_memory -gt $REQUIRED_MEMORY_MB ]; then
        SUFFICIENT_MEMORY_GPUS="$SUFFICIENT_MEMORY_GPUS $i"
        echo "  -> GPU $i has sufficient memory"
        
        #Check if GPU is not too busy
        if is_gpu_available $i; then
            if [ -z "$AVAILABLE_GPUS" ]; then
                AVAILABLE_GPUS="$i"
            else
                AVAILABLE_GPUS="$AVAILABLE_GPUS,$i"
            fi
            TENSOR_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE + 1))
            echo "  -> GPU $i selected for vLLM"
        else
            echo "  -> GPU $i excluded (too busy: ${utilization}% utilization)"
        fi
    else
        echo "  -> GPU $i excluded (needs ${REQUIRED_MEMORY_MB}MB, has ${free_memory}MB)"
    fi
done

echo ""
echo "=== Resource Summary ==="
echo "GPUs with sufficient memory: $SUFFICIENT_MEMORY_GPUS"
echo "GPUs available for vLLM: $AVAILABLE_GPUS"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"

if [ $TENSOR_PARALLEL_SIZE -eq 0 ]; then
    echo ""
    echo "Error: No GPUs available for vLLM!"
    echo "Required: At least ${REQUIRED_MEMORY_MB}MB free memory per GPU and <80% utilization"
    echo ""
    echo "Suggestions:"
    echo "1. Try a smaller model (7B instead of 17B)"
    echo "2. Reduce context length (currently ${MAX_MODEL_LEN})"
    echo "3. Free up GPU memory by stopping other processes"
    echo "4. Wait for GPU utilization to decrease"
    echo "5. Use more GPUs if available"
    echo ""
    
    #Try with reduced context lengths
    for context_len in 16384 8192 4096; do
        echo "Attempting with reduced context length (${context_len})..."
        REQUIRED_MEMORY_MB=$(estimate_model_memory_with_context $MODEL_NAME $context_len)
        TENSOR_PARALLEL_SIZE=0
        AVAILABLE_GPUS=""
        
        for ((i=0; i<NUM_GPUS; i++)); do
            free_memory=$(get_gpu_memory $i)
            if [ $free_memory -gt $REQUIRED_MEMORY_MB ] && is_gpu_available $i; then
                if [ -z "$AVAILABLE_GPUS" ]; then
                    AVAILABLE_GPUS="$i"
                else
                    AVAILABLE_GPUS="$AVAILABLE_GPUS,$i"
                fi
                TENSOR_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE + 1))
            fi
        done
        
        if [ $TENSOR_PARALLEL_SIZE -gt 0 ]; then
            echo "Success with ${context_len} context length!"
            MAX_MODEL_LEN=$context_len
            break
        fi
    done
    
    if [ $TENSOR_PARALLEL_SIZE -eq 0 ]; then
        echo "Model is too large for available GPUs even with minimal context."
        echo ""
        echo "Recommended alternatives:"
        echo "1. Use Llama-4-7B instead: meta-llama/Llama-4-7b-chat-hf"
        echo "2. Use Llama-4-13B instead: meta-llama/Llama-4-13b-chat-hf"
        echo "3. Free up more GPU memory"
        echo "4. Use more GPUs if available"
        exit 1
    fi
fi

#Limit tensor parallelism for very large models
if [[ $MODEL_NAME == *"17b"* ]] || [[ $MODEL_NAME == *"17B"* ]]; then
    if [ $TENSOR_PARALLEL_SIZE -gt 4 ]; then
        echo "Warning: 17B model detected. Limiting tensor parallelism to 4 for stability."
        TENSOR_PARALLEL_SIZE=4
        #Take only first 4 GPUs
        AVAILABLE_GPUS=$(echo $AVAILABLE_GPUS | cut -d',' -f1-4)
    fi
fi

#Ensure we have at least 1 GPU
if [ $TENSOR_PARALLEL_SIZE -eq 0 ]; then
    echo "Error: No GPUs available after all checks!"
    exit 1
fi

echo ""
echo "=== Final Configuration ==="
echo "  Available GPUs: $AVAILABLE_GPUS"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Model: $MODEL_NAME"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  Estimated Memory per GPU: ${REQUIRED_MEMORY_MB}MB"

#Activate conda environment
echo ""
echo "=== Starting vLLM Server ==="
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vllm-env

#Disable NCCL P2P communication
export NCCL_P2P_DISABLE=1 
#Set visible GPUs to only use available ones
export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS

#Start vLLM server with calculated parameters
echo "Starting vLLM server with model: $MODEL_NAME"
echo "Using GPUs: $AVAILABLE_GPUS"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "Max model length: $MAX_MODEL_LEN"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype auto

#Handle script interruption
trap 'echo "Shutting down vLLM server..."; exit' INT TERM 