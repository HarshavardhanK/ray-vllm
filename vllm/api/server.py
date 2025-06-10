#!/usr/bin/env python3
#Start vLLM OpenAI-compatible API server using subprocess
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM OpenAI-compatible API server wrapper")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name or path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Fraction of GPU memory to use")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum sequence length")
    return parser.parse_args()

def main():
    args = parse_args()
    cli_args = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", args.host,
        "--port", str(args.port),
        "--model", args.model,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len)
    ]
    print(f"Launching: {' '.join(cli_args)}")
    subprocess.run(cli_args)

if __name__ == "__main__":
    main() 