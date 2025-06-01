from vllm import LLM, SamplingParams
import asyncio
import aiohttp
import json
import subprocess
import time
from typing import List
import torch

def optimized_offline_inference():
    """Demonstrate optimized offline batched inference using vLLM."""
    print("\n=== Optimized Offline Inference Demo ===")
    
    # Initialize LLM with optimized settings
    llm = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size=torch.cuda.device_count(),  #Use all available GPUs
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        max_num_batched_tokens=4096,  #Increase batch size for better throughput
        max_num_seqs=256,  #Maximum number of sequences to process in parallel
        max_model_len=2048,  #Maximum sequence length
        dtype="bfloat16",  #Use bfloat16 for better memory efficiency
    )
    
    # Define a larger batch of prompts for better throughput
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
        "Explain quantum computing in simple terms",
        "Write a short poem about AI",
        "What is machine learning?",
        "Describe the process of photosynthesis",
        "What are the benefits of exercise?",
    ]
    
    # Optimized sampling parameters
    sampling_params = SamplingParams(
        temperature=0.85,
        top_p=0.95,
        max_tokens=128,
        presence_penalty=0.1,  # Add slight presence penalty for diversity
        frequency_penalty=0.1,  # Add slight frequency penalty for diversity
        # #use_beam_search=False,  # Disable beam search for faster generation
        # stop=None,  # No stop tokens for faster generation
        # ignore_eos=True,  # Ignore EOS token for faster generation
    )
    
    # Generate outputs with timing
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    # Print results with timing information
    print(f"\nProcessed {len(prompts)} prompts in {end_time - start_time:.2f} seconds")
    print(f"Average time per prompt: {(end_time - start_time) / len(prompts):.2f} seconds")
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

def optimized_api_server():
    """Demonstrate optimized vLLM API server functionality."""
    print("\n=== Optimized API Server Demo ===")
    
    # Start the API server with optimized settings
    server_process = subprocess.Popen(
        ["python", "-m", "vllm.entrypoints.api_server",
         "--model", "mistralai/Mistral-7B-v0.1",
         "--trust-remote-code",
         "--port", "8000",
         "--tensor-parallel-size", str(torch.cuda.device_count()),
         "--gpu-memory-utilization", "0.9",
         "--max-num-batched-tokens", "4096",
         "--max-num-seqs", "256",
         "--max-model-len", "2048",
         "--dtype", "bfloat16"]
    )
    
    # Wait for server to start
    time.sleep(5)
    
    # Make multiple concurrent requests to demonstrate batching
    async def make_requests():
        async def make_request(prompt: str):
            url = "http://localhost:8000/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.8,
                "top_p": 0.95,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1,
                "ignore_eos": True
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    return await response.json()
        
        prompts = [
            "The capital of France is",
            "Explain quantum computing",
            "Write a short poem about AI"
        ]
        
        start_time = time.time()
        tasks = [make_request(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"\nProcessed {len(prompts)} API requests in {end_time - start_time:.2f} seconds")
        print(f"Average time per request: {(end_time - start_time) / len(prompts):.2f} seconds")
        
        for prompt, result in zip(prompts, results):
            print(f"\nPrompt: {prompt}")
            print(f"Response: {json.dumps(result, indent=2)}")
    
    try:
        asyncio.run(make_requests())
    except Exception as e:
        print(f"Error making API requests: {e}")
    finally:
        # Clean up
        server_process.terminate()
        server_process.wait()

def main():
    # Run offline inference demo
    optimized_offline_inference()
    
    # Run API server demo
    optimized_api_server()

if __name__ == "__main__":
    main() 