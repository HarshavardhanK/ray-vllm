import json
import argparse
import time
from typing import List, Dict, Any
import torch
import torch.cuda
from vllm import LLM, SamplingParams
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil
import GPUtil

def get_gpu_metrics():
    """Get current GPU metrics."""
    gpus = GPUtil.getGPUs()
    metrics = []
    for gpu in gpus:
        metrics.append({
            'id': gpu.id,
            'load': gpu.load * 100,
            'memory_used': gpu.memoryUsed,
            'memory_total': gpu.memoryTotal,
            'temperature': gpu.temperature
        })
    return metrics

def get_system_metrics():
    """Get current system metrics."""
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used': psutil.virtual_memory().used / (1024 * 1024 * 1024),  # GB
        'memory_total': psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    }

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_batch(
    input_file: str,
    output_file: str,
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 128,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    batch_size: int = 32,
    num_workers: int = 4
):
    """Process a batch of prompts from a JSONL file with optimized performance."""
    
    # Initialize performance metrics
    performance_metrics = {
        'start_time': time.time(),
        'gpu_metrics': [],
        'system_metrics': [],
        'batch_times': []
    }
    
    # Clear CUDA cache and set memory allocation strategy
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.85)
    
    print("Loading model...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        max_num_batched_tokens=8192,
        max_num_seqs=512,
        max_model_len=2048,
        dtype="bfloat16",
        trust_remote_code=True
    )
    
    print("Loading input data...")
    data = load_jsonl(input_file)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty
    )
    
    print("Processing prompts...")
    results = []
    
    # Process in batches with dynamic batch size
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        prompts = [item["prompt"] for item in batch]
        
        # Record metrics before processing
        performance_metrics['gpu_metrics'].append(get_gpu_metrics())
        performance_metrics['system_metrics'].append(get_system_metrics())
        
        # Process batch
        batch_start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        batch_end_time = time.time()
        
        # Record batch processing time
        performance_metrics['batch_times'].append(batch_end_time - batch_start_time)
        
        # Process outputs
        for j, output in enumerate(outputs):
            result = {
                "id": batch[j]["id"],
                "prompt": batch[j]["prompt"],
                "response": output.outputs[0].text,
                "processing_time": batch_end_time - batch_start_time
            }
            results.append(result)
    
    # Record final metrics
    performance_metrics['end_time'] = time.time()
    performance_metrics['gpu_metrics'].append(get_gpu_metrics())
    performance_metrics['system_metrics'].append(get_system_metrics())
    
    # Save results
    print("\nSaving results...")
    save_jsonl(results, output_file)
    
    # Save performance metrics
    metrics_file = output_file.replace('.jsonl', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    
    # Print summary
    total_time = performance_metrics['end_time'] - performance_metrics['start_time']
    print(f"\nProcessing complete!")
    print(f"Total prompts processed: {len(results)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per prompt: {total_time/len(results):.2f} seconds")
    print(f"Average time per batch: {np.mean(performance_metrics['batch_times']):.2f} seconds")
    print(f"Results saved to: {output_file}")
    print(f"Performance metrics saved to: {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a batch of prompts using optimized vLLM")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf", help="Model name or path")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--frequency_penalty", type=float, default=0.0, help="Frequency penalty")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    
    process_batch(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    ) 