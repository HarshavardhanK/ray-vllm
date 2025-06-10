import json
import argparse
import time
from typing import List, Dict, Any
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm

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
    batch_size: int = 8,
    num_workers: int = 4
):
    """Process a batch of prompts from a JSONL file with optimized performance."""
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    print("Loading model...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,  # High memory utilization for better throughput
        enforce_eager=True,  # Keep eager mode for faster initialization
        max_num_batched_tokens=4096,  # Match model's max position embeddings
        max_num_seqs=256,  # Reduced for better stability
        max_model_len=4096,  # Match model's max position embeddings
        dtype="bfloat16",  # Using bfloat16 for better memory efficiency
        trust_remote_code=True,
        enable_chunked_prefill=True  # Enable chunked prefill for better memory efficiency
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
    start_time = time.time()
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        prompts = [item["prompt"] for item in batch]
        
        # Process batch
        outputs = llm.generate(prompts, sampling_params)
        
        # Process outputs
        for j, output in enumerate(outputs):
            result = {
                "id": batch[j]["id"],
                "prompt": batch[j]["prompt"],
                "response": output.outputs[0].text
            }
            results.append(result)
    
    end_time = time.time()
    
    # Save results
    print("\nSaving results...")
    save_jsonl(results, output_file)
    
    # Print summary
    total_time = end_time - start_time
    print(f"\nProcessing complete!")
    print(f"Total prompts processed: {len(results)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per prompt: {total_time/len(results):.2f} seconds")
    print(f"Results saved to: {output_file}")

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
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
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