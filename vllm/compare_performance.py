import argparse
import subprocess
import time
import os
from datetime import datetime

def run_inference(script_path, input_file, output_file, batch_size, **kwargs):
    """Run inference script and return execution time."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    cmd = [
        "python", script_path,
        "--input", input_file,
        "--output", output_file,
        "--batch_size", str(batch_size)  # Explicitly set batch size
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    start_time = time.time()
    subprocess.run(cmd, check=True)
    end_time = time.time()
    
    return end_time - start_time

def main():
    parser = argparse.ArgumentParser(description="Compare performance between original and optimized batch inference")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_output = f"{args.output_dir}/original_results_{timestamp}.jsonl"
    optimized_output = f"{args.output_dir}/optimized_results_{timestamp}.jsonl"
    
    print(f"\nRunning comparison with batch size: {args.batch_size}")
    
    # Run original inference
    print("\nRunning original batch inference...")
    original_time = run_inference(
        "batch_inference.py",
        args.input,
        original_output,
        args.batch_size  # Pass batch size explicitly
    )
    
    # Run optimized inference
    print("\nRunning optimized batch inference...")
    optimized_time = run_inference(
        "optimized_batch_inference.py",
        args.input,
        optimized_output,
        args.batch_size,  # Pass batch size explicitly
        num_workers=args.num_workers
    )
    
    # Calculate performance improvements
    time_improvement = (original_time - optimized_time) / original_time * 100
    
    # Print results
    print("\nPerformance Comparison Results:")
    print(f"Batch Size: {args.batch_size}")
    print(f"Original Processing Time: {original_time:.2f} seconds")
    print(f"Optimized Processing Time: {optimized_time:.2f} seconds")
    print(f"Performance Improvement: {time_improvement:.2f}%")
    print(f"\nResults saved to:")
    print(f"Original: {original_output}")
    print(f"Optimized: {optimized_output}")

if __name__ == "__main__":
    main() 