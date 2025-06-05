import json
import argparse
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def run_inference(script_path, input_file, output_file, **kwargs):
    """Run inference script and return execution time."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    cmd = [
        "python", script_path,
        "--input", input_file,
        "--output", output_file
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    start_time = time.time()
    subprocess.run(cmd, check=True)
    end_time = time.time()
    
    return end_time - start_time

def plot_metrics(original_metrics, optimized_metrics, output_dir):
    """Plot performance comparison metrics."""
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Total Processing Time
    times = [
        original_metrics.get('end_time', 0) - original_metrics.get('start_time', 0),
        optimized_metrics.get('end_time', 0) - optimized_metrics.get('start_time', 0)
    ]
    ax1.bar(['Original', 'Optimized'], times)
    ax1.set_title('Total Processing Time')
    ax1.set_ylabel('Seconds')
    
    # Plot 2: Average Time per Prompt
    avg_times = [
        times[0] / 200,  # Assuming 200 prompts
        times[1] / 200
    ]
    ax2.bar(['Original', 'Optimized'], avg_times)
    ax2.set_title('Average Time per Prompt')
    ax2.set_ylabel('Seconds')
    
    # Plot 3: GPU Memory Usage
    gpu_memory = [
        np.mean([m.get('memory_used', 0) for m in original_metrics.get('gpu_metrics', [{'memory_used': 0}])]),
        np.mean([m.get('memory_used', 0) for m in optimized_metrics.get('gpu_metrics', [{'memory_used': 0}])])
    ]
    ax3.bar(['Original', 'Optimized'], gpu_memory)
    ax3.set_title('Average GPU Memory Usage')
    ax3.set_ylabel('MB')
    
    # Plot 4: System Memory Usage
    sys_memory = [
        np.mean([m.get('memory_used', 0) for m in original_metrics.get('system_metrics', [{'memory_used': 0}])]),
        np.mean([m.get('memory_used', 0) for m in optimized_metrics.get('system_metrics', [{'memory_used': 0}])])
    ]
    ax4.bar(['Original', 'Optimized'], sys_memory)
    ax4.set_title('Average System Memory Usage')
    ax4.set_ylabel('GB')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png")
    plt.close()

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
    
    # Run original inference
    print("Running original batch inference...")
    original_time = run_inference(
        "batch_inference.py",
        args.input,
        original_output,
        batch_size=args.batch_size
    )
    
    # Run optimized inference
    print("Running optimized batch inference...")
    optimized_time = run_inference(
        "optimized_batch_inference.py",
        args.input,
        optimized_output,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create basic metrics for original inference
    original_metrics = {
        'start_time': time.time() - original_time,
        'end_time': time.time(),
        'gpu_metrics': [{'memory_used': 0}],
        'system_metrics': [{'memory_used': 0}]
    }
    
    # Load optimized metrics
    with open(optimized_output.replace('.jsonl', '_metrics.json'), 'r') as f:
        optimized_metrics = json.load(f)
    
    # Calculate performance improvements
    time_improvement = (original_time - optimized_time) / original_time * 100
    
    # Print results
    print("\nPerformance Comparison Results:")
    print(f"Original Processing Time: {original_time:.2f} seconds")
    print(f"Optimized Processing Time: {optimized_time:.2f} seconds")
    print(f"Performance Improvement: {time_improvement:.2f}%")
    
    # Plot metrics
    plot_metrics(original_metrics, optimized_metrics, args.output_dir)
    print(f"\nPerformance comparison plot saved to: {args.output_dir}/performance_comparison.png")

if __name__ == "__main__":
    main() 