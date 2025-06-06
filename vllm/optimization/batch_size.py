import torch
import time
import psutil
import GPUtil
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import json

class BatchSizeOptimizer:
    def __init__(self, model_name: str, test_prompts: List[str]):
        self.model_name = model_name
        self.test_prompts = test_prompts
        self.results = {}
        
    def get_gpu_memory_info(self):
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,      # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }
        return None
    
    def benchmark_batch_size(self, batch_size: int, iterations: int = 5) -> Dict:
        """Benchmark a specific batch size."""
        print(f"\nTesting batch size: {batch_size}")
        
        # Clear cache before each test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Initialize vLLM model with current batch size considerations
            llm = LLM(
                model=self.model_name,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=0.9,
                max_num_batched_tokens=batch_size * 256,  # Estimate tokens per prompt
                max_num_seqs=batch_size * 2,
                dtype="bfloat16"
            )
            
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=128,
                top_p=0.95
            )
            
            # Get baseline memory
            baseline_memory = self.get_gpu_memory_info()
            
            # Prepare test batches
            test_batches = []
            for i in range(0, len(self.test_prompts), batch_size):
                batch = self.test_prompts[i:i + batch_size]
                if len(batch) == batch_size:  # Only use full batches for fair comparison
                    test_batches.append(batch)
            
            if not test_batches:
                return {"error": "Not enough prompts for full batch"}
            
            # Warmup
            _ = llm.generate(test_batches[0], sampling_params)
            torch.cuda.synchronize()
            
            # Benchmark iterations
            times = []
            memory_peaks = []
            
            for iteration in range(iterations):
                start_time = time.perf_counter()
                
                for batch in test_batches[:3]:  # Test first 3 batches
                    _ = llm.generate(batch, sampling_params)
                
                torch.cuda.synchronize()  # Ensure all GPU work is complete
                end_time = time.perf_counter()
                
                iteration_time = end_time - start_time
                times.append(iteration_time)
                
                memory_info = self.get_gpu_memory_info()
                memory_peaks.append(memory_info['max_allocated'])
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            total_prompts = len(test_batches[:3]) * batch_size
            throughput = total_prompts / avg_time  # prompts per second
            avg_memory_peak = np.mean(memory_peaks)
            
            # GPU utilization (approximation)
            gpu_util = self.estimate_gpu_utilization(batch_size)
            
            result = {
                'batch_size': batch_size,
                'avg_time': avg_time,
                'std_time': std_time,
                'throughput': throughput,
                'prompts_tested': total_prompts,
                'memory_peak_gb': avg_memory_peak,
                'gpu_utilization_est': gpu_util,
                'efficiency_score': throughput / avg_memory_peak,  # throughput per GB
                'error': None
            }
            
            # Clean up
            del llm
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            torch.cuda.empty_cache()
            return {
                'batch_size': batch_size,
                'error': str(e),
                'throughput': 0,
                'memory_peak_gb': float('inf')
            }
    
    def estimate_gpu_utilization(self, batch_size: int) -> float:
        """Estimate GPU utilization based on batch size."""
        # This is a rough approximation based on typical GPU architectures
        gpu_cores = 6912  # Approximate for A100
        threads_per_prompt = 256  # Rough estimate
        total_threads = batch_size * threads_per_prompt
        
        utilization = min(total_threads / gpu_cores, 1.0) * 100
        return utilization
    
    def find_optimal_batch_size(self, batch_sizes: List[int] = None) -> Dict:
        """Find the optimal batch size through systematic testing."""
        if batch_sizes is None:
            # Default test range - exponential growth
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
        print("Starting batch size optimization...")
        print(f"Testing batch sizes: {batch_sizes}")
        
        results = []
        
        for batch_size in batch_sizes:
            result = self.benchmark_batch_size(batch_size)
            results.append(result)
            
            if result.get('error'):
                print(f"Batch size {batch_size}: ERROR - {result['error']}")
                # If we hit memory limits, stop testing larger sizes
                if 'memory' in result['error'].lower() or 'oom' in result['error'].lower():
                    print("Hit memory limit, stopping larger batch size tests")
                    break
            else:
                print(f"Batch size {batch_size}: {result['throughput']:.1f} prompts/s, "
                      f"{result['memory_peak_gb']:.1f}GB peak memory")
        
        # Filter out errors and find optimal
        valid_results = [r for r in results if not r.get('error')]
        
        if not valid_results:
            return {"error": "No valid batch sizes found"}
        
        # Find optimal by different criteria
        best_throughput = max(valid_results, key=lambda x: x['throughput'])
        best_efficiency = max(valid_results, key=lambda x: x['efficiency_score'])
        best_memory = min(valid_results, key=lambda x: x['memory_peak_gb'])
        
        # Save detailed results
        self.results = {
            'all_results': results,
            'valid_results': valid_results,
            'best_throughput': best_throughput,
            'best_efficiency': best_efficiency,
            'best_memory': best_memory,
            'recommendation': self.get_recommendation(valid_results)
        }
        
        return self.results
    
    def get_recommendation(self, valid_results: List[Dict]) -> Dict:
        """Provide intelligent recommendation based on results."""
        if not valid_results:
            return {"error": "No valid results to analyze"}
        
        # Sort by throughput
        sorted_by_throughput = sorted(valid_results, key=lambda x: x['throughput'], reverse=True)
        
        # Find the "knee" of the curve - point of diminishing returns
        best_throughput = sorted_by_throughput[0]['throughput']
        
        for result in sorted_by_throughput:
            throughput_ratio = result['throughput'] / best_throughput
            if throughput_ratio > 0.9:  # Within 90% of best throughput
                return {
                    'recommended_batch_size': result['batch_size'],
                    'reason': f"Best balance of throughput ({result['throughput']:.1f} prompts/s) "
                             f"and memory usage ({result['memory_peak_gb']:.1f}GB)",
                    'throughput': result['throughput'],
                    'memory_gb': result['memory_peak_gb'],
                    'efficiency_score': result['efficiency_score']
                }
        
        # Fallback to best throughput
        best = sorted_by_throughput[0]
        return {
            'recommended_batch_size': best['batch_size'],
            'reason': f"Maximum throughput: {best['throughput']:.1f} prompts/s",
            'throughput': best['throughput'],
            'memory_gb': best['memory_peak_gb'],
            'efficiency_score': best['efficiency_score']
        }
    
    def plot_results(self, save_path: str = None):
        """Plot optimization results."""
        if not self.results or not self.results.get('valid_results'):
            print("No results to plot")
            return
        
        valid_results = self.results['valid_results']
        batch_sizes = [r['batch_size'] for r in valid_results]
        throughputs = [r['throughput'] for r in valid_results]
        memory_usage = [r['memory_peak_gb'] for r in valid_results]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Throughput plot
        ax1.plot(batch_sizes, throughputs, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (prompts/sec)')
        ax1.set_title('Throughput vs Batch Size')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Memory usage plot
        ax2.plot(batch_sizes, memory_usage, 'r-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Peak Memory (GB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Efficiency plot
        efficiency_scores = [r['efficiency_score'] for r in valid_results]
        ax3.plot(batch_sizes, efficiency_scores, 'g-^', linewidth=2, markersize=6)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Efficiency (prompts/sec/GB)')
        ax3.set_title('Efficiency vs Batch Size')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Mark recommended batch size
        rec = self.results['recommendation']
        if not rec.get('error'):
            rec_batch_size = rec['recommended_batch_size']
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=rec_batch_size, color='orange', linestyle='--', 
                          label=f'Recommended: {rec_batch_size}')
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

# Example usage
def run_batch_size_optimization():
    # Generate test prompts
    test_prompts = [
        f"Write a short story about {topic}" 
        for topic in ["space exploration", "artificial intelligence", "ocean depths", 
                     "ancient civilizations", "future technology", "magical forests",
                     "time travel", "parallel universes", "robot companions", "alien contact"] * 10
    ]
    
    optimizer = BatchSizeOptimizer(
        model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        test_prompts=test_prompts
    )
    
    # Run optimization
    results = optimizer.find_optimal_batch_size()
    
    if results.get('error'):
        print(f"Optimization failed: {results['error']}")
        return
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    rec = results['recommendation']
    print(f"RECOMMENDED BATCH SIZE: {rec['recommended_batch_size']}")
    print(f"REASON: {rec['reason']}")
    print(f"Expected throughput: {rec['throughput']:.1f} prompts/second")
    print(f"Memory usage: {rec['memory_gb']:.1f} GB")
    
    print(f"\nBest throughput: {results['best_throughput']['batch_size']} "
          f"({results['best_throughput']['throughput']:.1f} prompts/s)")
    print(f"Best efficiency: {results['best_efficiency']['batch_size']} "
          f"({results['best_efficiency']['efficiency_score']:.2f} prompts/s/GB)")
    print(f"Lowest memory: {results['best_memory']['batch_size']} "
          f"({results['best_memory']['memory_peak_gb']:.1f} GB)")
    
    # Plot results
    optimizer.plot_results(f"batch_size_optimization_{model_name}.png")
    
    # Save detailed results
    with open(f"batch_optimization_results_{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return optimizer

if __name__ == "__main__":
    optimizer = run_batch_size_optimization()