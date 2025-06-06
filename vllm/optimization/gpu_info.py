import torch
import GPUtil
import subprocess
import json
from typing import Dict, List, Optional

def get_gpu_info() -> Dict:
    """
    Get detailed information about available GPUs including cores and threads.
    Returns a dictionary with GPU information.
    """
    gpu_info = {}
    
    if not torch.cuda.is_available():
        return {"error": "No CUDA devices available"}
    
    try:
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        gpu_info["total_gpus"] = gpu_count
        
        # Get detailed info for each GPU
        gpu_info["gpus"] = []
        
        for i in range(gpu_count):
            gpu = {}
            
            # Get basic GPU info using torch
            gpu["device_id"] = i
            gpu["name"] = torch.cuda.get_device_name(i)
            gpu["total_memory"] = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            
            # Get detailed GPU info using nvidia-smi
            try:
                nvidia_smi = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=compute_cap,memory.total,memory.free,utilization.gpu", 
                     "--format=csv,noheader,nounits", f"--id={i}"]
                ).decode().strip().split(',')
                
                gpu["compute_capability"] = nvidia_smi[0]
                gpu["total_memory_smi"] = float(nvidia_smi[1])  # GB
                gpu["free_memory"] = float(nvidia_smi[2])  # GB
                gpu["gpu_utilization"] = float(nvidia_smi[3])  # %
                
            except subprocess.CalledProcessError:
                gpu["error"] = "Failed to get detailed GPU info"
            
            # Get CUDA cores information
            try:
                # This is an approximation based on compute capability
                compute_cap = float(gpu["compute_capability"])
                if compute_cap >= 8.0:  # Ampere or newer
                    gpu["cuda_cores"] = 6912  # A100
                elif compute_cap >= 7.5:  # Turing
                    gpu["cuda_cores"] = 4608  # RTX 2080 Ti
                elif compute_cap >= 7.0:  # Volta
                    gpu["cuda_cores"] = 5120  # V100
                else:
                    gpu["cuda_cores"] = "Unknown for this architecture"
            except:
                gpu["cuda_cores"] = "Unknown"
            
            gpu_info["gpus"].append(gpu)
        
        # Get CUDA runtime info
        gpu_info["cuda_version"] = torch.version.cuda
        gpu_info["cudnn_version"] = torch.backends.cudnn.version()
        
        # Get current memory usage
        gpu_info["current_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)  # GB
        gpu_info["current_memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)  # GB
        
        return gpu_info
        
    except Exception as e:
        return {"error": f"Error getting GPU info: {str(e)}"}

def estimate_threads_per_prompt(model_name: str, batch_size: int = 1) -> Dict:
    """
    Estimate the number of threads needed per prompt based on model and batch size.
    This is an approximation and may need to be adjusted based on specific use cases.
    """
    try:
        # Get GPU info first
        gpu_info = get_gpu_info()
        if "error" in gpu_info:
            return gpu_info
        
        # Basic estimation based on model size and batch size
        # These are rough estimates and should be tuned based on actual performance
        model_size_estimates = {
            "7b": 256,    # Threads per prompt for 7B models
            "13b": 384,   # Threads per prompt for 13B models
            "70b": 512,   # Threads per prompt for 70B models
        }
        
        # Default to 256 if model size not found
        base_threads = 256
        for size, threads in model_size_estimates.items():
            if size in model_name.lower():
                base_threads = threads
                break
        
        # Adjust for batch size
        threads_per_prompt = base_threads * batch_size
        
        # Get available GPU cores
        available_cores = gpu_info["gpus"][0]["cuda_cores"]
        if isinstance(available_cores, str):
            available_cores = 6912  # Default to A100 cores if unknown
        
        # Calculate theoretical occupancy
        theoretical_occupancy = (threads_per_prompt * batch_size) / available_cores
        
        return {
            "model_name": model_name,
            "batch_size": batch_size,
            "estimated_threads_per_prompt": threads_per_prompt,
            "available_gpu_cores": available_cores,
            "theoretical_occupancy": min(theoretical_occupancy, 1.0),
            "gpu_info": gpu_info
        }
        
    except Exception as e:
        return {"error": f"Error estimating threads: {str(e)}"}

def main():
    """Main function to demonstrate usage"""
    # Get general GPU info
    print("\n=== GPU Information ===")
    gpu_info = get_gpu_info()
    print(json.dumps(gpu_info, indent=2))
    
    # Estimate threads for different models
    print("\n=== Thread Estimation Examples ===")
    models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf"
    ]
    
    for model in models:
        print(f"\nEstimating for {model}:")
        estimate = estimate_threads_per_prompt(model)
        print(json.dumps(estimate, indent=2))

if __name__ == "__main__":
    main() 