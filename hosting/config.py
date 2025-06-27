import os
import torch
import subprocess
from typing import Optional

class Config:
    def __init__(self):
        self.gpu_id = self._find_free_gpu()
        self.device = self._get_device()
        self.model_configs = {
            'cross_encoder': 'cross-encoder/ms-marco-MiniLM-L6-v2',
            'embedding': 'Qwen/Qwen3-Embedding-8B',
            'vllm_embedding': 'Qwen/Qwen3-Embedding-8B'
        }
    
    def _find_free_gpu(self, min_memory_gb: int = 8) -> Optional[int]:
        """Find a GPU with at least min_memory_gb free memory."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_id = int(parts[0])
                        memory_used = int(parts[1])
                        memory_total = int(parts[2])
                        memory_free = memory_total - memory_used
                        
                        if memory_free >= min_memory_gb * 1024:  #Convert GB to MB
                            print(f"Found free GPU {gpu_id} with {memory_free}MB free memory")
                            return gpu_id
            
            print("No GPU with sufficient free memory found")
            return None
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return None
    
    def _get_device(self) -> str:
        """Get the device string for the selected GPU."""
        if self.gpu_id is not None and torch.cuda.is_available():
            #Since we set CUDA_VISIBLE_DEVICES, we use cuda:0 for the selected GPU
            device = 'cuda:0'
            print(f"Using device: {device}")
            return device
        else:
            device = 'cpu'
            print(f"Using device: {device}")
            return device
    
    def get_model_config(self, model_type: str) -> str:
        """Get model configuration for a specific type."""
        return self.model_configs.get(model_type, '')

#Global config instance
config = Config() 