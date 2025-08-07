#Disable flash-attention and use PyTorch native SDPA
import os
os.environ['USE_FLASH_ATTENTION_2'] = 'false'
os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'sdpa'

from sentence_transformers import CrossEncoder
import torch


class CrossEncoderService:
    def __init__(self):
        #Check if CUDA is available, otherwise use CPU
        if torch.cuda.is_available():
            #Check available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = gpu_memory - allocated_memory
            
            #Estimate model size (rough estimate for MiniLM-L6-v2)
            estimated_model_size = 80 * 1024 * 1024  # ~80MB for MiniLM-L6-v2
            
            if free_memory > estimated_model_size * 2:  # Leave some buffer
                try:
                    self.device = 'cuda:0'
                    model_name = 'cross-encoder/ms-marco-MiniLM-L6-v2'
                    self.model = CrossEncoder(model_name, device=self.device)
                    print(f"CrossEncoder model loaded on {self.device}")
                except Exception as e:
                    print(f"Failed to load model on GPU: {e}, falling back to CPU")
                    self.device = 'cpu'
                    model_name = 'cross-encoder/ms-marco-MiniLM-L6-v2'
                    self.model = CrossEncoder(model_name, device=self.device)
                    print(f"CrossEncoder model loaded on {self.device}")
            else:
                print(f"GPU memory insufficient (free: {free_memory/1024**3:.2f}GB), falling back to CPU")
                self.device = 'cpu'
                model_name = 'cross-encoder/ms-marco-MiniLM-L6-v2'
                self.model = CrossEncoder(model_name, device=self.device)
                print(f"CrossEncoder model loaded on {self.device}")
        else:
            self.device = 'cpu'
            model_name = 'cross-encoder/ms-marco-MiniLM-L6-v2'
            self.model = CrossEncoder(model_name, device=self.device)
            print(f"CrossEncoder model loaded on {self.device}")

    def predict(self, query: str, passages: list) -> list:
        #Create query-passage pairs
        pairs = [(query, passage) for passage in passages]
        #Get predictions
        scores = self.model.predict(pairs)
        return scores.tolist() 