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
            self.device = 'cuda:0'  # Use GPU 0 as configured in docker-compose
        else:
            self.device = 'cpu'
        
        #Initialize the model with device specification
        model_name = 'cross-encoder/ms-marco-MiniLM-L6-v2'
        self.model = CrossEncoder(model_name, device=self.device)
        print(f"CrossEncoder model loaded on {self.device}")

    def predict(self, query: str, passages: list) -> list:
        #Create query-passage pairs
        pairs = [(query, passage) for passage in passages]
        #Get predictions
        scores = self.model.predict(pairs)
        return scores.tolist() 