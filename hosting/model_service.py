#Disable flash-attention and use PyTorch native SDPA
import os
os.environ['USE_FLASH_ATTENTION_2'] = 'false'
os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'sdpa'

from sentence_transformers import CrossEncoder
import torch


class CrossEncoderService:
    def __init__(self):
        #Manually use the free GPU
        self.device = 'cuda:1'
        
        #Initialize the model
        model_name = 'cross-encoder/ms-marco-MiniLM-L6-v2'
        self.model = CrossEncoder(model_name)
        #Move model to the selected device
        self.model.to(self.device)
        print(f"CrossEncoder model loaded on {self.device}")

    def predict(self, query: str, passages: list) -> list:
        #Create query-passage pairs
        pairs = [(query, passage) for passage in passages]
        #Get predictions
        scores = self.model.predict(pairs)
        return scores.tolist() 