from sentence_transformers import CrossEncoder
import torch

class CrossEncoderService:
    def __init__(self):
        #Initialize the model
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
        #Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict(self, query: str, passages: list) -> list:
        #Create query-passage pairs
        pairs = [(query, passage) for passage in passages]
        #Get predictions
        scores = self.model.predict(pairs)
        return scores.tolist() 