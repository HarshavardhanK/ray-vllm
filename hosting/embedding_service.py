#Disable flash-attention and use PyTorch native SDPA
import os
os.environ['USE_FLASH_ATTENTION_2'] = 'false'
os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'sdpa'

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str = None, use_sentence_transformers: bool = True):
        """
        Initialize the embedding service with Qwen3-Embedding-8B model.
        
        Args:
            model_name: HuggingFace model identifier (if None, uses default)
            use_sentence_transformers: Whether to use sentence-transformers (recommended) or transformers directly
        """
        self.model_name = model_name or 'Qwen/Qwen3-Embedding-8B'
        self.use_sentence_transformers = use_sentence_transformers
        self.device = 'cuda:1'  #Manually use the free GPU
        
        if use_sentence_transformers:
            #Use sentence-transformers for easier usage
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs={"attn_implementation": "sdpa"},
                tokenizer_kwargs={"padding_side": "left"}
            )
            self.tokenizer = None
        else:
            #Use transformers directly for more control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                attn_implementation="sdpa", 
                torch_dtype=torch.float16
            ).to(self.device)
        
        print(f"Embedding service initialized with {self.model_name} on {self.device}")
    
    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract embeddings using last token pooling."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format instruction for query embedding."""
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def encode_texts(self, texts: List[str], task_description: Optional[str] = None, 
                    max_length: int = 8192, normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            task_description: Optional task description for instruction-aware encoding
            max_length: Maximum sequence length
            normalize: Whether to normalize embeddings
            
        Returns:
            numpy array of embeddings
        """
        if self.use_sentence_transformers:
            if task_description:
                #For queries, use instruction-aware encoding
                formatted_texts = [self.get_detailed_instruct(task_description, text) for text in texts]
                embeddings = self.model.encode(formatted_texts, prompt_name="query")
            else:
                #For documents, use regular encoding
                embeddings = self.model.encode(texts)
            
            if normalize:
                embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1).numpy()
            else:
                embeddings = np.array(embeddings)
        else:
            #Use transformers directly
            if task_description:
                formatted_texts = [self.get_detailed_instruct(task_description, text) for text in texts]
            else:
                formatted_texts = texts
            
            batch_dict = self.tokenizer(
                formatted_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            batch_dict.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def compute_similarity(self, query_embeddings: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document embeddings."""
        return np.dot(query_embeddings, document_embeddings.T)
    
    def embed_queries_and_documents(self, queries: List[str], documents: List[str], 
                                  task_description: str = "Given a web search query, retrieve relevant passages that answer the query") -> Dict[str, Any]:
        """
        Embed queries and documents, then compute similarities.
        
        Args:
            queries: List of query texts
            documents: List of document texts
            task_description: Task description for instruction-aware encoding
            
        Returns:
            Dictionary containing embeddings and similarity scores
        """
        #Encode queries with instruction
        query_embeddings = self.encode_texts(queries, task_description=task_description)
        
        #Encode documents without instruction
        document_embeddings = self.encode_texts(documents)
        
        #Compute similarities
        similarities = self.compute_similarity(query_embeddings, document_embeddings)
        
        return {
            "query_embeddings": query_embeddings.tolist(),
            "document_embeddings": document_embeddings.tolist(),
            "similarities": similarities.tolist(),
            "model_name": self.model_name
        }
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        if self.use_sentence_transformers:
            return self.model.get_sentence_embedding_dimension()
        else:
            #For Qwen3-Embedding-8B, default dimension is 4096
            return 4096 