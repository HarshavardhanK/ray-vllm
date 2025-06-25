import torch
import vllm
from vllm import LLM
from typing import List, Dict, Any, Optional
import numpy as np


class VLLMEmbeddingService:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B"):
        """
        Initialize the vLLM embedding service with Qwen3-Embedding-8B model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        
        #Initialize vLLM model for embedding task
        self.model = LLM(model=model_name, task="embed")
        
        print(f"vLLM embedding service initialized with {model_name}")
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format instruction for query embedding."""
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def embed_texts(self, texts: List[str], task_description: Optional[str] = None) -> np.ndarray:
        """
        Embed texts using vLLM for efficient batch processing.
        
        Args:
            texts: List of texts to encode
            task_description: Optional task description for instruction-aware encoding
            
        Returns:
            numpy array of embeddings
        """
        if task_description:
            #For queries, use instruction-aware encoding
            formatted_texts = [self.get_detailed_instruct(task_description, text) for text in texts]
        else:
            #For documents, use regular encoding
            formatted_texts = texts
        
        #Get embeddings using vLLM
        outputs = self.model.embed(formatted_texts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        
        return embeddings.numpy()
    
    def embed_queries_and_documents(self, queries: List[str], documents: List[str], 
                                  task_description: str = "Given a web search query, retrieve relevant passages that answer the query") -> Dict[str, Any]:
        """
        Embed queries and documents, then compute similarities using vLLM.
        
        Args:
            queries: List of query texts
            documents: List of document texts
            task_description: Task description for instruction-aware encoding
            
        Returns:
            Dictionary containing embeddings and similarity scores
        """
        #Combine all texts for batch processing
        input_texts = []
        
        #Add queries with instruction
        for query in queries:
            input_texts.append(self.get_detailed_instruct(task_description, query))
        
        #Add documents without instruction
        input_texts.extend(documents)
        
        #Get embeddings for all texts in one batch
        outputs = self.model.embed(input_texts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        
        #Split embeddings back into queries and documents
        num_queries = len(queries)
        query_embeddings = embeddings[:num_queries]
        document_embeddings = embeddings[num_queries:]
        
        #Compute similarities
        similarities = (query_embeddings @ document_embeddings.T).numpy()
        
        return {
            "query_embeddings": query_embeddings.numpy().tolist(),
            "document_embeddings": document_embeddings.numpy().tolist(),
            "similarities": similarities.tolist(),
            "model_name": self.model_name
        }
    
    def embed_documents_only(self, documents: List[str]) -> np.ndarray:
        """
        Embed only documents (no queries) for document storage/indexing.
        
        Args:
            documents: List of document texts to encode
            
        Returns:
            numpy array of document embeddings
        """
        outputs = self.model.embed(documents)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        
        return embeddings.numpy()
    
    def embed_queries_only(self, queries: List[str], 
                          task_description: str = "Given a web search query, retrieve relevant passages that answer the query") -> np.ndarray:
        """
        Embed only queries with instruction for search operations.
        
        Args:
            queries: List of query texts
            task_description: Task description for instruction-aware encoding
            
        Returns:
            numpy array of query embeddings
        """
        formatted_queries = [self.get_detailed_instruct(task_description, query) for query in queries]
        outputs = self.model.embed(formatted_queries)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        
        return embeddings.numpy()
    
    def compute_similarity(self, query_embeddings: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document embeddings."""
        return np.dot(query_embeddings, document_embeddings.T)
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        #For Qwen3-Embedding-8B, default dimension is 4096
        return 4096 