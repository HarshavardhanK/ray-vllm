from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
import uvicorn
from model_service import CrossEncoderService
from prompt_guard_service import PromptGuardService
from embedding_service import EmbeddingService
from vllm_embedding_service import VLLMEmbeddingService

app = FastAPI(title="Cross-Encoder Reranking, Prompt Guard, and Embedding Service")
model_service = CrossEncoderService()
prompt_guard_service = PromptGuardService()
embedding_service = EmbeddingService()
vllm_embedding_service = VLLMEmbeddingService()

class RerankRequest(BaseModel):
    query: str
    passages: List[str]

class RerankResponse(BaseModel):
    scores: List[float]
    ranked_passages: List[Tuple[str, float]]

class ValidateInputRequest(BaseModel):
    text: str

class ValidateInputResponse(BaseModel):
    is_malicious: bool
    label: str
    confidence: float
    input_text: str

class EmbedRequest(BaseModel):
    texts: List[str]
    task_description: Optional[str] = None
    max_length: Optional[int] = 8192
    normalize: Optional[bool] = True

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    embedding_dimension: int

class EmbedQueriesDocumentsRequest(BaseModel):
    queries: List[str]
    documents: List[str]
    task_description: Optional[str] = "Given a web search query, retrieve relevant passages that answer the query"

class EmbedQueriesDocumentsResponse(BaseModel):
    query_embeddings: List[List[float]]
    document_embeddings: List[List[float]]
    similarities: List[List[float]]
    model_name: str

class VLLMEmbedRequest(BaseModel):
    texts: List[str]
    task_description: Optional[str] = None

class VLLMEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    embedding_dimension: int

class VLLMEmbedDocumentsRequest(BaseModel):
    documents: List[str]

class VLLMEmbedQueriesRequest(BaseModel):
    queries: List[str]
    task_description: Optional[str] = "Given a web search query, retrieve relevant passages that answer the query"

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    try:
        scores = model_service.predict(request.query, request.passages)
        #Pair passages with their scores and sort by score in descending order
        ranked_pairs = sorted(zip(request.passages, scores), key=lambda x: x[1], reverse=True)
        return RerankResponse(
            scores=scores,
            ranked_passages=ranked_pairs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate/input", response_model=ValidateInputResponse)
async def validate_input(request: ValidateInputRequest):
    try:
        result = prompt_guard_service.validate_input(request.text)
        return ValidateInputResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Embed a list of texts using Qwen3-Embedding-8B model (sentence-transformers)."""
    try:
        embeddings = embedding_service.encode_texts(
            texts=request.texts,
            task_description=request.task_description,
            max_length=request.max_length,
            normalize=request.normalize
        )
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model_name=embedding_service.model_name,
            embedding_dimension=embedding_service.get_embedding_dimension()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/queries-documents", response_model=EmbedQueriesDocumentsResponse)
async def embed_queries_and_documents(request: EmbedQueriesDocumentsRequest):
    """Embed queries and documents, then compute similarities (sentence-transformers)."""
    try:
        result = embedding_service.embed_queries_and_documents(
            queries=request.queries,
            documents=request.documents,
            task_description=request.task_description
        )
        return EmbedQueriesDocumentsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vllm/embed", response_model=VLLMEmbedResponse)
async def vllm_embed_texts(request: VLLMEmbedRequest):
    """Embed a list of texts using vLLM with Qwen3-Embedding-8B model (more efficient for batch processing)."""
    try:
        embeddings = vllm_embedding_service.embed_texts(
            texts=request.texts,
            task_description=request.task_description
        )
        return VLLMEmbedResponse(
            embeddings=embeddings.tolist(),
            model_name=vllm_embedding_service.model_name,
            embedding_dimension=vllm_embedding_service.get_embedding_dimension()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vllm/embed/documents", response_model=VLLMEmbedResponse)
async def vllm_embed_documents_only(request: VLLMEmbedDocumentsRequest):
    """Embed only documents using vLLM (optimized for document storage/indexing)."""
    try:
        embeddings = vllm_embedding_service.embed_documents_only(
            documents=request.documents
        )
        return VLLMEmbedResponse(
            embeddings=embeddings.tolist(),
            model_name=vllm_embedding_service.model_name,
            embedding_dimension=vllm_embedding_service.get_embedding_dimension()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vllm/embed/queries", response_model=VLLMEmbedResponse)
async def vllm_embed_queries_only(request: VLLMEmbedQueriesRequest):
    """Embed only queries with instruction using vLLM (optimized for search operations)."""
    try:
        embeddings = vllm_embedding_service.embed_queries_only(
            queries=request.queries,
            task_description=request.task_description
        )
        return VLLMEmbedResponse(
            embeddings=embeddings.tolist(),
            model_name=vllm_embedding_service.model_name,
            embedding_dimension=vllm_embedding_service.get_embedding_dimension()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vllm/embed/queries-documents", response_model=EmbedQueriesDocumentsResponse)
async def vllm_embed_queries_and_documents(request: EmbedQueriesDocumentsRequest):
    """Embed queries and documents, then compute similarities using vLLM (most efficient for batch processing)."""
    try:
        result = vllm_embedding_service.embed_queries_and_documents(
            queries=request.queries,
            documents=request.documents,
            task_description=request.task_description
        )
        return EmbedQueriesDocumentsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embed/info")
async def get_embedding_info():
    """Get information about the embedding models."""
    return {
        "sentence_transformers": {
            "model_name": embedding_service.model_name,
            "embedding_dimension": embedding_service.get_embedding_dimension(),
            "max_context_length": 32768,
            "supported_languages": "100+ Languages",
            "model_type": "Text Embedding"
        },
        "vllm": {
            "model_name": vllm_embedding_service.model_name,
            "embedding_dimension": vllm_embedding_service.get_embedding_dimension(),
            "max_context_length": 32768,
            "supported_languages": "100+ Languages",
            "model_type": "Text Embedding (vLLM optimized)",
            "advantages": ["Faster batch processing", "Better memory efficiency", "Optimized for production"]
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) 