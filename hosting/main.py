from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn
from model_service import CrossEncoderService
from prompt_guard_service import PromptGuardService

app = FastAPI(title="Cross-Encoder Reranking and Prompt Guard Service")
model_service = CrossEncoderService()
prompt_guard_service = PromptGuardService()

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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) 