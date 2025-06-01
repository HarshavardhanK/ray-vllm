from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import torch
import uvicorn
from typing import Optional, List

app = FastAPI(title="vLLM Inference Server")

# Initialize the model
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.9,
    enforce_eager=True,
    max_num_batched_tokens=4096,
    max_num_seqs=256,
    max_model_len=2048,
    dtype="bfloat16"
)

class GenerationRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 128
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    try:
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty
        )
        
        # Generate response
        outputs = llm.generate([request.prompt], sampling_params)
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 