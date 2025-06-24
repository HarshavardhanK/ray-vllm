from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import AsyncGenerator
from vllm.entrypoints.openai.api_server import create_app

#Create FastAPI app with vLLM's OpenAI-compatible endpoints
app = create_app()

#Add a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 