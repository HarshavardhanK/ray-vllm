# vLLM OpenAI-Compatible API Server

This server provides an OpenAI-compatible API endpoint for running Llama 2 7B model using vLLM.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have access to Llama 2 7B model from Meta. You'll need to:
   - Accept the model license on Hugging Face
   - Have a Hugging Face token with access to the model

## Running the Server

Basic usage:
```bash
python server.py
```

Advanced options:
```bash
python server.py --host 0.0.0.0 --port 8000 --model meta-llama/Llama-2-7b-chat-hf --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 4096
```

## API Usage

The server implements OpenAI's chat completions API. You can use it with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key is not required for local deployment
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

## Notes

- The server uses vLLM's PagedAttention for efficient memory management
- Default configuration uses 90% of available GPU memory
- Maximum sequence length is set to 4096 tokens by default
- For multi-GPU setups, adjust tensor-parallel-size accordingly 