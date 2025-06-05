from openai import OpenAI
import torch
import torch.cuda
from numba import cuda
import numpy as np

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

# Initialize CUDA streams for parallel processing
streams = [cuda.stream() for _ in range(4)]

def optimize_memory():
    # Clear CUDA cache
    torch.cuda.empty_cache()
    # Set memory allocation strategy
    torch.cuda.set_per_process_memory_fraction(0.8)

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Optimize memory before processing
optimize_memory()

# Get number of inputs for batch processing
num_inputs = int(input("How many questions do you want to ask? "))
messages_batch = []

# Collect inputs
for i in range(num_inputs):
    user_message = input(f"Enter question {i+1}: ")
    messages_batch.append([{"role": "user", "content": user_message}])

# Process batch with streams
results = []
for i, messages in enumerate(messages_batch):
    with streams[i % len(streams)]:
        response = client.chat.completions.create(model=model, messages=messages)
        results.append({
            'reasoning': response.choices[0].message.reasoning_content,
            'content': response.choices[0].message.content
        })

# Display results
for i, result in enumerate(results):
    print(f"\nQuestion {i+1} Results:")
    print("reasoning_content:", result['reasoning'])
    print("content:", result['content'])