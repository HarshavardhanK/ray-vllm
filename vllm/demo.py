from vllm import LLM, SamplingParams
import asyncio
import requests
import json
import subprocess
import time

def offline_inference_demo():
    """Demonstrate offline batched inference using vLLM."""
    print("\n=== Offline Inference Demo ===")
    
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
    
    # Define some example prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.85,
        top_p=0.95,
        max_tokens=128
    )
    
    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

def api_server_demo():
    """Demonstrate vLLM API server functionality."""
    print("\n=== API Server Demo ===")
    
    # Start the API server in a separate process
    server_process = subprocess.Popen(
        ["python", "-m", "vllm.entrypoints.api_server",
         "--model", "mistralai/Mistral-7B-v0.1",
         "--trust-remote-code",  # Added trust-remote-code flag
         "--port", "8000"]
    )
    
    # Wait for server to start
    time.sleep(5)
    
    # Make a request to the API
    url = "http://localhost:8000/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.8,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error making API request: {e}")
    finally:
        # Clean up
        server_process.terminate()
        server_process.wait()

def main():
    # Run offline inference demo
    offline_inference_demo()
    
    # Run API server demo
    api_server_demo()

if __name__ == "__main__":
    main()
