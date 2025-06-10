#!/usr/bin/env python3
#Test client for vLLM OpenAI-compatible API server
from openai import OpenAI
import time

def test_chat_completion():
    #Initialize OpenAI client pointing to our local server
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"  #API key not required for local deployment
    )
    
    #Test message
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    try:
        #Make API call
        response = client.chat.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        #Print response
        print("\nResponse from model:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    print("Testing vLLM API server...")
    #Give server time to start
    time.sleep(5)
    test_chat_completion() 