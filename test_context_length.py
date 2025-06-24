#!/usr/bin/env python3

import requests
import json
import time

def create_long_prompt(target_words=15000):
    """Create a prompt with approximately target_words words"""
    #Base prompt
    base_prompt = "Please analyze the following text and provide a detailed summary:\n\n"
    
    #Create repetitive content to reach target word count
    content = ""
    sentence = "This is a sample sentence that contains various words and concepts for testing purposes. "
    sentence += "It includes technical terms, scientific concepts, and general knowledge that should help "
    sentence += "us reach the desired token count while maintaining reasonable content quality. "
    sentence += "The sentence continues with additional information about machine learning, artificial intelligence, "
    sentence += "data science, and computational linguistics to provide diverse vocabulary and context. "
    
    #Repeat the sentence to build up content
    word_count = len(base_prompt.split())
    while word_count < target_words:
        content += sentence
        word_count = len((base_prompt + content).split())
    
    full_prompt = base_prompt + content
    
    print(f"Created prompt with ~{len(full_prompt.split())} words")
    print(f"Prompt length: {len(full_prompt)} characters")
    
    return full_prompt

def test_context_length():
    """Test the context length by sending a huge prompt"""
    url = "http://localhost:8000/v1/chat/completions"
    
    #Create a very long prompt (~15K words, which should be ~12K tokens)
    long_prompt = create_long_prompt(15000)
    
    #Prepare the request
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  #Use the model that's running
        "messages": [
            {
                "role": "user",
                "content": long_prompt
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    
    print("Sending request to vLLM endpoint...")
    print(f"URL: {url}")
    print(f"Model: {payload['model']}")
    print(f"Input words: ~{len(long_prompt.split())}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)  #5 minute timeout
        end_time = time.time()
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS: Large context handled successfully!")
            print(f"Response: {result['choices'][0]['message']['content'][:200]}...")
            
            #Check usage statistics
            if 'usage' in result:
                usage = result['usage']
                print(f"\nUsage Statistics:")
                print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Request took too long (>5 minutes)")
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Could not connect to vLLM server")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_context_length() 