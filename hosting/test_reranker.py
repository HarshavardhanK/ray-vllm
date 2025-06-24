import requests
import json
from typing import List, Dict
import time

def test_reranker(query: str, passages: List[str]) -> Dict:
    """Test the reranking service with a query and list of passages."""
    url = "http://localhost:8001/rerank"
    
    payload = {
        "query": query,
        "passages": passages
    }
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {query}")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print("\nRanked Results:")
        for passage, score in result["ranked_passages"]:
            print(f"\nScore: {score:.4f}")
            print(f"Passage: {passage}")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def main():
    #Test case 1: Information retrieval
    query1 = "What is the capital of France?"
    passages1 = [
        "Paris is the capital and largest city of France.",
        "The Eiffel Tower is a famous landmark in Paris.",
        "France is known for its wine and cuisine.",
        "The Louvre Museum is located in Paris.",
        "The French Revolution took place in the late 18th century."
    ]
    
    #Test case 2: Technical question
    query2 = "How does a transformer model work?"
    passages2 = [
        "A transformer model uses self-attention mechanisms to process input sequences.",
        "The model was introduced in the paper 'Attention is All You Need'.",
        "Transformers have revolutionized natural language processing tasks.",
        "The architecture consists of encoder and decoder components.",
        "Self-attention allows the model to weigh the importance of different words in a sequence."
    ]
    
    #Test case 3: Factual question
    query3 = "What is the population of Tokyo?"
    passages3 = [
        "Tokyo is the capital city of Japan.",
        "Tokyo has a population of approximately 37 million people.",
        "The city is known for its advanced technology and pop culture.",
        "Tokyo hosted the 2020 Summer Olympics.",
        "The city is located on the eastern coast of Honshu island."
    ]
    
    print("Testing Reranking Service...")
    print("\n" + "="*50)
    
    #Run all test cases
    test_reranker(query1, passages1)
    print("\n" + "="*50)
    
    test_reranker(query2, passages2)
    print("\n" + "="*50)
    
    test_reranker(query3, passages3)

if __name__ == "__main__":
    main() 