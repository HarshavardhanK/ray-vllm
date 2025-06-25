import requests
import json
import time

def test_embedding_service():
    """Test the embedding service endpoints."""
    base_url = "http://localhost:8001"
    
    print("Testing Embedding Service...")
    print("=" * 50)
    
    #Test 1: Get embedding model info
    print("\n1. Testing /embed/info endpoint...")
    try:
        response = requests.get(f"{base_url}/embed/info")
        if response.status_code == 200:
            info = response.json()
            print(f"✓ Model: {info['model_name']}")
            print(f"✓ Embedding dimension: {info['embedding_dimension']}")
            print(f"✓ Max context length: {info['max_context_length']}")
            print(f"✓ Supported languages: {info['supported_languages']}")
        else:
            print(f"✗ Failed to get model info: {response.status_code}")
    except Exception as e:
        print(f"✗ Error getting model info: {e}")
    
    #Test 2: Embed simple texts
    print("\n2. Testing /embed endpoint...")
    test_texts = [
        "What is the capital of China?",
        "Explain gravity",
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other."
    ]
    
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={
                "texts": test_texts,
                "normalize": True
            }
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Successfully embedded {len(result['embeddings'])} texts")
            print(f"✓ Embedding dimension: {result['embedding_dimension']}")
            print(f"✓ Model used: {result['model_name']}")
        else:
            print(f"✗ Failed to embed texts: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"✗ Error embedding texts: {e}")
    
    #Test 3: Embed queries and documents with similarity computation
    print("\n3. Testing /embed/queries-documents endpoint...")
    queries = [
        "What is the capital of China?",
        "Explain gravity"
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    
    try:
        response = requests.post(
            f"{base_url}/embed/queries-documents",
            json={
                "queries": queries,
                "documents": documents,
                "task_description": "Given a web search query, retrieve relevant passages that answer the query"
            }
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Successfully computed embeddings and similarities")
            print(f"✓ Query embeddings: {len(result['query_embeddings'])}")
            print(f"✓ Document embeddings: {len(result['document_embeddings'])}")
            print(f"✓ Similarity matrix shape: {len(result['similarities'])}x{len(result['similarities'][0])}")
            
            #Print similarity scores
            print("\nSimilarity scores:")
            for i, query in enumerate(queries):
                print(f"Query: {query}")
                for j, doc in enumerate(documents):
                    similarity = result['similarities'][i][j]
                    print(f"  Document {j+1}: {similarity:.4f}")
                print()
        else:
            print(f"✗ Failed to compute similarities: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"✗ Error computing similarities: {e}")
    
    #Test 4: Test with custom task description
    print("\n4. Testing with custom task description...")
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={
                "texts": ["Find information about machine learning"],
                "task_description": "Given a research query, find relevant academic papers and technical documents",
                "normalize": True
            }
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Successfully embedded with custom task description")
            print(f"✓ Embedding dimension: {result['embedding_dimension']}")
        else:
            print(f"✗ Failed with custom task: {response.status_code}")
    except Exception as e:
        print(f"✗ Error with custom task: {e}")
    
    print("\n" + "=" * 50)
    print("Embedding service tests completed!")

if __name__ == "__main__":
    test_embedding_service() 