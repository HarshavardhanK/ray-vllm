import requests
import json
import time
import numpy as np

def test_vllm_embedding_service():
    """Test the vLLM embedding service endpoints."""
    base_url = "http://localhost:8001"
    
    print("Testing vLLM Embedding Service...")
    print("=" * 50)
    
    #Test 1: Get embedding model info
    print("\n1. Testing /embed/info endpoint...")
    try:
        response = requests.get(f"{base_url}/embed/info")
        if response.status_code == 200:
            info = response.json()
            print("✓ Sentence Transformers:")
            print(f"  Model: {info['sentence_transformers']['model_name']}")
            print(f"  Dimension: {info['sentence_transformers']['embedding_dimension']}")
            print("✓ vLLM:")
            print(f"  Model: {info['vllm']['model_name']}")
            print(f"  Dimension: {info['vllm']['embedding_dimension']}")
            print(f"  Advantages: {', '.join(info['vllm']['advantages'])}")
        else:
            print(f"✗ Failed to get model info: {response.status_code}")
    except Exception as e:
        print(f"✗ Error getting model info: {e}")
    
    #Test 2: Embed documents only (most common use case for document storage)
    print("\n2. Testing /vllm/embed/documents endpoint...")
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.",
        "Python is a high-level programming language known for its simplicity and readability."
    ]
    
    try:
        response = requests.post(
            f"{base_url}/vllm/embed/documents",
            json={"documents": documents}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Successfully embedded {len(result['embeddings'])} documents")
            print(f"✓ Embedding dimension: {result['embedding_dimension']}")
            print(f"✓ Model used: {result['model_name']}")
            
            #Check embedding shapes
            embeddings = np.array(result['embeddings'])
            print(f"✓ Embedding shape: {embeddings.shape}")
            
            #Check that embeddings are normalized (cosine similarity ready)
            norms = np.linalg.norm(embeddings, axis=1)
            print(f"✓ Embedding norms (should be ~1.0): {norms[:3]}...")
        else:
            print(f"✗ Failed to embed documents: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"✗ Error embedding documents: {e}")
    
    #Test 3: Embed queries only
    print("\n3. Testing /vllm/embed/queries endpoint...")
    queries = [
        "What is the capital of China?",
        "Explain gravity",
        "How does machine learning work?",
        "What is Python programming?"
    ]
    
    try:
        response = requests.post(
            f"{base_url}/vllm/embed/queries",
            json={
                "queries": queries,
                "task_description": "Given a web search query, retrieve relevant passages that answer the query"
            }
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Successfully embedded {len(result['embeddings'])} queries")
            print(f"✓ Embedding dimension: {result['embedding_dimension']}")
            print(f"✓ Model used: {result['model_name']}")
        else:
            print(f"✗ Failed to embed queries: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"✗ Error embedding queries: {e}")
    
    #Test 4: Embed queries and documents with similarity computation
    print("\n4. Testing /vllm/embed/queries-documents endpoint...")
    test_queries = [
        "What is the capital of China?",
        "Explain gravity"
    ]
    test_documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    
    try:
        response = requests.post(
            f"{base_url}/vllm/embed/queries-documents",
            json={
                "queries": test_queries,
                "documents": test_documents,
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
            for i, query in enumerate(test_queries):
                print(f"Query: {query}")
                for j, doc in enumerate(test_documents):
                    similarity = result['similarities'][i][j]
                    print(f"  Document {j+1}: {similarity:.4f}")
                print()
        else:
            print(f"✗ Failed to compute similarities: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"✗ Error computing similarities: {e}")
    
    #Test 5: Compare vLLM vs sentence-transformers performance
    print("\n5. Comparing vLLM vs sentence-transformers performance...")
    test_texts = ["This is a test document for performance comparison."] * 10
    
    #Test vLLM
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/vllm/embed",
            json={"texts": test_texts}
        )
        vllm_time = time.time() - start_time
        if response.status_code == 200:
            print(f"✓ vLLM time: {vllm_time:.3f}s")
        else:
            print(f"✗ vLLM failed: {response.status_code}")
    except Exception as e:
        print(f"✗ vLLM error: {e}")
        vllm_time = float('inf')
    
    #Test sentence-transformers
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={"texts": test_texts}
        )
        st_time = time.time() - start_time
        if response.status_code == 200:
            print(f"✓ Sentence-transformers time: {st_time:.3f}s")
        else:
            print(f"✗ Sentence-transformers failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Sentence-transformers error: {e}")
        st_time = float('inf')
    
    if vllm_time != float('inf') and st_time != float('inf'):
        if vllm_time < st_time:
            print(f"✓ vLLM is {st_time/vllm_time:.1f}x faster")
        else:
            print(f"✓ Sentence-transformers is {vllm_time/st_time:.1f}x faster")
    
    print("\n" + "=" * 50)
    print("vLLM embedding service tests completed!")

if __name__ == "__main__":
    test_vllm_embedding_service() 