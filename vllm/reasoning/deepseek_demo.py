from openai import OpenAI
import torch
import torch.cuda

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

def optimize_memory():
    # Clear CUDA cache
    torch.cuda.empty_cache()
    # Set memory allocation strategy
    torch.cuda.set_per_process_memory_fraction(0.8)

def create_prompt_template(question):
    # Create a structured prompt template for reasoning tasks
    prompt = f"""
    
                # Query Complexity Analysis and Decomposition Prompt

You are an expert query analyzer that determines whether a user query is simple or complex, and breaks down complex queries into optimal sub-queries for better retrieval.

## Your Task
Analyze the following query and determine if it needs to be broken down into multiple questions for better information retrieval.

**Original Query:** {question}

## Complexity Assessment Guidelines

### Simple Queries (output as-is):
- **Single entity focus**: "What was Apple's revenue in 2023?"
- **Single concept requests**: "How does machine learning work?"
- **Single comparison**: "Is Python faster than Java?"
- **Single time period**: "What happened in the 2008 financial crisis?"
- **Single location**: "What's the population of Tokyo?"

### Complex Queries (requires decomposition):
- **Multiple entities**: "What were Apple's and Google's revenue in 2023?"
- **Multiple time periods**: "Compare unemployment rates in 2020 vs 2023"
- **Multiple concepts**: "Explain machine learning, deep learning, and neural networks"
- **Multi-step reasoning**: "How did the 2008 crisis affect tech stocks and what lessons apply today?"
- **Multiple comparisons**: "Compare the GDP of US, China, and EU in 2023"
- **Compound questions**: "What is climate change and how does it affect agriculture and economy?"

## Instructions

1. **First, assess complexity**: Determine if the query contains multiple distinct information needs
2. **If SIMPLE**: Output the original query exactly as provided
3. **If COMPLEX**: Break it down into focused sub-queries that:
   - Each target a single entity, concept, or comparison
   - Can be answered independently
   - Cover all aspects of the original query
   - Are optimized for retrieval systems

## Output Format

**For Simple Queries:**
```
[
    Question 1: <original query>
]
```

**For Complex Queries:**
```
[
    Question 1: <focused_sub_query_1>,
    Question 2: <focused_sub_query_2>,
    Question 3: <focused_sub_query_3>,
    ...
]
```

## Examples

**Example 1 - Simple:**
Query: "What was Apple's revenue in 2023?"
Output: `[Question 1: What was Apple's revenue in 2023?]`

**Example 2 - Complex:**
Query: "What were Apple's and Google's revenue in 2023?"
Output: 
```
[
    Question 1: What was Apple's revenue in 2023?,
    Question 2: What was Google's revenue in 2023?
]
```

**Example 3 - Complex:**
Query: "How did the 2008 financial crisis affect tech companies and what recovery strategies worked?"
Output:
```
[
    Question 1: How did the 2008 financial crisis affect tech companies?,
    Question 2: What recovery strategies did tech companies use after the 2008 financial crisis?,
    Question 3: Which recovery strategies were most effective for tech companies after 2008?
]
```

Now analyze the query: {question}
                
"""
    
    return prompt

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Optimize memory before processing
optimize_memory()

# Get user input
user_question = input("Enter your question: ")

# Create prompt using template
prompt_content = create_prompt_template(user_question)

# Prepare messages for the API call
messages = [{"role": "user", "content": prompt_content}]

# Get response from the model
response = client.chat.completions.create(model=model, messages=messages)

# Display results
print(f"\nQuestion: {user_question}")
print("=" * 50)
if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
    print("Reasoning:", response.choices[0].message.reasoning_content)
    print("-" * 30)
print("Answer:", response.choices[0].message.content)