from vllm import LLM
from vllm.sampling_params import SamplingParams

model_name = "mistralai/Mistral-Small-Instruct-2409"
sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(
    model=model_name,
    tokenizer_mode="mistral",
    load_format="mistral",
    config_format="mistral",
)

messages = [
    {
        "role": "user",
        "content": "Who is the best French painter. Answer with detailed explanations.",
    }
]

res = llm.chat(messages=messages, sampling_params=sampling_params)
print(res[0].outputs[0].text)
