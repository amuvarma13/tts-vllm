from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
model_name = "amuvarma/luna-tts-tags"
llm = LLM(model="amuvarma/luna-tts-tags")
tokeniser = AutoTokenizer.from_pretrained(model_name)

tokens = tokeniser.encode("Hello, my name is")["input_ids"]
print(tokens)


outputs = llm.generate(prompt_token_ids=tokens, sampling_params=sampling_params, max_length=100)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")