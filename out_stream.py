from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import time
import asyncio

# Set up model and parameters
sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=100)
model_name = "amuvarma/brian-luna-w_emotags-nowhisp"
llm = LLM(model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare your input_ids as you had them before
prompt = "Ugh, You are such a cunty piece of shit. <brian>"
start_token = torch.tensor([[ 128259]], dtype=torch.int64)  # Start of human
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

# Define the async streaming function
async def stream_output():
    # Get your input_ids ready as in your original code
    prompt_token_ids = modified_input_ids[0].tolist()
    
    # Use the same parameters you provided
    results_generator = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        stream=True
    )
    
    outputs = ""
    async for request_output in results_generator:
        if request_output.finished:
            print()
        else:
            out = request_output.outputs[0].text
            if len(out) == 0:
                continue
            out_delta = out[len(outputs):]
            print(out_delta, end="", flush=True)
            outputs = out
    
    return outputs

# Run the async function
if __name__ == "__main__":
    asyncio.run(stream_output())