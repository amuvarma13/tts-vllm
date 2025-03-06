import torch
import time
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer


# Set up sampling parameters and model
sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=1200)
model_name = "amuvarma/brian-luna-w_emotags-nowhisp"
engine_args = AsyncEngineArgs(model=model_name)
model = AsyncLLMEngine.from_engine_args(engine_args)
tokeniser = AutoTokenizer.from_pretrained(model_name)

# Define special tokens
start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human

# Define prompt(s)
p = "Ugh, You are such a cunty piece of shit. <zac>"
prompts = [p]
all_input_ids = []
for prompt in prompts:
    input_ids = tokeniser(prompt, return_tensors="pt").input_ids
    all_input_ids.append(input_ids)

print("Number of prompts:", len(all_input_ids))

# Concatenate tokens: start token, prompt, then end tokens
all_modified_input_ids = []
for input_ids in all_input_ids:
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    all_modified_input_ids.append(modified_input_ids)

# Left-pad with 128263 to have a uniform length
all_padded_tensors = []
all_attention_masks = []
max_length = max(modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids)
for modified_input_ids in all_modified_input_ids:
    padding = max_length - modified_input_ids.shape[1]
    padded_tensor = torch.cat(
        [torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids],
        dim=1
    )
    attention_mask = torch.cat(
        [torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)],
        dim=1
    )
    all_padded_tensors.append(padded_tensor)
    all_attention_masks.append(attention_mask)

all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
all_attention_masks = torch.cat(all_attention_masks, dim=0)
print("all_padded_tensors:", all_padded_tensors[0].tolist())

input_ids = all_padded_tensors[0].tolist()
print("input_ids:", input_ids)

iids_string = tokeniser.decode(input_ids)
print("iids_string:", iids_string)

# Determine the initial token count (prompt tokens)
initial_tokens = len(tokeniser(iids_string, return_tensors="pt").input_ids[0])
print("Initial token count (prompt tokens):", initial_tokens)

# Define thresholds (generated tokens excluding the prompt)
thresholds = [7,28, 150, 500]
recorded_thresholds = {}

# Async function for streaming generation and tracking time at thresholds
async def stream_generation(prompt_string):
    start_time = time.monotonic()
    results_generator = model.generate(prompt_string, sampling_params, request_id=time.monotonic())
    previous_text = ""
    
    async for request_output in results_generator:
        text = request_output.outputs[0].text
        new_text = text[len(previous_text):]
        # print(new_text, end='', flush=True)
        previous_text = text
        
        # Compute current token count and the number of tokens generated (excluding the prompt)
        current_total_tokens = len(tokeniser(text, return_tensors="pt").input_ids[0])
        generated_tokens = current_total_tokens - initial_tokens

        # Check if any thresholds have been reached
        for th in thresholds:
            if generated_tokens >= th and th not in recorded_thresholds:
                elapsed = time.monotonic() - start_time
                recorded_thresholds[th] = elapsed
                print(f"\nReached {th} tokens in {elapsed:.2f} seconds\n")
                
        # Stop generation if all thresholds have been reached
        if len(recorded_thresholds) == len(thresholds):
            break

# Run the asynchronous streaming generation
asyncio.run(stream_generation(iids_string))
