from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import time

sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=500)
model_name = "amuvarma/brian-luna-w_emotags-nowhisp"
llm = LLM(model=model_name, pipeline_parallel_size=2)
tokeniser = AutoTokenizer.from_pretrained(model_name)


start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human
p= "Ugh, You are such a cunty piece of shit. <zac>",
prompts = [
   p 
]
all_input_ids = []
for prompt in prompts:
  input_ids = tokeniser(prompt, return_tensors="pt").input_ids
  all_input_ids.append(input_ids)

print(len(all_input_ids))

start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human

# # Concatenate the tensors
all_modified_input_ids = []
for input_ids in all_input_ids:
  modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
  all_modified_input_ids.append(modified_input_ids)

#now convert to tensor by left padding with 128263
all_padded_tensors = []
all_attention_masks = []
#get longest modified_tensors
max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
for modified_input_ids in all_modified_input_ids:
  padding = max_length - modified_input_ids.shape[1]
  padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
  attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
  all_padded_tensors.append(padded_tensor)
  all_attention_masks.append(attention_mask)

all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
all_attention_masks = torch.cat(all_attention_masks, dim=0)
print("all_padded_tensors", all_padded_tensors[0].tolist())
input_ids = all_padded_tensors[0].tolist()
# attention_mask = torch.ones_like(new_input_ids).to("cuda")

# input_ids = tokeniser(p, return_tensors="pt").input_ids
start_time = time.monotonic()
outputs = llm.generate(
    prompt_token_ids=input_ids, 
    sampling_params=sampling_params, 
)
end_time = time.monotonic()

print(f"Generation took {end_time - start_time:.2f} seconds")


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")