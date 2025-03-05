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

# Prepare prompt and tokens
prompt = "Ugh, You are such a cunty piece of shit. <brian>"
start_token = torch.tensor([[ 128259]], dtype=torch.int64)  # Start of human
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)  # SOH SOT Text EOT EOH

# Define the async streaming function
async def stream_tokens():
    # Create a unique request ID
    request_id = str(time.monotonic())
    
    # Start the generation
    results_generator = llm.generate(
        prompt_token_ids=modified_input_ids[0].tolist(),
        sampling_params=sampling_params,
        request_id=request_id,  # Pass the request ID
        stream=True  # Enable streaming
    )
    
    outputs = ""
    async for request_output in results_generator:
        if request_output.finished:
            print("\n[Generation completed]")
        else:
            out = request_output.outputs[0].text
            if len(out) == 0:
                continue
            out_delta = out[len(outputs):]
            print(out_delta, end="", flush=True)
            outputs = out
    
    return outputs

# Function to run the async code
def run_streaming():
    # Create and run the event loop
    loop = asyncio.get_event_loop()
    final_output = loop.run_until_complete(stream_tokens())
    print(f"\nFinal output: {final_output}")

# Run the streaming function
if __name__ == "__main__":
    run_streaming()

# Alternative version using asyncio.run() (Python 3.7+)
# This is a simpler way to run async code if you're using Python 3.7 or later

"""
import asyncio

# ... (same code as above) ...

if __name__ == "__main__":
    final_output = asyncio.run(stream_tokens())
    print(f"\nFinal output: {final_output}")
"""