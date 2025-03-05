from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import time
import sys

# Set up model and parameters
sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=100)
model_name = "amuvarma/brian-luna-w_emotags-nowhisp"
llm = LLM(model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare prompt
prompt = "Ugh, You are such a cunty piece of shit. <brian>"

# Prepare tokens as in your code
start_token = torch.tensor([[ 128259]], dtype=torch.int64)  # Start of human
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human

# Tokenize prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Concatenate all tokens
modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)  # SOH SOT Text EOT EOH

# Method 1: Using the streaming API
def stream_tokens():
    print("Streaming tokens method:")
    
    # Create streaming generator
    outputs_generator = llm.generate(
        prompt_token_ids=modified_input_ids[0].tolist(),
        sampling_params=sampling_params,
        stream=True  # Enable streaming
    )
    
    generated_text = ""
    for request_output in outputs_generator:
        for output in request_output.outputs:
            # Get the newly generated token(s)
            new_text = output.text[len(generated_text):]
            if new_text:
                print(f"New token: {new_text!r}", end=" ")
                # Show token ID as well
                new_token_ids = tokenizer.encode(new_text, add_special_tokens=False)
                print(f"(ID: {new_token_ids})")
                sys.stdout.flush()  # Ensure output is displayed immediately
                generated_text = output.text
                # time.sleep(0.1)  # Small delay to make the output easier to follow
    
    print(f"\nFinal text: {generated_text!r}")

# Method 2: Monitoring via callback function
def stream_with_callback():
    print("\nCallback method:")
    
    # Define a callback function that will be called on each new token
    generated_text = ""
    
    def token_callback(output):
        nonlocal generated_text
        # Get only the newly generated text
        for output_item in output.outputs:
            new_text = output_item.text[len(generated_text):]
            if new_text:
                print(f"New token: {new_text!r}", end=" ")
                # Show token ID as well
                new_token_ids = tokenizer.encode(new_text, add_special_tokens=False)
                print(f"(ID: {new_token_ids})")
                sys.stdout.flush()
                generated_text = output_item.text
                time.sleep(0.1)  # Small delay
    
    # Generate with the callback
    outputs = llm.generate(
        prompt_token_ids=modified_input_ids[0].tolist(),
        sampling_params=sampling_params,
        stream=True,
        on_stream_output=token_callback
    )
    
    # Collect the final output
    final_output = None
    for output in outputs:
        final_output = output
    
    if final_output:
        print(f"\nFinal text (callback): {final_output.outputs[0].text!r}")

# Run both methods
stream_tokens()
# Uncomment to use the callback method instead
# stream_with_callback()