from flask import Flask, Response, request
import time
import threading
import queue
import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer

app = Flask(__name__)

# --- Global state for prompt ordering ---
prompt_queue = []
next_prompt_id = 1
queue_lock = threading.Lock()

# --- Set up sampling parameters and model (loaded only once) ---
sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=1200)
model_name = "amuvarma/brian-luna-w_emotags-nowhisp"
engine_args = AsyncEngineArgs(model=model_name, dtype=torch.float16)
model = AsyncLLMEngine.from_engine_args(engine_args)
tokeniser = AutoTokenizer.from_pretrained(model_name)
print("Model loaded.")

# --- Define special tokens ---
start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human

# --- Preprocess the prompt ---
def process_prompt(prompt):
    input_ids = tokeniser(prompt, return_tensors="pt").input_ids
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    iids_string = tokeniser.decode(modified_input_ids[0].tolist())
    initial_tokens = len(tokeniser(iids_string, return_tensors="pt").input_ids[0])
    return iids_string, initial_tokens

# --- Asynchronous token generation ---
def async_token_generator(prompt_string, initial_tokens):
    async def generator():
        start_time = time.monotonic()
        recorded_thresholds = {}
        results_generator = model.generate(prompt_string, sampling_params, request_id=time.monotonic())
        previous_text = ""
        async for request_output in results_generator:
            text = request_output.outputs[0].text
            # Extract only the new portion of text
            new_text = text[len(previous_text):]
            previous_text = text

            # (Optional) Send threshold updates
            current_total_tokens = len(tokeniser(text, return_tensors="pt").input_ids[0])
            generated_tokens = current_total_tokens - initial_tokens
            for th in [7, 28, 150, 500]:
                if generated_tokens >= th and th not in recorded_thresholds:
                    elapsed = time.monotonic() - start_time
                    recorded_thresholds[th] = elapsed
                    yield f"Reached {th} tokens in {elapsed:.2f} seconds. "
            yield new_text
    return generator()

# --- Dummy processor: groups tokens into batches of 7 ---
def dummy_processor(token_gen):
    buffer = ""
    count = 0
    for token in token_gen:
        # Append the token (which may be a string of text) to the buffer
        buffer += token
        count += 1
        if count == 7:
            yield buffer
            buffer = ""
            count = 0
    # Emit any remaining tokens (if fewer than 7)
    if buffer:
        yield buffer

# --- Synchronous generator wrapping the async generation and processing ---
def sse_event_stream(prompt):
    q = queue.Queue()
    
    # Preprocess the prompt (tokenize and add special tokens)
    prompt_string, initial_tokens = process_prompt(prompt)
    
    # Assign a unique prompt ID and add it to the queue
    with queue_lock:
        global next_prompt_id
        prompt_id = next_prompt_id
        next_prompt_id += 1
        prompt_queue.append(prompt_id)
    with queue_lock:
        position = prompt_queue.index(prompt_id) + 1
    # Immediately send an event with the prompt's queue information.
    q.put(f"Connected. Your prompt ID is {prompt_id}. Queue position: {position}")

    # This function runs in a background thread to push tokens into the queue.
    def run_async_gen():
        async def run():
            async for token in async_token_generator(prompt_string, initial_tokens):
                q.put(token)
            q.put(None)  # Sentinel indicating generation is complete.
        asyncio.run(run())
    
    threading.Thread(target=run_async_gen, daemon=True).start()
    
    # Create a generator to yield raw tokens from the queue.
    def raw_tokens():
        while True:
            token = q.get()
            if token is None:
                break
            yield token

    # Apply the dummy processor to transform raw tokens into groups of 7.
    for processed_token in dummy_processor(raw_tokens()):
        yield f"data: {processed_token}\n\n"
    
    # Cleanup: remove the prompt from the queue.
    with queue_lock:
        if prompt_id in prompt_queue:
            prompt_queue.remove(prompt_id)
    yield "data: Processing complete. Goodbye.\n\n"

# --- Flask SSE endpoint ---
@app.route('/events', methods=['GET'])
def sse():
    prompt = request.args.get('prompt', 'No prompt provided')
    return Response(sse_event_stream(prompt), mimetype='text/event-stream')

if __name__ == '__main__':
    # Disable the reloader to prevent multiple model loads.
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8080)
