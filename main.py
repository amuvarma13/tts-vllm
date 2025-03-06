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

# --- Set up sampling parameters and load model (once) ---
sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=1200)
model_name = "amuvarma/brian-luna-w_emotags-nowhisp"
engine_args = AsyncEngineArgs(model=model_name, dtype=torch.float16)
model = AsyncLLMEngine.from_engine_args(engine_args)
tokeniser = AutoTokenizer.from_pretrained(model_name)

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
            new_text = text[len(previous_text):]
            previous_text = text

            # Compute token counts for threshold updates
            current_total_tokens = len(tokeniser(text, return_tensors="pt").input_ids[0])
            generated_tokens = current_total_tokens - initial_tokens
            for th in [7, 28, 150, 500]:
                if generated_tokens >= th and th not in recorded_thresholds:
                    elapsed = time.monotonic() - start_time
                    recorded_thresholds[th] = elapsed
                    yield f"Reached {th} tokens in {elapsed:.2f} seconds"
            yield new_text
    return generator()

# --- Synchronous generator wrapping the async code with queueing ---
def sse_event_stream(prompt):
    # Create a thread‑safe queue for sending SSE events
    q = queue.Queue()

    # Preprocess the prompt: tokenize, add special tokens, get initial token count
    prompt_string, initial_tokens = process_prompt(prompt)
    
    # Assign a unique prompt ID and add it to the global queue
    with queue_lock:
        global next_prompt_id
        prompt_id = next_prompt_id
        next_prompt_id += 1
        prompt_queue.append(prompt_id)
    with queue_lock:
        position = prompt_queue.index(prompt_id) + 1
    q.put(f"Connected. Your prompt ID is {prompt_id}. Queue position: {position}")

    # Wait until it's this prompt's turn (only one generation at a time)
    while True:
        with queue_lock:
            if prompt_queue[0] == prompt_id:
                break
        q.put("Waiting for your turn...")
        time.sleep(1)
    
    # Start asynchronous generation in a background thread
    def run_async_gen():
        async def run():
            async for token in async_token_generator(prompt_string, initial_tokens):
                q.put(token)
            q.put(None)  # Sentinel to mark completion
        asyncio.run(run())
    
    threading.Thread(target=run_async_gen, daemon=True).start()
    
    # Yield tokens as SSE events from the queue
    while True:
        token = q.get()
        if token is None:
            break
        yield f"data: {token}\n\n"
    
    # Clean up: remove this prompt from the queue and send a final event
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
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8080)
