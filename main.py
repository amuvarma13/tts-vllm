from flask import Flask, Response, request
import time
import threading
import queue
import asyncio
import struct
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from tokens_decoder import dummy_processor

app = Flask(__name__)

# --- Global state for prompt ordering ---
prompt_queue = []
next_prompt_id = 1
queue_lock = threading.Lock()

# --- Set up sampling parameters and model (loaded only once) ---
sampling_params = SamplingParams(temperature=0.9, top_p=0.6, max_tokens=2000, repetition_penalty=1.1, stop_token_ids=[128258])
model_name = "amuvarma/bl-2"
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

# --- Create WAV Header ---
def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    # In streaming, we don't know the final file size, so we use placeholder values.
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
                         b'RIFF', 0,       # File size placeholder
                         b'WAVE',
                         b'fmt ', 16,      # Subchunk1Size for PCM
                         1,                # AudioFormat (PCM)
                         channels,
                         sample_rate,
                         byte_rate,
                         block_align,
                         bits_per_sample,
                         b'data', 0)       # Data size placeholder
    return header

# --- Asynchronous token generation ---
def async_token_generator(prompt_string, initial_tokens):
    async def generator():
        results_generator = model.generate(prompt_string, sampling_params, request_id=time.monotonic())
        previous_text = ""
        async for request_output in results_generator:
            text = request_output.outputs[0].text
            new_text = text[len(previous_text):]
            previous_text = text
            yield new_text
    return generator()

# --- Synchronous generator wrapping the async generation and processing ---
def sse_event_stream(prompt):
    # First, yield the WAV header
    wav_header = create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1)
    yield wav_header

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
    # (Optional) You can log or use the prompt's queue information here.

    # This function runs in a background thread to push tokens into the queue.
    def run_async_gen():
        async def run():
            async for token in async_token_generator(prompt_string, initial_tokens):
                # Here, assume the dummy_processor and token generator produce raw PCM audio bytes.
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

    # Apply the dummy processor to transform raw tokens into groups of 7 (audio bytes).
    for processed_token in dummy_processor(raw_tokens()):
        yield processed_token
    
    # Cleanup: remove the prompt from the queue.
    with queue_lock:
        if prompt_id in prompt_queue:
            prompt_queue.remove(prompt_id)

# --- Flask audio streaming endpoint ---
@app.route('/events', methods=['GET'])
def sse():
    prompt = request.args.get('prompt', 'No prompt provided')
    return Response(sse_event_stream(prompt), mimetype='audio/wav')

if __name__ == '__main__':
    # Disable the reloader to prevent multiple model loads.
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8080)
