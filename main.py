from flask import Flask, Response, request
from flask_cors import CORS
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
CORS(app)

# --- Global state for request management ---
request_queue = queue.Queue()
processing_lock = threading.Lock()
loop = asyncio.new_event_loop()  # Single event loop for all async operations
engine_thread = None
is_engine_running = False

# --- Set up sampling parameters and model ---
sampling_params = SamplingParams(temperature=0.9, top_p=0.6, max_tokens=2000, repetition_penalty=1.1, stop_token_ids=[128258])
model_name = "amuvarma/bl-2"
engine_args = AsyncEngineArgs(model=model_name, dtype=torch.float16)
model = AsyncLLMEngine.from_engine_args(engine_args)
tokeniser = AutoTokenizer.from_pretrained(model_name)
print("Model loaded.")

# --- Define special tokens ---
start_token = torch.tensor([[128259]], dtype=torch.int64)
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

# --- Preprocess the prompt ---
def process_prompt(prompt):
    prompt = prompt + " " + "<zac>"
    input_ids = tokeniser(prompt, return_tensors="pt").input_ids
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    iids_string = tokeniser.decode(modified_input_ids[0].tolist())
    initial_tokens = len(tokeniser(iids_string, return_tensors="pt").input_ids[0])
    return iids_string, initial_tokens

# --- Create WAV Header ---
def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
                         b'RIFF', 0,
                         b'WAVE',
                         b'fmt ', 16,
                         1,
                         channels,
                         sample_rate,
                         byte_rate,
                         block_align,
                         bits_per_sample,
                         b'data', 0)
    return header

# --- Engine worker that processes requests sequentially ---
def engine_worker():
    global is_engine_running
    
    async def process_queue():
        while True:
            try:
                # Get the next request from the queue
                request_data = request_queue.get(block=False)
                if request_data is None:
                    # None is our signal to exit
                    break
                    
                prompt, token_queue, request_id = request_data
                
                # Process this request
                prompt_string, initial_tokens = process_prompt(prompt)
                
                try:
                    # Generate tokens
                    results_generator = model.generate(prompt_string, sampling_params, request_id=request_id)
                    previous_text = ""
                    
                    async for request_output in results_generator:
                        text = request_output.outputs[0].text
                        new_text = text[len(previous_text):]
                        previous_text = text
                        
                        # Put the raw token into the queue
                        if new_text:
                            token_queue.put(new_text)
                            
                except Exception as e:
                    print(f"Error generating response: {e}")
                finally:
                    # Signal that generation is complete
                    token_queue.put(None)
                    request_queue.task_done()
                    
            except queue.Empty:
                # No requests in queue, sleep briefly before checking again
                await asyncio.sleep(0.1)
    
    # Set up the asyncio event loop in this thread
    asyncio.set_event_loop(loop)
    is_engine_running = True
    
    try:
        loop.run_until_complete(process_queue())
    finally:
        loop.close()
        is_engine_running = False

# --- Start the engine worker thread if not already running ---
def ensure_engine_thread():
    global engine_thread, is_engine_running
    
    with processing_lock:
        if engine_thread is None or not engine_thread.is_alive():
            engine_thread = threading.Thread(target=engine_worker, daemon=True)
            engine_thread.start()
            # Give the thread a moment to start up
            time.sleep(0.1)

# --- Flask endpoint for audio streaming ---
@app.route('/events', methods=['GET'])
def sse():
    prompt = request.args.get('prompt', 'No prompt provided')
    
    # Create a queue for the raw tokens
    token_queue = queue.Queue()
    
    # Generate a unique request ID
    request_id = f"{time.time()}-{hash(prompt) % 10000}"
    
    # Make sure the engine thread is running
    ensure_engine_thread()
    
    # Add this request to the queue
    request_queue.put((prompt, token_queue, request_id))
    
    def event_stream():
        # First, yield the WAV header
        wav_header = create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1)
        yield wav_header
        
        # Helper function to yield tokens from the queue
        def raw_tokens():
            while True:
                token = token_queue.get()
                if token is None:
                    break
                yield token
        
        # Process tokens through the dummy processor
        for processed_token in dummy_processor(raw_tokens()):
            print("Sending token")
            yield processed_token
    
    return Response(event_stream(), mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8080, threaded=True)