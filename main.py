from flask import Flask, Response, request
from flask_cors import CORS
import time
import threading
import queue
import asyncio
import struct
import torch
import logging
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from tokens_decoder import dummy_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- Global state ---
request_queue = queue.Queue()
model_lock = threading.RLock()  # Reentrant lock for model operations
model_health_status = {"healthy": True, "last_error": None, "last_restart": None}

# --- Model Manager Class ---
class LLMModelManager:
    def __init__(self, model_name="amuvarma/bl-2"):
        self.model_name = model_name
        self.model = None
        self.tokeniser = None
        self.loop = None
        self.engine_thread = None
        self.is_running = False
        self.start_token = torch.tensor([[128259]], dtype=torch.int64)
        self.end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
        self.sampling_params = SamplingParams(
            temperature=0.9, 
            top_p=0.6, 
            max_tokens=2000, 
            repetition_penalty=1.1, 
            stop_token_ids=[128258]
        )
        self.initialize_model()

    def initialize_model(self):
        """Initialize the model and tokenizer."""
        with model_lock:
            try:
                logger.info(f"Initializing model: {self.model_name}")
                model_health_status["last_restart"] = time.time()
                
                # Create a new event loop for the model
                self.loop = asyncio.new_event_loop()
                
                # Initialize model in the main thread
                engine_args = AsyncEngineArgs(model=self.model_name, dtype=torch.float16)
                self.model = AsyncLLMEngine.from_engine_args(engine_args)
                self.tokeniser = AutoTokenizer.from_pretrained(self.model_name)
                
                # Start processing thread
                self.is_running = True
                self.engine_thread = threading.Thread(target=self._run_engine_loop, daemon=True)
                self.engine_thread.start()
                
                # Give the thread a moment to start
                time.sleep(0.5)
                
                model_health_status["healthy"] = True
                model_health_status["last_error"] = None
                logger.info("Model initialization complete")
                return True
            except Exception as e:
                model_health_status["healthy"] = False
                model_health_status["last_error"] = str(e)
                logger.error(f"Model initialization failed: {e}")
                return False

    def shutdown(self):
        """Shutdown the model and event loop."""
        with model_lock:
            self.is_running = False
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.engine_thread and self.engine_thread.is_alive():
                self.engine_thread.join(timeout=5)
            logger.info("Model shutdown complete")

    def restart_model(self):
        """Restart the model if it's unhealthy."""
        logger.warning("Attempting to restart the model...")
        self.shutdown()
        return self.initialize_model()

    def process_prompt(self, prompt):
        """Process a prompt by adding tokens and encoding."""
        prompt = prompt + " " + "<zac>"
        input_ids = self.tokeniser(prompt, return_tensors="pt").input_ids
        modified_input_ids = torch.cat([self.start_token, input_ids, self.end_tokens], dim=1)
        iids_string = self.tokeniser.decode(modified_input_ids[0].tolist())
        initial_tokens = len(self.tokeniser(iids_string, return_tensors="pt").input_ids[0])
        return iids_string, initial_tokens

    def _run_engine_loop(self):
        """Run the engine loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        
        async def process_queue():
            while self.is_running:
                try:
                    # Get the next request
                    if not request_queue.empty():
                        request_data = request_queue.get()
                        prompt, token_queue, request_id, attempt = request_data
                        
                        # Process this request
                        await self._process_request(prompt, token_queue, request_id, attempt)
                        request_queue.task_done()
                    else:
                        # No requests, sleep briefly
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Engine loop error: {e}")
                    model_health_status["healthy"] = False
                    model_health_status["last_error"] = str(e)
                    # Don't try to restart here - let the watchdog handle it
                    break
        
        try:
            self.loop.run_until_complete(process_queue())
        except Exception as e:
            logger.error(f"Engine thread crashed with error: {e}")
            model_health_status["healthy"] = False
            model_health_status["last_error"] = str(e)
        finally:
            self.loop.close()
            logger.info("Engine loop closed")

    async def _process_request(self, prompt, token_queue, request_id, attempt):
        """Process a single request and put tokens in the queue."""
        try:
            # Preprocess prompt
            prompt_string, initial_tokens = self.process_prompt(prompt)
            
            # Generate tokens
            results_generator = self.model.generate(
                prompt_string, 
                self.sampling_params, 
                request_id=request_id
            )
            
            previous_text = ""
            async for request_output in results_generator:
                text = request_output.outputs[0].text
                new_text = text[len(previous_text):]
                previous_text = text
                
                if new_text:
                    token_queue.put(new_text)
            
            # Request completed successfully
            token_queue.put(None)  # Signal completion
            
        except Exception as e:
            logger.error(f"Error processing request {request_id} (attempt {attempt}): {e}")
            
            if "Background loop has errored already" in str(e) or "model is not initialized" in str(e):
                # This is a critical vLLM error, mark model as unhealthy
                model_health_status["healthy"] = False
                model_health_status["last_error"] = str(e)
                
                # If this was our first attempt, retry
                if attempt < 3:
                    logger.info(f"Queuing retry for request {request_id}, attempt {attempt+1}")
                    # Put a retry signal in the token queue
                    token_queue.put("RETRY")
                else:
                    # Too many retries, signal error
                    token_queue.put("ERROR")
            else:
                # Other errors, just signal completion with error
                token_queue.put("ERROR")

# --- Helper Functions ---
def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create a WAV header for streaming audio."""
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

def watchdog_thread():
    """Monitor model health and restart if necessary."""
    while True:
        try:
            if not model_health_status["healthy"]:
                logger.warning("Watchdog detected unhealthy model. Attempting restart...")
                if model_manager.restart_model():
                    logger.info("Model restarted successfully")
                else:
                    logger.error("Failed to restart model")
            
            # Check every 5 seconds
            time.sleep(5)
        except Exception as e:
            logger.error(f"Watchdog error: {e}")
            time.sleep(10)  # Back off on errors

# --- Initialize model manager ---
model_manager = LLMModelManager()

# --- Start watchdog ---
threading.Thread(target=watchdog_thread, daemon=True).start()

# --- Flask endpoint for audio streaming ---
@app.route('/events', methods=['GET'])
def sse():
    prompt = request.args.get('prompt', 'No prompt provided')
    
    # Create a queue for tokens
    token_queue = queue.Queue()
    
    # Generate a unique request ID
    request_id = f"{time.time()}-{hash(prompt) % 10000}"
    
    # Add request to queue with attempt number
    request_queue.put((prompt, token_queue, request_id, 1))
    
    def event_stream():
        attempt = 1
        max_attempts = 3
        
        while attempt <= max_attempts:
            # First, yield the WAV header
            wav_header = create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1)
            yield wav_header
            
            # Process tokens
            tokens_processed = False
            retry_requested = False
            
            def raw_tokens():
                nonlocal tokens_processed, retry_requested
                
                while True:
                    token = token_queue.get()
                    
                    # Check for control signals
                    if token is None:
                        # Normal completion
                        tokens_processed = True
                        break
                    elif token == "RETRY":
                        # Retry requested
                        retry_requested = True
                        break
                    elif token == "ERROR":
                        # Error completion
                        tokens_processed = True
                        break
                    else:
                        # Normal token
                        yield token
            
            # Process tokens through dummy processor
            try:
                for processed_token in dummy_processor(raw_tokens()):
                    logger.debug(f"Sending token for request {request_id}")
                    yield processed_token
            except Exception as e:
                logger.error(f"Error processing tokens: {e}")
            
            # Check if we need to retry
            if tokens_processed or attempt >= max_attempts:
                # Either completed successfully or too many attempts
                break
            
            if retry_requested:
                # Wait for the model to restart before retrying
                logger.info(f"Waiting for model restart before retry {attempt+1}")
                time.sleep(2)  # Give watchdog time to restart the model
                
                # Clear the token queue for the retry
                while not token_queue.empty():
                    token_queue.get()
                
                # Requeue the request with incremented attempt count
                attempt += 1
                request_queue.put((prompt, token_queue, request_id, attempt))
    
    return Response(event_stream(), mimetype='audio/wav')

# --- Health check endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    status = {
        "healthy": model_health_status["healthy"],
        "last_error": model_health_status["last_error"],
        "last_restart": model_health_status["last_restart"],
        "pending_requests": request_queue.qsize()
    }
    return status

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8080, threaded=True)