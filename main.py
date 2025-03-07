from flask import Flask, Response, request
from flask_cors import CORS
import time
import threading
import queue
import asyncio
import struct
import torch
import gc
import logging
import os
import sys
import signal
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
model_lock = threading.RLock()
model_health_status = {
    "healthy": True, 
    "last_error": None, 
    "last_restart": None,
    "cuda_errors": 0,
    "restart_count": 0
}

# --- GPU Memory Management ---
def reset_gpu_memory():
    """Aggressively reset GPU memory"""
    try:
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Optional: More aggressive memory reset for persistent issues
            if model_health_status["cuda_errors"] > 2:
                logger.warning("Multiple CUDA errors detected. Performing full GPU reset...")
                # Re-initialize CUDA context (extreme measure)
                devices = torch.cuda.device_count()
                for i in range(devices):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    del torch.cuda._tls.contexts
                    torch.cuda._tls.contexts = {}
                gc.collect()
                torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        logger.error(f"Failed to reset GPU memory: {e}")
        return False

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
        
        # Set environment variable to enable CUDA device-side assert tracking
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    def initialize_model(self):
        """Initialize the model and tokenizer."""
        with model_lock:
            try:
                # Make sure GPU memory is clear before initializing
                reset_gpu_memory()
                
                logger.info(f"Initializing model: {self.model_name}")
                model_health_status["last_restart"] = time.time()
                
                # Create a new event loop for the model
                if self.loop is None or self.loop.is_closed():
                    self.loop = asyncio.new_event_loop()
                
                # Initialize model with appropriate error handling
                engine_args = AsyncEngineArgs(
                    model=self.model_name, 
                    dtype=torch.float16,
                    # Set more conservative GPU memory usage
                    gpu_memory_utilization=0.8,  # Use 80% of GPU memory to leave headroom
                    enforce_eager=True  # Helps with synchronous error detection
                )
                
                self.model = AsyncLLMEngine.from_engine_args(engine_args)
                self.tokeniser = AutoTokenizer.from_pretrained(self.model_name)
                
                # Start processing thread
                self.is_running = True
                self.engine_thread = threading.Thread(target=self._run_engine_loop, daemon=True)
                self.engine_thread.start()
                
                # Give the thread a moment to start
                time.sleep(1.0)
                
                model_health_status["healthy"] = True
                model_health_status["last_error"] = None
                model_health_status["restart_count"] += 1
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
            logger.info("Shutting down model...")
            
            # Signal thread to stop
            self.is_running = False
            
            # Cancel any pending tasks in the event loop
            if self.loop and not self.loop.is_closed():
                try:
                    for task in asyncio.all_tasks(self.loop):
                        task.cancel()
                    
                    # Stop the loop if it's running
                    if self.loop.is_running():
                        self.loop.call_soon_threadsafe(self.loop.stop)
                except Exception as e:
                    logger.error(f"Error shutting down event loop: {e}")
            
            # Wait for thread to finish
            if self.engine_thread and self.engine_thread.is_alive():
                try:
                    self.engine_thread.join(timeout=5)
                except Exception as e:
                    logger.error(f"Error joining engine thread: {e}")
            
            # Explicitly clear model references
            self.model = None
            
            # Clear GPU memory
            reset_gpu_memory()
            
            logger.info("Model shutdown complete")

    def restart_model(self):
        """Restart the model with exponential backoff for CUDA errors."""
        logger.warning("Attempting to restart the model...")
        
        # Calculate backoff time based on CUDA error count
        backoff_time = min(2 ** model_health_status["cuda_errors"], 30)  # Max 30 seconds
        
        if "CUDA error" in str(model_health_status["last_error"]) or "cuDNN error" in str(model_health_status["last_error"]):
            model_health_status["cuda_errors"] += 1
            logger.warning(f"CUDA error detected. Backoff time: {backoff_time}s. Error count: {model_health_status['cuda_errors']}")
            time.sleep(backoff_time)
        
        # Shutdown existing model
        self.shutdown()
        
        # Reset Python interpreter if too many CUDA errors
        if model_health_status["cuda_errors"] > 5:
            logger.critical("Too many CUDA errors. Requesting server restart...")
            # Signal parent process to restart the server
            with open('/tmp/vllm_restart_needed', 'w') as f:
                f.write(str(time.time()))
            # Continue anyway in case the parent doesn't restart us
        
        # Initialize new model
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
                    # Get the next request if any
                    if not request_queue.empty():
                        request_data = request_queue.get()
                        prompt, token_queue, request_id, attempt = request_data
                        
                        # Process this request
                        await self._process_request(prompt, token_queue, request_id, attempt)
                        request_queue.task_done()
                    else:
                        # No requests, sleep briefly
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    # Handle cancellation cleanly
                    logger.info("Engine loop was cancelled")
                    break
                except Exception as e:
                    logger.error(f"Engine loop error: {e}")
                    model_health_status["healthy"] = False
                    model_health_status["last_error"] = str(e)
                    break
        
        try:
            self.loop.run_until_complete(process_queue())
        except Exception as e:
            logger.error(f"Engine thread crashed with error: {e}")
            model_health_status["healthy"] = False
            model_health_status["last_error"] = str(e)
        finally:
            if not self.loop.is_closed():
                self.loop.close()
            logger.info("Engine loop closed")

    async def _process_request(self, prompt, token_queue, request_id, attempt):
        """Process a single request and put tokens in the queue."""
        try:
            # Preprocess prompt
            prompt_string, initial_tokens = self.process_prompt(prompt)
            
            # Generate tokens with better error handling
            try:
                results_generator = self.model.generate(
                    prompt_string, 
                    self.sampling_params, 
                    request_id=request_id
                )
                
                previous_text = ""
                async for request_output in results_generator:
                    if not self.is_running:
                        # Stop if manager is shutting down
                        break
                        
                    text = request_output.outputs[0].text
                    new_text = text[len(previous_text):]
                    previous_text = text
                    
                    if new_text:
                        token_queue.put(new_text)
                
                # Request completed successfully
                token_queue.put(None)  # Signal completion
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"Generation error: {error_str}")
                
                # Check for specific CUDA errors
                is_cuda_error = any(err in error_str for err in [
                    "CUDA error", "cuDNN error", "device-side assert", 
                    "CUBLAS", "CUDNN_STATUS", "out of memory"
                ])
                
                if is_cuda_error:
                    model_health_status["healthy"] = False
                    model_health_status["last_error"] = error_str
                    token_queue.put("CUDA_ERROR")
                elif "Background loop has errored" in error_str:
                    model_health_status["healthy"] = False
                    model_health_status["last_error"] = error_str
                    token_queue.put("RETRY")
                else:
                    # More common error, just retry
                    token_queue.put("RETRY" if attempt < 3 else "ERROR")
                    
        except Exception as e:
            logger.error(f"Error processing request {request_id} (attempt {attempt}): {e}")
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
    restart_attempts = 0
    max_restarts = 10
    restart_window = 3600  # 1 hour window to count restarts
    last_restart_time = time.time()
    
    while True:
        try:
            # Check for restart flag from parent process
            if os.path.exists('/tmp/vllm_restart_needed'):
                try:
                    os.remove('/tmp/vllm_restart_needed')
                    logger.warning("External restart requested. Sending SIGTERM.")
                    # Send signal to parent process to initiate restart
                    os.kill(os.getppid(), signal.SIGTERM)
                    time.sleep(5)  # Wait a bit before continuing
                except Exception as e:
                    logger.error(f"Failed to process restart request: {e}")
            
            # Check model health
            if not model_health_status["healthy"]:
                current_time = time.time()
                
                # Reset restart counter if we're outside the window
                if current_time - last_restart_time > restart_window:
                    restart_attempts = 0
                    last_restart_time = current_time
                
                # Check if we're not restarting too frequently
                if restart_attempts < max_restarts:
                    logger.warning(f"Watchdog detected unhealthy model. Attempting restart ({restart_attempts+1}/{max_restarts})...")
                    if model_manager.restart_model():
                        logger.info("Model restarted successfully")
                    else:
                        logger.error("Failed to restart model")
                    
                    restart_attempts += 1
                else:
                    logger.critical(f"Too many restart attempts ({restart_attempts}) within time window. Requesting server restart...")
                    # Create a signal file for the process manager to restart us
                    with open('/tmp/vllm_restart_needed', 'w') as f:
                        f.write(str(time.time()))
                    time.sleep(60)  # Wait longer before trying again
                    restart_attempts = 0  # Reset counter after the wait
            
            # Check every few seconds
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
            cuda_error = False
            
            def raw_tokens():
                nonlocal tokens_processed, retry_requested, cuda_error
                
                while True:
                    try:
                        token = token_queue.get(timeout=30)  # 30s timeout for stuck queues
                    except queue.Empty:
                        logger.warning(f"Timeout waiting for tokens for request {request_id}")
                        retry_requested = True
                        break
                    
                    # Check for control signals
                    if token is None:
                        # Normal completion
                        tokens_processed = True
                        break
                    elif token == "RETRY":
                        # Retry requested
                        retry_requested = True
                        break
                    elif token == "CUDA_ERROR":
                        # CUDA error - needs special handling
                        cuda_error = True
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
                retry_requested = True
            
            # Check if we need to retry
            if tokens_processed or attempt >= max_attempts:
                # Either completed successfully or too many attempts
                break
            
            if retry_requested:
                # For CUDA errors, wait longer to allow more thorough cleanup
                wait_time = 5 if cuda_error else 2
                
                # Wait for the model to restart before retrying
                logger.info(f"Waiting {wait_time}s before retry {attempt+1}")
                time.sleep(wait_time)
                
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
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpu_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated(0)/1024**3:.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0)/1024**3:.2f} GB",
                "max_memory": f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB"
            }
        except Exception as e:
            gpu_info = {"error": str(e)}
    
    status = {
        "healthy": model_health_status["healthy"],
        "last_error": model_health_status["last_error"],
        "last_restart": model_health_status["last_restart"],
        "cuda_errors": model_health_status["cuda_errors"],
        "restart_count": model_health_status["restart_count"],
        "pending_requests": request_queue.qsize(),
        "gpu_info": gpu_info
    }
    return status

# --- Process manager script (separate file) ---
def write_process_manager_script():
    """Writes a process manager script that can be used to keep the server running."""
    script_content = """#!/bin/bash
# vllm_process_manager.sh
# Keep the vLLM server running and restart it when needed

MAX_RESTARTS=20
restart_count=0
restart_window_start=$(date +%s)

while true; do
    # Check if we need to reset the restart counter
    current_time=$(date +%s)
    elapsed=$((current_time - restart_window_start))
    
    if [ $elapsed -gt 3600 ]; then
        echo "Resetting restart counter"
        restart_count=0
        restart_window_start=$current_time
    fi
    
    # Start the server
    echo "Starting vLLM server..."
    python server.py &
    server_pid=$!
    
    # Wait for server to exit or for restart signal
    wait $server_pid
    exit_code=$?
    
    # Increment restart counter
    restart_count=$((restart_count + 1))
    
    if [ $restart_count -gt $MAX_RESTARTS ]; then
        echo "Too many restarts ($restart_count) in the last hour. Waiting 5 minutes before continuing."
        sleep 300
        restart_count=0
        restart_window_start=$(date +%s)
    else
        echo "Server exited with code $exit_code. Restarting in 5 seconds... (restart $restart_count/$MAX_RESTARTS)"
        sleep 5
    fi
done
"""
    
    with open("vllm_process_manager.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    try:
        os.chmod("vllm_process_manager.sh", 0o755)
        logger.info("Process manager script written to vllm_process_manager.sh")
        logger.info("Run with: bash vllm_process_manager.sh")
    except:
        logger.warning("Could not make process manager script executable")

# Write process manager script on startup
write_process_manager_script()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8080, threaded=True)