from flask import Flask, Response, request
import time
import threading

app = Flask(__name__)

# Global state for prompt ordering
prompt_queue = []
next_prompt_id = 1
queue_lock = threading.Lock()

def event_stream(prompt_id, prompt):
    # Immediately send an event with the queue position
    with queue_lock:
        position = prompt_queue.index(prompt_id) + 1
    yield f"data: Connected. Your prompt ID is {prompt_id}. Queue position: {position}\n\n"
    
    # Simulate some processing delay
    time.sleep(2)
    
    # Send the received prompt back to the client
    yield f"data: Received prompt: {prompt}\n\n"
    
    # Simulate additional processing (if needed)
    time.sleep(2)
    
    # Once processing is complete, remove the prompt from the queue
    with queue_lock:
        if prompt_id in prompt_queue:
            prompt_queue.remove(prompt_id)
    yield "data: Processing complete. Goodbye.\n\n"

@app.route('/events', methods=['GET'])
def sse():
    global next_prompt_id
    # Get the prompt from the query parameter; use a default if none is provided.
    prompt = request.args.get('prompt', 'No prompt provided')
    
    # Assign a unique prompt ID and add it to the queue
    with queue_lock:
        prompt_id = next_prompt_id
        next_prompt_id += 1
        prompt_queue.append(prompt_id)
    
    return Response(event_stream(prompt_id, prompt), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
