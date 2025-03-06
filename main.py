from flask import Flask, Response
import time

app = Flask(__name__)

def event_stream():
    # Immediately send an event upon connection
    yield "data: Connected to the server\n\n"
    # Wait for 5 seconds
    time.sleep(5)
    # Send another event after 5 seconds
    yield "data: 5 seconds later event\n\n"

@app.route('/events')
def sse():
    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
