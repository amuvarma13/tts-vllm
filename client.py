import requests
from sseclient import SSEClient

# URL of your SSE endpoint
url = "https://es3tvk7pyofuwd-8080.proxy.runpod.net/events"

def main():
    # Initiate a streaming request to the SSE endpoint
    response = requests.get(url, stream=True)
    
    # Wrap the response in an SSEClient instance to parse the events
    client = SSEClient(response)
    
    # Loop over the events as they arrive and print each event's data
    for event in client.events():
        print("Event received:", event.data)

if __name__ == '__main__':
    main()
