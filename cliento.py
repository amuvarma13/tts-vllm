import requests

def main():
    # Define your prompt (URL-encoded if necessary)
    prompt = "Test prompt for generation"
    # Build the URL with the prompt query parameter
    url = f"https://es3tvk7pyofuwd-8080.proxy.runpod.net/events?prompt={prompt}"
    
    print(f"Connecting to {url}")
    with requests.get(url, stream=True) as response:
        # Iterate over each line in the streaming response
        for line in response.iter_lines():
            if line:  # skip keep-alive newlines
                decoded_line = line.decode("utf-8")
                # Check if the line contains an SSE data event
                if decoded_line.startswith("data:"):
                    event_data = decoded_line[len("data:"):].strip()
                    print("Event received:", event_data)

if __name__ == "__main__":
    main()
