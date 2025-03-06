import requests

def main():
    # Replace 'Hello+world' with your desired prompt
    url = "http://localhost:8080/events?prompt=Hello+world"
    with requests.get(url, stream=True) as response:
        # Iterate over each line in the streaming response
        for line in response.iter_lines():
            if line:  # skip keep-alive newlines
                decoded_line = line.decode("utf-8")
                # SSE messages typically start with "data:"
                if decoded_line.startswith("data:"):
                    # Extract and print the event data (strip the "data:" prefix)
                    event_data = decoded_line[len("data:"):].strip()
                    print("Event received:", event_data)

if __name__ == '__main__':
    main()
