import requests
import json
def is_tensor(data_str):
    # Check if it matches tensor format (starts with "tensor([")
    if data_str.strip().startswith("tensor("):
        return True
    
    # Try to parse as JSON to see if it's a nested array structure
    try:
        data = json.loads(data_str)
        # Check if it's a list of lists (nested array structure typical of tensors)
        if isinstance(data, list) and all(isinstance(item, list) for item in data):
            return True
    except (json.JSONDecodeError, TypeError):
        pass
    
    return False
def main():
    # Define your prompt (URL-encoded if necessary)
    prompt = "Test prompt for generation"
    # Build the URL with the prompt query parameter
    url = f"https://es3tvk7pyofuwd-8080.proxy.runpod.net/events?prompt={prompt}"
    collect_tensors = []
    print(f"Connecting to {url}")
    with requests.get(url, stream=True) as response:
        # Iterate over each line in the streaming response
        for line in response.iter_lines():
            if line:  # skip keep-alive newlines
                decoded_line = line.decode("utf-8")
                # Check if the line contains an SSE data event
                if decoded_line.startswith("data:"):
                    event_data = decoded_line[len("data:"):].strip()
                    #check if data is tensor
                    if is_tensor(event_data):
                        collect_tensors.append(event_data)
                        print("Tensor data received:", event_data)
                    else:
                        print("Non-tensor event:", event_data)

    print("Collected tensors:", collect_tensors[0].shape)
if __name__ == "__main__":
    main()
