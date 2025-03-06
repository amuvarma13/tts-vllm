import requests
import json

def main():
    # Replace 'Hello+world' with your desired prompt
    prompt = "Test prompt for generation"
    url = f"https://es3tvk7pyofuwd-8080.proxy.runpod.net/events?prompt={prompt}"
    
    with requests.get(url, stream=True) as response:
        # Iterate over each line in the streaming response
        for line in response.iter_lines():
            if line:  # skip keep-alive newlines
                # Decode the line from bytes to string
                decoded_line = line.decode('utf-8')
                
                # Try to parse it as JSON if it appears to be JSON data
                try:
                    # Remove "data: " prefix if it exists (common in SSE)
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[6:]
                    else:
                        json_str = decoded_line
                        
                    # Parse the JSON
                    json_data = json.loads(json_str)
                    print(f"Decoded JSON: {json.dumps(json_data, indent=2)[:100]}...")
                except json.JSONDecodeError:
                    # If it's not valid JSON, just print the decoded string
                    print(f"Decoded text: {decoded_line[:100]}...")

if __name__ == '__main__':
    main()