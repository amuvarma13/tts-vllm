import requests

def main():
    # Replace 'Hello+world' with your desired prompt
    prompt = "Test prompt for generation"
    url = f"https://es3tvk7pyofuwd-8080.proxy.runpod.net/events?prompt={prompt}"
    with requests.get(url, stream=True) as response:
        # Iterate over each line in the streaming response
        for line in response.iter_lines():
            if line:  # skip keep-alive newlines
                # decoded_line = line.decode("utf-8")
                print(line)

    

if __name__ == '__main__':
    main()
