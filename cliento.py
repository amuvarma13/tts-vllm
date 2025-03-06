import requests

def main():
    url = "https://es3tvk7pyofuwd-8080.proxy.runpod.net/events"
    response = requests.get(url, stream=True)
    
    # Iterate over the response lines
    for line in response.iter_lines():
        if line:  # filter out keep-alive new lines
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data:"):
                # Extract event data
                event_data = decoded_line[len("data:"):].strip()
                print("Event received:", event_data)

if __name__ == "__main__":
    main()
