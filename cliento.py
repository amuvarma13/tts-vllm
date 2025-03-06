import requests
import base64

def main():
    prompt = "Test prompt for generation"
    url = f"https://es3tvk7pyofuwd-8080.proxy.runpod.net/events?prompt={prompt}"
    print(f"Connecting to {url}")

    # Open the file for writing in binary mode.
    with open("output.wav", "wb") as wav_file:
        with requests.get(url, stream=True) as response:
            for line in response.iter_lines():
                if line:  # Skip keep-alive newlines
                    decoded_line = line.decode("utf-8")
                    # SSE events usually have lines starting with "data:"
                    if decoded_line.startswith("data:"):
                        event_data = decoded_line[len("data:"):].strip()
                        # Assume the server sends the audio bytes base64 encoded.
                        try:
                            audio_bytes = base64.b64decode(event_data)
                        except Exception as e:
                            print("Error decoding base64:", e)
                            continue
                        wav_file.write(audio_bytes)
                        # Optionally, flush to ensure data is written immediately.
                        wav_file.flush()
                        print("Wrote", len(audio_bytes), "bytes")
                        
    print("Streaming complete.")

if __name__ == "__main__":
    main()
