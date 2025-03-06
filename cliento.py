import requests
import base64
import io
import torch

def main():
    prompt = "Test prompt for generation"
    url = f"https://es3tvk7pyofuwd-8080.proxy.runpod.net/events?prompt={prompt}"
    print(f"Connecting to {url}")

    tensor_list = []
    
    with requests.get(url, stream=True) as response:
        for line in response.iter_lines():
            if line:  # Skip keep-alive newlines
                decoded_line = line.decode("utf-8")
                # SSE events usually have lines starting with "data:"
                if decoded_line.startswith("data:"):
                    event_data = decoded_line[len("data:"):].strip()
                    try:
                        serialized_tensor = base64.b64decode(event_data)
                    except Exception as e:
                        print("Error decoding base64:", e)
                        continue
                    
                    # Reconstruct the tensor using torch.load from an in-memory bytes buffer
                    try:
                        buffer = io.BytesIO(serialized_tensor)
                        tensor = torch.load(buffer)
                        tensor_list.append(tensor)
                        print("Added tensor with shape:", tensor.shape)
                    except Exception as e:
                        print("Error loading tensor:", e)
                        
    print("Streaming complete.")
    print(f"Collected {len(tensor_list)} tensors.")
    return tensor_list

if __name__ == "__main__":
    tensors = main()
