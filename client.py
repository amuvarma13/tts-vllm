import requests
import numpy as np
import soundfile as sf
import json

def main():
    prompt = "Test prompt for generation"
    url = f"https://es3tvk7pyofuwd-8080.proxy.runpod.net/events?prompt={prompt}"
    
    audio_chunks = []
    text_messages = []
    
    with requests.get(url, stream=True) as response:
        for line in response.iter_lines():
            if not line:
                continue
                
            # Check if it's an SSE line (typically starts with "data: ")
            try:
                decoded_line = line.decode('utf-8', errors='ignore')
                
                if decoded_line.startswith('data: '):
                    # Extract the data part
                    data_content = decoded_line[6:].strip()
                    
                    try:
                        # Try to parse as JSON
                        json_data = json.loads(data_content)
                        text_messages.append(json_data)
                        print(f"Received JSON message: {json_data}")
                    except json.JSONDecodeError:
                        # If it's not JSON, it might be a different format
                        print(f"Received non-JSON text: {data_content[:50]}...")
                        text_messages.append(data_content)
                else:
                    # This could be binary data
                    print(f"Received potential binary data of length: {len(line)}")
                    audio_chunks.append(line)
            except UnicodeDecodeError:
                # This is likely binary data (audio)
                print(f"Received binary chunk of length: {len(line)}")
                audio_chunks.append(line)
    
    # Process audio chunks if any were received
    if audio_chunks:
        # First, try to determine the correct element size
        # Common audio formats are float32 (4 bytes) or int16 (2 bytes)
        for element_size in [4, 2, 8, 1]:  # Try different element sizes
            try:
                # Join all binary chunks
                all_audio = b''.join(audio_chunks)
                
                # Make sure the buffer size is a multiple of element size
                truncated_size = (len(all_audio) // element_size) * element_size
                truncated_audio = all_audio[:truncated_size]
                
                if element_size == 4:
                    audio_array = np.frombuffer(truncated_audio, dtype=np.float32)
                elif element_size == 2:
                    audio_array = np.frombuffer(truncated_audio, dtype=np.int16)
                elif element_size == 8:
                    audio_array = np.frombuffer(truncated_audio, dtype=np.float64)
                else:
                    audio_array = np.frombuffer(truncated_audio, dtype=np.int8)
                
                print(f"Successfully parsed audio with element size {element_size}")
                print(f"Audio array shape: {audio_array.shape}")
                
                # Save as WAV
                output_file = f"output_audio_elemsize_{element_size}.wav"
                sample_rate = 22050  # Adjust as needed
                sf.write(output_file, audio_array, sample_rate)
                print(f"Saved audio to {output_file}")
                
                # Only try one successful format
                break
                
            except Exception as e:
                print(f"Failed with element size {element_size}: {str(e)}")
    
    # Print summary of received data
    print(f"\nReceived {len(text_messages)} text/JSON messages")
    print(f"Received {len(audio_chunks)} audio chunks")

if __name__ == '__main__':
    main()