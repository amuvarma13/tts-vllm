import wave
import numpy as np

# Suppose 'complete_data' is the concatenated PCM data (as bytes).
# If your original data is float32, you need to convert it.
# Example: convert a NumPy array of floats in [-1, 1] to int16 PCM.
pcm_array = np.frombuffer(complete_data, dtype=np.float32)
int_pcm = (pcm_array * 32767).astype(np.int16)
audio_bytes = int_pcm.tobytes()

with wave.open("output.wav", "wb") as wav_file:
    wav_file.setnchannels(1)      # mono audio
    wav_file.setsampwidth(2)        # 2 bytes for int16
    wav_file.setframerate(24000)    # sample rate of 24000 Hz
    wav_file.writeframes(audio_bytes)
