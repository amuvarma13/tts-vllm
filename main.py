from flask import Flask, Response, request
import struct
from orpheus_tts import OrpheusModel

app = Flask(__name__)
engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """
    Create a minimal WAV header. 
    We can't know the final data size in advance for streaming, 
    so we'll temporarily set it to an arbitrary 0 in the header. 
    Some audio players will accept a streaming WAV with a 0 size, 
    but if your client needs the exact data size, you have to buffer or 
    do some more sophisticated workaround.
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    # 36 + SubChunk2Size (which we set to 0 for streaming)
    data_size = 0

    # struct.pack with little-endian ('<'):
    #   - RIFF chunk descriptor
    #   - 36 + SubChunk2Size (4 bytes)
    #   - 'WAVE'
    #   - 'fmt ' subchunk
    #   - Subchunk1Size (16 for PCM)
    #   - AudioFormat (1 for PCM)
    #   - NumChannels
    #   - SampleRate
    #   - ByteRate
    #   - BlockAlign
    #   - BitsPerSample
    #   - 'data'
    #   - SubChunk2Size again (0 for streaming)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,       # 4 bytes
        b'WAVE',
        b'fmt ',
        16,                   # SubChunk1Size
        1,                    # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

@app.route('/tts', methods=['GET'])
def tts():
    prompt = request.args.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')

    def generate_audio_stream():
        # First, yield a minimal WAV header
        yield create_wav_header()

        # Now, stream out the audio chunks as they are produced
        syn_tokens = engine.generate_speech(
            prompt=prompt,
            voice="tara",
            repetition_penalty=1.1,
            stop_token_ids=[128258],
            max_tokens=2000,
            temperature=0.6,
            top_p=0.9
        )
        for chunk in syn_tokens:
            # Each chunk is already 16-bit PCM frames
            yield chunk

    return Response(generate_audio_stream(), mimetype='audio/wav')

if __name__ == '__main__':
    # Enable threaded=True (the default) to allow concurrent requests
    app.run(host='0.0.0.0', port=8080, threaded=True)
