from flask import Flask, Response, request
import wave
from io import BytesIO
from orpheus_tts import OrpheusModel

app = Flask(__name__)
engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

@app.route('/tts', methods=['GET'])
def tts():
    prompt = request.args.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')
    default_voice = "tara"
    syn_tokens = engine.generate_speech(
        prompt=prompt,
        voice=default_voice,
        repetition_penalty=1.1,
        stop_token_ids=[128258],
        max_tokens=2000,
        temperature=0.3,
        top_p=1
    )
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        for chunk in syn_tokens:
            wf.writeframes(chunk)
    buffer.seek(0)
    return Response(buffer, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
