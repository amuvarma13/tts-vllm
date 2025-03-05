import asyncio
import time
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

engine_args = AsyncEngineArgs(model="facebook/opt-125m", enforce_eager=True)
model = AsyncLLMEngine.from_engine_args(engine_args)

async def generate_streaming(prompt):
    results_generator = model.generate(prompt, SamplingParams(), request_id=time.monotonic())
    previous_text = ""
    async for request_output in results_generator:
        text = request_output.outputs[0].text
        print(text[len(previous_text):])
        previous_text = text

asyncio.run(generate_streaming("hello"))