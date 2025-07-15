from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import io
import contextlib
import asyncio

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with your model
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.to("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/generate/")
async def generate(files: list[UploadFile] = File(...)):
    async def stream_response():
        # Read and combine all file contents
        text = ""
        for file in files:
            content = await file.read()
            text += content.decode(errors="ignore") + "\n"

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Redirect stdout to capture stream output
        buffer = io.StringIO()
        streamer = TextStreamer(tokenizer, skip_prompt=True)

        def run():
            with contextlib.redirect_stdout(buffer):
                _ = model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=4096,
                    use_cache=True,
                    temperature=0.9,
                    min_p=0.2,
                )

        task = asyncio.to_thread(run)

        last = 0
        while not task.done():
            await asyncio.sleep(0.1)
            output = buffer.getvalue()
            if len(output) > last:
                yield output[last:]
                last = len(output)

        # Flush remaining
        output = buffer.getvalue()
        if len(output) > last:
            yield output[last:]

        yield "\n[Done]\n"

    return StreamingResponse(stream_response(), media_type="text/plain")
