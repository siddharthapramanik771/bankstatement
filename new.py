import io
import contextlib
import time

stream_buffer = io.StringIO()
streamer = TextStreamer(tokenizer, skip_prompt=True)

with contextlib.redirect_stdout(stream_buffer):
    threading.Thread(target=model.generate, kwargs={
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 4096,
        "use_cache": True,
        "temperature": 0.9,
        "min_p": 0.2
    }).start()

    last = 0
    while True:
        text = stream_buffer.getvalue()
        if len(text) > last:
            print(text[last:], end="", flush=True)
            last = len(text)
        time.sleep(0.1)
        if threading.active_count() == 1:
            break
