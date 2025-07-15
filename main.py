from unsloth import FastVisionModel
import io
import contextlib
import time
import threading
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import io
import contextlib
import asyncio
import os
import torch
# from datasets import load_dataset
from transformers import TextStreamer
# from unsloth import is_bf16_supported
# from unsloth.trainer import UnslothVisionDataCollator
# from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
from PIL import Image
from pdf2image import convert_from_bytes
# from google.colab import userdata
hf_token = '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'

login(token=hf_token)
model_name = 'unsloth/Llama-3.2-11B-Vision-Instruct'
app = FastAPI()
model, tokenizer = FastVisionModel.from_pretrained(
    model_name,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules      = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
instruction_account_info = """
You are an expert in analyzing bank statements. Given the OCR-extracted text from a bank statement image, extract the following details accurately and return them in a structured Python dictionary format:

{
    "name": "...",
    "address": "...",
    "account number": "...",
    "ifsc code": "...",
    "branch": "...",
    "statement_period": "...",  # from which date to which date
    "bank name": "..."
}

If any information is missing or unclear, return the corresponding value as None or an empty string.please no explanation 
"""
instruction_transactions = """You are an expert bank statement analyzer.return data in form of list of dictionary . please no explanation 

Extract all the transactions from the image in the form of a list of dictionaries with the following keys (type of the field is defined in the bracket):  

- TXN_DATE (Type: Date, Format: YYYY-MM-DD)
- TXN_DESC (Type: String)
- CHEQUE_REF_NO (Type: String. Only include alphanumeric cheque or reference numbers. Do NOT include monetary amounts.)
- WITHDRAWAL_AMT (Type: Float. Only include if money is withdrawn.may be mentioned as debit in statement. Set as null if not applicable.)
- DEPOSIT_AMT (Type: Float. Only include if money is deposited.may be mentioned as credit in statement. Set as null if not applicable.)
- BALANCE_AMT (Type: Float)

"""
FastVisionModel.for_inference(model)

@app.post("/generate/")
async def generate(pdfs: list[UploadFile] = File(...)):
    async def stream_response():
        buffer = io.StringIO()
        streamer = TextStreamer(tokenizer, skip_prompt=True)

        for pdf in pdfs:
            pdf_bytes = await pdf.read()
            try:
                pages: list[Image.Image] = convert_from_bytes(pdf_bytes)
            except Exception as e:
                yield f"\n[Failed to read {pdf.filename}: {str(e)}]\n"
                continue

            for idx, image in enumerate(pages):
                # Preprocess for your model (change if using processor)
                inputs = []
                # instruction_account_info,
                if idx == 0:
                    instructs = [instruction_account_info, instruction_transactions]
                else: 
                    instructs = [ instruction_transactions]
                    
                for inst in instructs:
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": inst}
                        ]}
                    ]
                    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                    inputs.append(tokenizer(
                        image,
                        input_text,
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).to("cuda"))
                    for input in inputs:
                        _ = model.generate(
                            **input,
                            streamer= streamer,
                            max_new_tokens= 4096,
                            use_cache = True,
                            temperature= 0.9,
                            min_p = 0.2)
                # inputs = tokenizer(images=image, return_tensors="pt").to(model.device)
                with contextlib.redirect_stdout(buffer):
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
                            text = buffer.getvalue()
                            if len(text) > last:
                                print(text[last:], end="", flush=True)
                                last = len(text)
                            time.sleep(0.1)
                            if threading.active_count() == 1:
                                break

                # Run model and redirect stdout from TextStreamer
                # def run():
                #     with contextlib.redirect_stdout(buffer):
                #         _ = model.generate(
                #             **inputs,
                #             streamer=streamer,
                #             max_new_tokens=4096,
                #             use_cache=True,
                #             temperature=0.9,
                #             min_p=0.2,
                #         )

                # task = asyncio.to_thread(run)

                # last = 0
                # yield f"\n📄 {pdf.filename} - Page {idx + 1}:\n"
                # while not task.done():
                #     await asyncio.sleep(0.1)
                #     out = buffer.getvalue()
                #     if len(out) > last:
                #         yield out[last:]
                #         last = len(out)

                # # Flush final output
                # out = buffer.getvalue()
                # if len(out) > last:
                #     yield out[last:]

                # yield f"\n[Done: {pdf.filename} - Page {idx + 1}]\n"

    return {'output':'processing'}

if __name__ =="__main__":
    import uvicorn
    uvicorn.run("main:app",host='0.0.0.0',port=8000,reload=False)
