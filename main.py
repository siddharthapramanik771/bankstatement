from unsloth import FastVisionModel
import io
import contextlib
import time
import threading
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import re
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
hf_token = ''

login(token=hf_token)
model_name = 'ChatDOC/OCRFlux-3B'
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
You are an expert in analyzing bank statements. Given the OCR-extracted text from a bank statement image.
account number is important , don't just put any number ,search exactly for 'account number'
extract the following details accurately and return them in a structured Python dictionary format. in form of a JSON:

{
    "name": "...",
    "address": "...",
    "account number": "...",
    "customerId number":"...."
    "ifsc code": "...",
    "branch": "...",
    "statement_period": "...",  # from which date to which date
    "bank name": "..."
}

If any information is missing or unclear, return the corresponding value as None or an empty string.please no explanation
"""
instruction_transactions = """You are an expert bank statement analyzer.return data in form of list of dictionary ,in form of list of JSON objects. please no explanation

Extract all the transactions from the image in the form of a list of dictionaries with the following keys (type of the field is defined in the bracket):

- TXN_DATE (Type: Date, Format: YYYY-MM-DD)
- TXN_DESC (Type: String)
- CHEQUE_REF_NO (Type: String. Only include alphanumeric cheque or reference numbers. Do NOT include monetary amounts.)
- WITHDRAWAL_AMT (Type: Float. Only include if money is withdrawn.may be mentioned as debit in statement. Set as null if not applicable.)
- DEPOSIT_AMT (Type: Float. Only include if money is deposited.may be mentioned as credit in statement. Set as null if not applicable.)
- BALANCE_AMT (Type: Float)

"""

streamer = TextStreamer(tokenizer, skip_prompt=True)
FastVisionModel.for_inference(model)
def run_llm(inputs):
    # model = get_model()
    print(f'starting new page,page number{inputs[0][0]+1}')
    # print(inputs[0][1])
    # print('prompt ends here.....')
    output_ids = model.generate(
        **inputs[1],
        streamer=streamer,
        max_new_tokens=4096,
        use_cache=True,
        do_sample=False
    )
    # del model  # or any other large CUDA object
    # gc.collect()
    
    # Then clear the cache
    torch.cuda.empty_cache()
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    match = re.search(inputs[0][2], output_text.replace(inputs[0][1],''), re.DOTALL)

    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            print(data)
            return data
        except json.JSONDecodeError as e:
            print("❌ JSON parsing error:", e)
    else:
        print("❌ No JSON array of objects found.")
    print(output_text)
    return output_text

@app.post("/extract_data/")
async def generate(pdfs: list[UploadFile] = File(...)):
    # buffer = io.StringIO()
    # streamer = TextStreamer(tokenizer, skip_prompt=True)
    results=[]
    for pdf in pdfs:
        inputs = []
        pdf_bytes = await pdf.read()
        try:
            pages: list[Image.Image] = convert_from_bytes(pdf_bytes,dpi=300)
            print(len(pages))
        except Exception as e:
            print(f"\n[Failed to read {pdf.filename}: {str(e)}]\n")
            continue

        for idx, image in enumerate(pages):
            # Preprocess for your model (change if using processor)
            # instruction_account_info,
            account_info = [instruction_account_info,r'\{[^{}]*\}']
            transactions = [instruction_transactions,r'\[\s*\{.*?\}\s*\]']
            if idx == 0:
                instructs = [account_info, transactions]
            else:
                instructs = [ transactions]
                
            for inst in instructs:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": inst[0]}
                    ]}
                ]
                input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                params = [idx,inst[0],inst[1]]
                inputs.append([params,tokenizer(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to("cuda")])
        pdf_results=[]
        for input in inputs:
            pdf_results.append(run_llm(input))    
        results.append(pdf_results) 

    return {'result':results}

if __name__ =="__main__":
    import uvicorn
    uvicorn.run("main:app",host='0.0.0.0',port=8000,reload=False)
