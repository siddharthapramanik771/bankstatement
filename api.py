
import json
from fastapi import FastAPI, UploadFile, File
import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import pytesseract
import torch
import json
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
# model_name = 'ChatDOC/OCRFlux-3B'
app = FastAPI()
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
streamer = TextStreamer(tokenizer, skip_prompt=True)

def clean_text_for_json(text: str) -> str:
    # Replace newlines and tabs with spaces
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\\n', ' ', text)
    # Remove other control characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)

    # Collapse multiple spaces into one
    text = re.sub(r'\s{2,}', ' ', text)
    # Remove extra spaces before or after colons and commas
    cleaned = re.sub(r'\s*:\s*', ': ', text)
    cleaned = re.sub(r'\s*,\s*', ', ', cleaned)

    # Remove extra space inside keys (e.g. " key" → "key")
    cleaned = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', cleaned)

    # Fix trailing commas (if any)
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    # Strip leading/trailing whitespace
    return cleaned.strip()

# Step 1: OCR function
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

# Step 2: Pass OCR text to LLM for restructuring
def restructure_text_with_model(ocr_text: str,account_info=False):
    prompt_account_info = f"""
You are an expert in analyzing bank statements. Given the OCR-extracted text from a bank statement image.
account number is important , don't just put any number ,search exactly for 'account number'
extract the following details accurately and return them in form of a JSON:

with the following keys
    "name": "...",
    "address": "...",
    "account number": "...",
    "customerId number":"...."
    "ifsc code": "...",
    "branch": "...",
    "statement_period": "...",  # from which date to which date
    "bank name": "..."


If any information is missing or unclear, return the corresponding value as None or an empty string.please no explanation

OCR Text:
\"\"\"
{ocr_text}
\"\"\"

JSON Output:
"""
    prompt_transactions = f"""You are an expert bank statement analyzer.Given the OCR-extracted text from a bank statement image.return data in form of list of dictionary ,in form of list of JSON objects. please no explanation

Extract all the transactions from the text in the form of a list of dictionaries with the following keys (type of the field is defined in the bracket):

- TXN_DATE (Type: Date, Format: YYYY-MM-DD)
- TXN_DESC (Type: String)
- CHEQUE_REF_NO (Type: String. Only include alphanumeric cheque or reference numbers. Do NOT include monetary amounts.)
- WITHDRAWAL_AMT (Type: Float. Only include if money is withdrawn.may be mentioned as debit in statement. Set as null if not applicable.)
- DEPOSIT_AMT (Type: Float. Only include if money is deposited.may be mentioned as credit in statement. Set as null if not applicable.)
- BALANCE_AMT (Type: Float)
OCR Text:
\"\"\"
{ocr_text}
\"\"\"
any Transaction amount(WITHDRAWAL_AMT/DEPOSIT_AMT) has to be Positive , If something is missing or unexpected , back calculate with math to match the BALANCE_AMT. return only JSON , nothing else, If you do not get any data just return ,but never return explanation or code snippet
JSON Output:
"""
    if account_info:
        prompt = prompt_account_info
        re_exp = r'\{[^{}]*\}'
    else:
        prompt = prompt_transactions
        re_exp = r'\[\s*\{.*?\}\s*\]'
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # streamer = streamer,
            max_new_tokens=4096,
            temperature=0.2,
            top_p=0.95,
            use_cache =True,
            do_sample=False
        )

    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result_text = result_text.replace(prompt,'')
    result_text = result_text.replace("\_","_")
    result_text =  clean_text_for_json(result_text)
    print(result_text)
    match = re.search(re_exp, result_text, re.DOTALL)

    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print("❌ JSON parsing error:", e)
    else:
        print("❌ No JSON array of objects found.")
    return result_text
    # Try extracting the JSON from model output
    # try:
    #     json_start = result_text.find("{")
    #     json_text = result_text[json_start:]
    #     structured_data = json.loads(json_text)
    # except Exception as e:
    #     structured_data = {"error": "Failed to parse JSON", "raw_output": result_text}
    # print("result_text",result_text)
    # return result_text.replace(prompt,'')


@app.post("/extract_data/")
async def generate(pdfs: list[UploadFile] = File(...)):
    # buffer = io.StringIO()
    # streamer = TextStreamer(tokenizer, skip_prompt=True)
    results=[]
    for pdf in pdfs:
        pdf_results = []
        pdf_bytes = await pdf.read()
        try:
            pages: list[Image.Image] = convert_from_bytes(pdf_bytes,dpi=300)
            print(len(pages))
        except Exception as e:
            print(f"\n[Failed to read {pdf.filename}: {str(e)}]\n")
            continue

        for idx, image in enumerate(pages):
            print(f"running for page number {idx+1}")
            print("🔍 Extracting text with OCR...")
            ocr_text = extract_text_from_image(image)
            # print("\n📝 OCR Text:\n", ocr_text)
            print("🤖 Restructuring text with LLM...")
            if idx == 0:
                r = restructure_text_with_model(ocr_text,account_info=True)
                pdf_results.append(r)
            structured_data = restructure_text_with_model(ocr_text)
            # print("\n📝 OCR Text:\n", ocr_text)
            # print("llm output",structured_data)
            pdf_results.append(structured_data)
        results.append(pdf_results)
    return {'result':results}

if __name__ =="__main__":
    import uvicorn
    uvicorn.run("api:app",host='0.0.0.0',port=8000,reload=False)
