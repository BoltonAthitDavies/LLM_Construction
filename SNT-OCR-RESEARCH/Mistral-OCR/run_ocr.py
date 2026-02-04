import base64
import os
from mistralai import Mistral
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

def encode_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')

pdf_path = "/Users/worakanlasudee/Documents/GitHub/SNT-OCR-RESEARCH/Datasets/Method Precast For Repairing Tunnet Segment (อุโมงค์ระบายน้ำบึงหนองบอ-1-4.pdf"
base64_pdf = encode_pdf(pdf_path)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}" 
    },
    table_format="html", # default is None
    # extract_header=True, # default is False
    # extract_footer=True, # default is False
    include_image_base64=True
)

# Output handling
import json

output_dir = Path("Mistral-OCR/output")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Save Raw Response
response_path = output_dir / "simple_ocr_response.json"
with open(response_path, "w", encoding='utf-8') as f:
    # Serialize the OCRResponse object
    json.dump(json.loads(ocr_response.model_dump_json()), f, indent=2, ensure_ascii=False)
print(f"OCR Response saved to: {response_path}")

# 2. Save Markdown
markdown_path = output_dir / "simple_ocr_output.md"
full_md = ""
for page in ocr_response.pages:
    full_md += f"<!-- Page {page.index + 1} -->\n{page.markdown}\n\n"

with open(markdown_path, "w") as f:
    f.write(full_md)
print(f"Markdown saved to: {markdown_path}")