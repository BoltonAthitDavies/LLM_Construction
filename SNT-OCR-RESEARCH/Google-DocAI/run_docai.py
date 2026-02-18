import os
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
from PIL import Image, ImageDraw
import fitz  # PyMuPDF, for converting PDF page to image for visualization

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# CONFIGURATION
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ill2-463109")
LOCATION = "us" # or 'eu'
PROCESSOR_ID = os.getenv("GCP_PROCESSOR_ID", "YOUR_PROCESSOR_ID_HERE") # <--- REPLACE THIS!
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google_credentials.json")

def run_docai(file_path):
    # Set credentials explictly if not in standard ENV
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    # Read the file
    with open(file_path, "rb") as image:
        image_content = image.read()

    # Determine MIME type
    mime_type = "application/pdf" if file_path.suffix == ".pdf" else "image/png"

    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    print(f"Sending request to Google DocAI (Processor: {PROCESSOR_ID})...")
    result = client.process_document(request=request)
    document = result.document

    print(f"Document processing complete. Detected {len(document.pages)} pages.")

    # Output Directory
    output_dir = Path("Google-DocAI/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Visualization Logic (The Red Boxes!) ---
    # We will process Page 1 for visualization
    page = document.pages[0]
    
    # Render PDF page to image for drawing (using PyMuPDF / fitz)
    viz_image = None
    if mime_type == "application/pdf":
        doc = fitz.open(file_path)
        page_pdf = doc.load_page(0)
        pix = page_pdf.get_pixmap()
        viz_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        viz_image = Image.open(file_path)

    draw = ImageDraw.Draw(viz_image)
    width, height = viz_image.size

    # Draw BBoxes for Paragraphs
    print("Drawing Bounding Boxes...")
    for para in page.paragraphs:
        # BBox format is normalized (0-1)
        vs = para.layout.bounding_poly.vertices
        # Convert valid vertices to denormalized coords
        # Google normalized vertices might behave oddly if some are missing, usually 4 points
        points = []
        for v in vs:
            points.append((v.x * width, v.y * height))
        
        # Draw Polygon
        if points:
            draw.polygon(points, outline="red", width=2)
            
    # Save Visualization
    vis_path = output_dir / "docai_debug_boxed.png"
    viz_image.save(vis_path)
    print(f"✅ Visualization with Red Boxes saved to: {vis_path}")

    # Save Text Output (RAG Ready)
    text_path = output_dir / "docai_text.md"
    with open(text_path, "w") as f:
        f.write(document.text)
    print(f"Full text saved to: {text_path}")

if __name__ == "__main__":
    # FILE = Path("Datasets/Screenshot 2569-02-04 at 10.25.32.png")
    # Using the PDF from before
    FILE = Path("Datasets/Method Precast อุโมงค์ระบายน้ำบึงหนองบอนลงสู่แม่น.pdf")
    
    if FILE.exists():
        try:
            run_docai(FILE)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Did you set GOOGLE_APPLICATION_CREDENTIALS and GCP_PROCESSOR_ID?")
    else:
        print(f"File not found: {FILE}")
