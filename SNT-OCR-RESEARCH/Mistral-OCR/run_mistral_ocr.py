import os
from mistralai import Mistral
from pathlib import Path
from dotenv import load_dotenv
import json
import base64

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def run_mistral_ocr(doc_path):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in .env file")

    client = Mistral(api_key=api_key)
    
    print(f"Uploading file: {doc_path}...")
    
    file_path = Path(doc_path)
    
    # Upload the file to Mistral's storage
    with open(file_path, "rb") as f:
        uploaded_file = client.files.upload(
            file={
                "file_name": file_path.name,
                "content": f,
            },
            purpose="ocr"
        )
    
    print(f"File uploaded. ID: {uploaded_file.id}")
    
    # Get a signed URL for the uploaded file
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
    
    print("Sending OCR request...")
    
    # Process the OCR request
    print("Sending OCR request (HTML tables)...")
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
        include_image_base64=True,
        table_format="html",
        pages=list(range(31)) # Pages 0 to 30 (Total 31 pages)
    )
    
    print("OCR completed.")
    
    # DEBUG: Check what was found
    total_images_found = sum(len(getattr(p, 'images', []) or []) for p in ocr_response.pages)
    print(f"DEBUG: Total Images Detected across all pages: {total_images_found}")
    
    if ocr_response.pages:
        # Define output directory
        output_dir = Path("Mistral-OCR/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Load original image for visualization (if possible)
        try:
            from PIL import Image, ImageDraw
            
            # Ensure path object
            img_path_obj = Path(doc_path)
            
            # Only attempt to open if it's an image format Pillow supports natively
            if img_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                original_py_image = Image.open(img_path_obj)
                draw = ImageDraw.Draw(original_py_image)
                has_drawings = False
            else:
                # PDF or other formats - skip visualization overlay
                original_py_image = None
                print(f"Input is {img_path_obj.suffix}, skipping visual overlay debug.")

        except ImportError:
            original_py_image = None
            print("Pillow not installed, skipping visualization.")
        except Exception as e:
            original_py_image = None
            print(f"Could not load original image for visualization: {e}")

        # Helper to parse Markdown into Structured Elements (2026 Schema)
        def parse_markdown_to_elements(markdown_text, page_num, images_map):
            elements = []
            if not markdown_text:
                return elements
            lines = markdown_text.split('\n')
            current_block = {"type": "text", "content": []}
            
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Detect Headings
                if line.startswith('#'):
                    # Save previous block if exists
                    if current_block["content"]:
                        elements.append({
                            "id": f"e_{page_num}_{len(elements)+1}",
                            "type": "text",
                            "text_content": "\n".join(current_block["content"]),
                            "bbox": None
                        })
                        current_block = {"type": "text", "content": []}
                    
                    level = len(line.split(' ')[0])
                    text = line.lstrip('#').strip()
                    elements.append({
                        "id": f"e_{page_num}_{len(elements)+1}",
                        "type": "heading",
                        "level": level,
                        "text_content": text,
                        "bbox": None
                    })
                
                # Detect Images (standard markdown or Mistral specific)
                elif line.startswith('![') and '](' in line:
                     # Save previous block
                    if current_block["content"]:
                        elements.append({
                            "id": f"e_{page_num}_{len(elements)+1}",
                            "type": "text",
                            "text_content": "\n".join(current_block["content"]),
                            "bbox": None
                        })
                        current_block = {"type": "text", "content": []}

                    # Extract Image ID/Path
                    # Format: ![alt](path)
                    try:
                        img_part = line.split('](')[1]
                        img_path = img_part.rstrip(')')
                        # Map back to our image object if possible
                        elements.append({
                            "id": f"e_{page_num}_{len(elements)+1}",
                            "type": "image",
                            "description": line.split('![')[1].split(']')[0],
                            "doc_path": img_path,
                            "bbox": None # Mistral didn't give us this, sadly
                        })
                    except:
                        current_block["content"].append(line)

                else:
                    current_block["content"].append(line)
            
            # Flush absolute last block
            if current_block["content"]:
                elements.append({
                    "id": f"e_{page_num}_{len(elements)+1}",
                    "type": "text",
                    "text_content": "\n".join(current_block["content"]),
                    "bbox": None
                })
            
            return elements

        # 1. RAG Preparation: 2026 Strict Schema
        rag_manifest = {
            "doc_id": f"doc_{file_path.stem}",
            "pages": []
        }
        
        full_markdown = ""
        
        for i, page in enumerate(ocr_response.pages):
            page_num = i + 1
            
            # DEBUG: See what hidden data exists on the page object
            if i == 0:
                print(f"--- DEBUG: Page {page_num} Attributes ---")
                print([d for d in dir(page) if not d.startswith('_')])
                print("---------------------------------------")

            # Initialize Page Object early
            width, height = original_py_image.size if original_py_image else (None, None)
            page_entry = {
                "page_num": page_num,
                "size": {"width": width, "height": height},
                "elements": [],
                "images": []
            }

            # 1. Process Images first to get their paths
            page_images_map = {}
            extracted_images = getattr(page, 'images', [])
            
            if extracted_images:
                for img_obj in extracted_images:
                    img_id = getattr(img_obj, 'id', 'unknown')
                    # ... image saving logic ...
                    img_b64 = getattr(img_obj, 'image_base64', None)
                    bbox = getattr(img_obj, 'bbox', None)
                    
                    if img_b64:
                        # Safety: Strip data URI prefix if present
                        if "," in img_b64:
                            img_b64 = img_b64.split(",", 1)[1]
                        
                        # Determine extension from ID or default
                        ext = "png" # Default to png
                        if img_id.lower().endswith(".jpg") or img_id.lower().endswith(".jpeg"):
                            ext = "jpg"
                        elif img_id.lower().endswith(".png"):
                            ext = "png"
                            
                        # Clean ID for filename (remove extension from ID if present)
                        clean_id = img_id.rsplit('.', 1)[0] if '.' in img_id else img_id
                        img_filename = f"p{page_num}_{clean_id}.{ext}"
                        img_save_path = images_dir / img_filename
                        
                        try:
                            with open(img_save_path, "wb") as img_file:
                                img_file.write(base64.b64decode(img_b64))
                            
                            page_entry["images"].append({
                                "id": img_id,
                                "file_path": str(img_save_path.relative_to(output_dir)),
                                "format": ext,
                            })
                            
                            # Fix Markdown link
                            if page.markdown:
                                # We replace the specific reference
                                page.markdown = page.markdown.replace(f"({img_id})", f"(images/{img_filename})")
                            
                            # --- Visualization Logic ---
                            if original_py_image and bbox:
                                w, h = original_py_image.size
                                if isinstance(bbox, list) and len(bbox) == 4:
                                    if all(isinstance(c, float) and c <= 1.0 for c in bbox):
                                         x0, y0, x1, y1 = bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h
                                    else:
                                         x0, y0, x1, y1 = bbox
                                    
                                    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                                    has_drawings = True

                        except Exception as e:
                            print(f"Failed to save image {img_id}: {e}")

            # 1.5 Process Tables (New!)
            tables_dir = output_dir / "tables"
            tables_dir.mkdir(exist_ok=True)
            
            extracted_tables = getattr(page, 'tables', [])
            if extracted_tables:
                for tbl_obj in extracted_tables:
                    tbl_id = getattr(tbl_obj, 'id', 'unknown')
                    tbl_html = getattr(tbl_obj, 'html_body', None) 
                    if not tbl_html:
                        tbl_html = getattr(tbl_obj, 'body', None)
                    if not tbl_html:
                        tbl_html = getattr(tbl_obj, 'content', None)

                    if tbl_html:
                        tbl_filename = f"{tbl_id}.html"
                        tbl_save_path = tables_dir / tbl_filename
                        
                        try:
                            # Wrap in basic HTML structure for viewing
                            full_html = f"<html><body>{tbl_html}</body></html>"
                            
                            with open(tbl_save_path, "w", encoding='utf-8') as tbl_file:
                                tbl_file.write(full_html)
                            
                            # Add to manifest elements ? (Optional, currently handled in markdown parse)
                            # But we should fix the markdown link
                            if page.markdown:
                                # Replace [link](link) with local path
                                page.markdown = page.markdown.replace(f"({tbl_filename})", f"(tables/{tbl_filename})")
                                page.markdown = page.markdown.replace(f"({tbl_id})", f"(tables/{tbl_filename})")

                        except Exception as e:
                            print(f"Failed to save table {tbl_id}: {e}")

            # 2. Parse Markdown into Strict Elements
            elements = parse_markdown_to_elements(page.markdown, page_num, page_images_map)
            page_entry["elements"] = elements

            # 3. Add to Manifest
            rag_manifest["pages"].append(page_entry)
            
            full_markdown += f"\n\n<!-- Page {page_num} -->\n{page.markdown}"

        # Save visualization if drawing occurred
        if original_py_image and has_drawings:
            vis_path = output_dir / "visual_debug_boxed.png"
            original_py_image.save(vis_path)
            print(f"Visualization with Bounding Boxes saved to: {vis_path}")
        elif original_py_image:
            print(f"No Bounding Boxes returned by API. 'visual_debug_boxed.png' was NOT created.")
            # Optional: Save plain image just to prove paths work?
            # original_py_image.save(output_dir / "visual_debug_plain.png")

        # 2. Save RAG Manifest (Strict Schema)
        manifest_path = output_dir / "rag_manifest_v2.json"
        with open(manifest_path, "w", encoding='utf-8') as f:
            json.dump(rag_manifest, f, indent=2, ensure_ascii=False)
        print(f"RAG Manifest (Strict Schema) saved to: {manifest_path}")

        # 3. Save Clean Markdown
        md_output_path = output_dir / "ocr_output.md"
        with open(md_output_path, "w") as f:
            f.write(full_markdown)

        # 4. Save Raw Response
        json_output_path = output_dir / "raw_ocr_response.json"
        response_dict = json.loads(ocr_response.model_dump_json())
        with open(json_output_path, "w", encoding='utf-8') as f:
            json.dump(response_dict, f, indent=2, ensure_ascii=False)
        print(f"Raw JSON backup saved to: {json_output_path}")

        print(f"="*50)
        print(f"Generated Markdown saved to: {md_output_path}")
        print(f"="*50)
        # print(full_markdown[:500] + "...") # Print preview
        
    else:
        print("No pages found in response.")
        print(ocr_response)

if __name__ == "__main__":
    # Use the image path defined in the user's workflow
    DOC_PATH = "/Users/worakanlasudee/Documents/GitHub/SNT-OCR-RESEARCH/Datasets/Method Precast For Repairing Tunnet Segment (อุโมงค์ระบายน้ำบึงหนองบอ.pdf"
    
    if os.path.exists(DOC_PATH):
        run_mistral_ocr(DOC_PATH)
    else:
        print(f"Error: Image not found at {DOC_PATH}")
