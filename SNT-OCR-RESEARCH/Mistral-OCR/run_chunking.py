import json
from pathlib import Path

def process_chunks(manifest_path):
    print(f"Loading manifest from: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    final_chunks = []
    
    # Configuration
    MIN_CHUNK_SIZE = 100  # Merge if smaller than this
    MAX_CHUNK_SIZE = 2000 # Split if larger than this
    
    def recursive_split(text, limit):
        if len(text) <= limit:
            return [text]
        
        # Try splitting by paragraph
        parts = text.split('\n\n')
        if len(parts) > 1:
            # Re-assemble safely
            new_chunks = []
            buffer = ""
            for part in parts:
                if len(buffer) + len(part) < limit:
                    buffer += "\n\n" + part if buffer else part
                else:
                    if buffer: new_chunks.append(buffer)
                    buffer = part
            if buffer: new_chunks.append(buffer)
            # If any part is STILL too big, drill down
            final_res = []
            for c in new_chunks:
                final_res.extend(recursive_split(c, limit))
            return final_res
            
        # Try splitting by sentence (period)
        parts = text.split('. ')
        if len(parts) > 1:
            new_chunks = []
            buffer = ""
            for part in parts:
                if len(buffer) + len(part) < limit:
                    buffer += ". " + part if buffer else part
                else:
                    if buffer: new_chunks.append(buffer)
                    buffer = part
            if buffer: new_chunks.append(buffer)
            return new_chunks
            
        # Hard split as last resort
        return [text[i:i+limit] for i in range(0, len(text), limit)]

    # State tracking for Context-Rich Chunking (Persist across pages)
    current_heading = "Unknown Section"
    current_subheading = ""

    # Pre-scan Phase: Find the document title from Page 1
    # This ensures that even the first image gets the correct Section Title
    first_page = next((p for p in data.get("pages", []) if p.get("page_num") == 1), None)
    if first_page:
        for el in first_page.get("elements", []):
            if el.get("type") == "text" and len(el.get("text_content", "")) > 5:
                potential_title = el.get("text_content", "").split('\n')[0].strip()
                if len(potential_title) < 100:
                    current_heading = potential_title
                    print(f"🔍 Auto-Detected Document Title: {current_heading}")
                    break

    for page in data.get("pages", []):
        page_num = page.get("page_num", 1)
        elements = page.get("elements", [])
        
        # Buffer for merging small text chunks
        text_buffer = ""
        buffer_metadata = {}

        for el in elements:
            el_type = el.get("type", "text")
            content = el.get("text_content", "") or el.get("description", "")
            
            if not content.strip():
                continue

            # Heuristic: If we are on Page 1 and still "Unknown Section", 
            # assume the first significant text is the Title/Header.
            if page_num == 1 and current_heading == "Unknown Section" and el_type == "text" and len(content) > 5:
                # Use first line of content as heuristic title
                potential_title = content.split('\n')[0].strip()
                if len(potential_title) < 100: # Sanity check
                     current_heading = potential_title
                     # Also create a header chunk for it
                     final_chunks.append({
                        "id": f"heuristic_header_{page_num}",
                        "text": f"Section: {current_heading}\nHeading: {current_heading}",
                        "metadata": {"page": page_num, "type": "heading", "section": current_heading, "heuristic": True}
                    })

            # Update Context
            if el_type == "heading":
                # Flushing buffer before changing section
                if text_buffer:
                    final_chunks.append({
                        "id": f"merged_p{page_num}",
                        "text": f"Document Section: {current_heading}\n\n{text_buffer}",
                        "metadata": {**buffer_metadata, "merged": True}
                    })
                    text_buffer = ""

                level = el.get("level", 1)
                if level == 1:
                    current_heading = content
                    current_subheading = ""
                elif level > 1:
                    current_subheading = content
                
                # Add Heading as chunk
                final_chunks.append({
                    "id": el.get("id"),
                    "text": f"Section: {current_heading}\nHeading: {content}",
                    "metadata": {"page": page_num, "type": "heading", "section": current_heading}
                })
            
            else:
                # Prepare context string
                context_prefix = f"Document Section: {current_heading}"
                if current_subheading:
                    context_prefix += f" > {current_subheading}"

                # Decision: Merge, Append, or Split?
                if len(content) < MIN_CHUNK_SIZE and el_type == "text":
                    # Buffer it
                    text_buffer += "\n" + content
                    buffer_metadata = {
                        "page": page_num,
                        "type": "text",
                        "section": current_heading
                    }
                else:
                    # Flush buffer first if exists
                    if text_buffer:
                        final_chunks.append({
                            "id": f"merged_prev_p{page_num}",
                            "text": f"{context_prefix}\n\n{text_buffer}",
                            "metadata": {"page": page_num, "type": "text", "section": current_heading}
                        })
                        text_buffer = ""

                    # Check if oversized
                    if len(content) > MAX_CHUNK_SIZE:
                        split_parts = recursive_split(content, MAX_CHUNK_SIZE)
                        for idx, part in enumerate(split_parts):
                            final_chunks.append({
                                "id": f"{el.get('id')}_part{idx+1}",
                                "text": f"{context_prefix}\n\n{part}",
                                "metadata": {
                                    "page": page_num,
                                    "type": el_type,
                                    "section": current_heading,
                                    "split_part": idx + 1
                                }
                            })
                    else:
                        # Standard Chunk
                        final_chunks.append({
                            "id": el.get("id"),
                            "text": f"{context_prefix}\n\n{content}",
                            "metadata": {
                                "page": page_num,
                                "type": el_type,
                                "section": current_heading,
                                "image_path": el.get("image_path", None)
                            }
                        })
        
        # Flush buffer at end of page
        if text_buffer:
             final_chunks.append({
                "id": f"merged_end_p{page_num}",
                "text": f"Document Section: {current_heading}\n\n{text_buffer}",
                "metadata": {"page": page_num, "type": "text", "section": current_heading}
            })

    return final_chunks

if __name__ == "__main__":
    # Path configuration
    base_dir = Path("Mistral-OCR")
    input_manifest = base_dir / "output/rag_manifest_v2.json"
    output_chunks = base_dir / "output/final_chunks.json"
    
    if not input_manifest.exists():
        # Fallback for nested path if user's previous path was weird
        input_manifest = base_dir / "Mistral-OCR/output/rag_manifest_v2.json"
        output_chunks = base_dir / "Mistral-OCR/output/final_chunks.json"

    if input_manifest.exists():
        chunks = process_chunks(input_manifest)
        
        # Save Result
        with open(output_chunks, "w", encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Chunking Complete!")
        print(f"Generated {len(chunks)} context-aware chunks.")
        print(f"Saved to: {output_chunks}")
        
        # Preview
        if chunks:
            print("\n--- Example Chunk 1 ---")
            print(chunks[0]['text'])
            print("-----------------------")
    else:
        print(f"❌ Error: Manifest file not found at {input_manifest}")
