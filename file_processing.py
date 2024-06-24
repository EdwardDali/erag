import docx
import json
import PyPDF2
import re
from tkinter import filedialog
from typing import Optional, List, Dict, Tuple
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHUNK_SIZE = 1000
OVERLAP_SIZE = 100

def format_db_entry(content: str, reference: Dict[str, str]) -> Tuple[str, str]:
    """Format a single database entry for both content-only and full versions."""
    content_only = f"{content.strip()}\n"
    full_entry = f"{content.strip()} | {json.dumps(reference)}\n"
    return content_only, full_entry

def upload_docx() -> Optional[Tuple[str, str]]:
    file_path = filedialog.askopenfilename(filetypes=[("DOCX Files", "*.docx")])
    if file_path:
        try:
            doc = docx.Document(file_path)
            text = " ".join([paragraph.text for paragraph in doc.paragraphs])
            return text, os.path.basename(file_path)
        except Exception as e:
            logging.error(f"Error processing DOCX file: {str(e)}")
    return None, None

def upload_pdf() -> Optional[Tuple[str, str]]:
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = " ".join([page.extract_text() for page in pdf_reader.pages])
                return text, os.path.basename(file_path)
        except Exception as e:
            logging.error(f"Error processing PDF file: {str(e)}")
    return None, None

def upload_txt() -> Optional[Tuple[str, str]]:
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            with open(file_path, 'r', encoding="utf-8") as txt_file:
                text = txt_file.read()
                return text, os.path.basename(file_path)
        except Exception as e:
            logging.error(f"Error processing TXT file: {str(e)}")
    return None, None

def handle_text_chunking(text: str, source: str) -> List[Dict[str, str]]:
    # Normalize whitespace and clean up text
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    start = 0
    chunk_num = 1

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        
        # Ensure we don't cut words in half
        if end < len(text):
            last_space = chunk_text.rfind(' ')
            if last_space != -1:
                end = start + last_space
                chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text.strip(),
            "reference": {
                "source": source,
                "chunk": chunk_num
            }
        })

        # Move start for next chunk, ensuring overlap
        start = end - OVERLAP_SIZE
        chunk_num += 1

    return chunks

def process_file(file_type: str) -> Optional[List[Dict[str, str]]]:
    upload_functions = {
        "DOCX": upload_docx,
        "PDF": upload_pdf,
        "Text": upload_txt
    }
    
    if file_type not in upload_functions:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    text, source = upload_functions[file_type]()
    if text and source:
        return handle_text_chunking(text, source)
    return None

def append_to_db(chunks: List[Dict[str, str]], db_file: str = "db.txt", db_r_file: str = "db_r.txt"):
    with open(db_file, "a", encoding="utf-8") as f, open(db_r_file, "a", encoding="utf-8") as f_r:
        for chunk in chunks:
            content_only, full_entry = format_db_entry(chunk['text'], chunk['reference'])
            f.write(content_only)
            f_r.write(full_entry)
    logging.info(f"Appended {len(chunks)} chunks to {db_file} (content-only) and {db_r_file} (full entries)")

if __name__ == "__main__":
    file_types = ["DOCX", "PDF", "Text"]
    for file_type in file_types:
        chunks = process_file(file_type)
        if chunks:
            append_to_db(chunks)
