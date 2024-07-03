import docx
import PyPDF2
import re
from tkinter import filedialog
from typing import Optional, List, Tuple
import logging
import os
from settings import settings
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHUNK_SIZE = 500
OVERLAP_SIZE = 200

def set_chunk_sizes(chunk_size: int, overlap_size: int):
    settings.file_chunk_size = chunk_size
    settings.file_overlap_size = overlap_size
    logging.info(f"File chunk size set to {settings.file_chunk_size} and overlap size set to {settings.file_overlap_size}")

def format_db_entry(content: str) -> str:
    """Format a single database entry."""
    return f"{content.strip()}\n"

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

def upload_json() -> Optional[Tuple[str, str]]:
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        try:
            with open(file_path, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
                text = json.dumps(data, indent=2)  # Convert JSON to formatted string
                return text, os.path.basename(file_path)
        except Exception as e:
            logging.error(f"Error processing JSON file: {str(e)}")
    return None, None

def handle_text_chunking(text: str) -> List[str]:
    # Normalize whitespace and clean up text
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    start = 0

    while start < len(text):
        end = start + settings.file_chunk_size
        chunk_text = text[start:end]
        
        # Ensure we don't cut words in half
        if end < len(text):
            last_space = chunk_text.rfind(' ')
            if last_space != -1:
                end = start + last_space
                chunk_text = text[start:end]

        chunks.append(chunk_text.strip())

        # Move start for next chunk, ensuring overlap
        start = end - settings.file_overlap_size

    return chunks

def process_file(file_type: str) -> Optional[List[str]]:
    upload_functions = {
        "DOCX": upload_docx,
        "PDF": upload_pdf,
        "Text": upload_txt,
        "JSON": upload_json
    }
    
    if file_type not in upload_functions:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    text, _ = upload_functions[file_type]()
    if text:
        return handle_text_chunking(text)
    return None

def append_to_db(chunks: List[str], db_file: str = "db.txt"):
    with open(db_file, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(format_db_entry(chunk))
    logging.info(f"Appended {len(chunks)} chunks to {db_file}")

if __name__ == "__main__":
    file_types = ["DOCX", "PDF", "Text", "JSON"]
    for file_type in file_types:
        chunks = process_file(file_type)
        if chunks:
            append_to_db(chunks)
