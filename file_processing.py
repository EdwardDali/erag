import docx
import json
import PyPDF2
import re
from tkinter import filedialog
from typing import Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_docx() -> Optional[str]:
    file_path = filedialog.askopenfilename(filetypes=[("DOCX Files", "*.docx")])
    if file_path:
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + "\n"
            return text
        except Exception as e:
            logging.error(f"Error processing DOCX file: {str(e)}")
    return None

def upload_json() -> Optional[str]:
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        try:
            with open(file_path, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
                return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error processing JSON file: {str(e)}")
    return None

def upload_pdf() -> Optional[str]:
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                text = ''
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    if page.extract_text():
                        text += page.extract_text() + " "
                return text
        except Exception as e:
            logging.error(f"Error processing PDF file: {str(e)}")
    return None

def upload_txt() -> Optional[str]:
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            with open(file_path, 'r', encoding="utf-8") as txt_file:
                return txt_file.read()
        except Exception as e:
            logging.error(f"Error processing TXT file: {str(e)}")
    return None

def handle_text_chunking(text: str, chunk_size: int = 1000, overlap_size: int = 100) -> List[str]:
    # Normalize whitespace and clean up text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split text into chunks with overlap
    chunks = []
    start = 0
    end = chunk_size
    while start < len(text):
        chunks.append(text[start:end])
        start += chunk_size - overlap_size
        end = start + chunk_size
    
    return chunks

def process_file(file_type: str) -> Optional[List[str]]:
    upload_functions = {
        "DOCX": upload_docx,
        "JSON": upload_json,
        "PDF": upload_pdf,
        "Text": upload_txt
    }
    
    if file_type not in upload_functions:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    text = upload_functions[file_type]()
    if text:
        return handle_text_chunking(text)
    return None
