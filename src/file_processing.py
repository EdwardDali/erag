import re
import docx
import fitz  # PyMuPDF
from tkinter import filedialog
from typing import Optional, List, Tuple
import logging
import os
from src.settings import settings
import json
from src.look_and_feel import success, info, warning, error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileProcessor:
    def __init__(self):
        self.toc = []
        self.ensure_output_folder()

    def ensure_output_folder(self):
        os.makedirs(settings.output_folder, exist_ok=True)

    def extract_toc_from_text(self, text: str) -> str:
        # Look for the "Contents" marker
        contents_match = re.search(r'\n\s*Contents\s*\n', text, re.IGNORECASE)
        if not contents_match:
            return "No table of contents found"

        start_index = contents_match.end()
        
        # Extract lines until we hit a clear end or reach a reasonable limit
        lines = text[start_index:].split('\n')
        toc_lines = []
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if re.match(r'^CHAPTER [IVXLC]+\.?\s+', stripped_line, re.IGNORECASE):
                toc_lines.append(stripped_line)
            elif len(toc_lines) > 0 and not re.match(r'^CHAPTER', stripped_line, re.IGNORECASE):
                # If we've already started collecting chapters and this isn't a new chapter,
                # it might be the start of the main content
                break
            if len(toc_lines) >= 50:  # Reasonable limit for number of chapters
                break

        clean_toc = '\n'.join(toc_lines)
        return clean_toc

    def generate_toc_docx(self, doc: docx.Document) -> List[Tuple[int, str, int]]:
        toc = []
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                level = int(para.style.name[-1])
                toc.append((level, para.text, 0))  # 0 as placeholder for page number
        return toc

    def generate_toc_pdf(self, pdf_path: str) -> List[Tuple[int, str, int]]:
        with fitz.open(pdf_path) as doc:
            toc = doc.get_toc()
        return [(level, title, page) for level, title, page in toc]

    def generate_toc_json(self, data: dict, prefix: str = "") -> List[Tuple[int, str, int]]:
        toc = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            toc.append((1, full_key, 0))  # 0 as placeholder for page number
            if isinstance(value, dict):
                toc.extend(self.generate_toc_json(value, full_key))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                toc.append((2, f"{full_key} (list of objects)", 0))
        return toc

    def upload_docx(self) -> Optional[Tuple[str, str]]:
        file_path = filedialog.askopenfilename(filetypes=[("DOCX Files", "*.docx")])
        if file_path:
            try:
                doc = docx.Document(file_path)
                text = " ".join([paragraph.text for paragraph in doc.paragraphs])
                self.toc = self.generate_toc_docx(doc)
                return text, os.path.basename(file_path)
            except Exception as e:
                logging.error(f"Error processing DOCX file: {str(e)}")
        return None, None

    def upload_pdf(self) -> Optional[Tuple[str, str]]:
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            try:
                with fitz.open(file_path) as doc:
                    text = " ".join([page.get_text() for page in doc])
                self.toc = self.generate_toc_pdf(file_path)
                return text, os.path.basename(file_path)
            except Exception as e:
                logging.error(f"Error processing PDF file: {str(e)}")
        return None, None

    def upload_txt(self) -> Optional[Tuple[str, str]]:
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r', encoding="utf-8") as txt_file:
                    text = txt_file.read()
                toc_text = self.extract_toc_from_text(text)
                self.toc = [(1, line, 0) for line in toc_text.split('\n') if line.strip()]
                return text, os.path.basename(file_path)
            except Exception as e:
                logging.error(f"Error processing TXT file: {str(e)}")
        return None, None

    def upload_json(self) -> Optional[Tuple[str, str]]:
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r', encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    text = json.dumps(data, indent=2)
                    self.toc = self.generate_toc_json(data)
                    return text, os.path.basename(file_path)
            except Exception as e:
                logging.error(f"Error processing JSON file: {str(e)}")
        return None, None

    def handle_text_chunking(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = start + settings.file_chunk_size
            chunk_text = text[start:end]
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space != -1:
                    end = start + last_space
                    chunk_text = text[start:end]
            chunks.append(chunk_text.strip())
            start = end - settings.file_overlap_size
        return chunks

    def process_file(self, file_type: str) -> Optional[List[str]]:
        upload_functions = {
            "DOCX": self.upload_docx,
            "PDF": self.upload_pdf,
            "Text": self.upload_txt,
            "JSON": self.upload_json
        }
        
        if file_type not in upload_functions:
            raise ValueError(error(f"Unsupported file type: {file_type}"))
        
        text, filename = upload_functions[file_type]()
        if text:
            print(info(f"Processing {filename}..."))
            chunks = self.handle_text_chunking(text)
            print(info("Extracting or generating table of contents..."))
            
            toc_str = self.format_toc()
            print(info(f"Method: TOC generated using {file_type}-specific method"))
            
            if not toc_str or toc_str.strip() == "":
                print(warning("Warning: Failed to generate a table of contents."))
                toc_str = "No table of contents could be generated for this file."
            
            print(info("TOC content:"))
            print(toc_str)
            
            print(info("Appending to db_content.txt..."))
            self.append_to_db_content(filename, toc_str)
            return chunks
        return None

    def format_toc(self) -> str:
        formatted_toc = []
        for level, title, page in self.toc:
            indent = "  " * (level - 1)
            formatted_toc.append(f"{indent}- {title}")
        return '\n'.join(formatted_toc)

    def append_to_db_content(self, filename: str, table_of_contents: str):
        db_content_file = os.path.join(settings.output_folder, "db_content.txt")
        with open(db_content_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- {filename} ---\n")
            f.write("Table of Contents:\n")
            f.write(table_of_contents)
            f.write("\n\n")

    def append_to_db(self, chunks: List[str], db_file: str = "db.txt"):
        db_file_path = os.path.join(settings.output_folder, db_file)
        total_chunks = len(chunks)
        with open(db_file_path, "a", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"{chunk.strip()}\n")
                if total_chunks < 10 or (i + 1) % (total_chunks // 10) == 0:
                    print(info(f"Progress: {(i + 1) / total_chunks * 100:.1f}% ({i + 1}/{total_chunks} chunks)"))
        print(success(f"Appended {total_chunks} chunks to {db_file_path}"))

# Create a global instance of FileProcessor
file_processor = FileProcessor()

# Expose the process_file and append_to_db functions
def process_file(file_type: str) -> Optional[List[str]]:
    return file_processor.process_file(file_type)

def append_to_db(chunks: List[str], db_file: str = "db.txt"):
    file_processor.append_to_db(chunks, db_file)

if __name__ == "__main__":
    settings.load_settings()
    file_types = ["DOCX", "PDF", "Text", "JSON"]
    for file_type in file_types:
        chunks = process_file(file_type)
        if chunks:
            append_to_db(chunks)
