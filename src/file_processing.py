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
from queue import Queue
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileProcessor:
    def __init__(self):
        self.toc = []
        self.ensure_output_folder()
        self.file_queue = Queue()
        self.processing_thread = None
        self.total_files = 0
        self.processed_files = 0

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

    def upload_multiple_files(self, file_type: str) -> int:
        filetypes = {
            "DOCX": [("DOCX Files", "*.docx")],
            "PDF": [("PDF Files", "*.pdf")],
            "Text": [("Text Files", "*.txt")],
            "JSON": [("JSON Files", "*.json")]
        }
        
        file_paths = filedialog.askopenfilenames(filetypes=filetypes[file_type])
        if file_paths:
            self.total_files += len(file_paths)
            for file_path in file_paths:
                self.file_queue.put((file_type, file_path))
            
            print(info(f"{len(file_paths)} {file_type} files added to queue. Total files in queue: {self.total_files}"))
            
            if not self.processing_thread or not self.processing_thread.is_alive():
                self.processing_thread = threading.Thread(target=self.process_file_queue)
                self.processing_thread.start()
            
            return len(file_paths)
        return 0

    def process_file_queue(self):
        print(info(f"Starting to process {self.total_files} files..."))
        while not self.file_queue.empty():
            file_type, file_path = self.file_queue.get()
            self.processed_files += 1
            try:
                text, filename = self.process_single_file(file_type, file_path)
                if text:
                    chunks = self.handle_text_chunking(text)
                    print(info(f"Processing {filename}... ({self.processed_files}/{self.total_files})"))
                    print(info("Extracting or generating table of contents..."))
                    
                    toc_str = self.format_toc()
                    print(info(f"Method: TOC generated using {file_type}-specific method"))
                    
                    if not toc_str or toc_str.strip() == "":
                        print(warning(f"Warning: Failed to generate a table of contents for {filename}."))
                        toc_str = f"No table of contents could be generated for {filename}."
                    
                    print(info("Appending to db_content.txt..."))
                    self.append_to_db_content(filename, toc_str)
                    self.append_to_db(chunks)
                    print(success(f"Processed and appended {filename} to db.txt and db_content.txt. ({self.processed_files}/{self.total_files})"))
                else:
                    print(warning(f"Warning: {filename} was empty or could not be processed. ({self.processed_files}/{self.total_files})"))
            except Exception as e:
                print(error(f"Error processing {file_path}: {str(e)} ({self.processed_files}/{self.total_files})"))
            finally:
                self.file_queue.task_done()

        print(success(f"All files processed. Total: {self.total_files}"))
        self.total_files = 0
        self.processed_files = 0

    def process_single_file(self, file_type: str, file_path: str) -> Optional[Tuple[str, str]]:
        try:
            if file_type == "DOCX":
                doc = docx.Document(file_path)
                text = " ".join([paragraph.text for paragraph in doc.paragraphs])
                self.toc = self.generate_toc_docx(doc)
            elif file_type == "PDF":
                with fitz.open(file_path) as doc:
                    text = " ".join([page.get_text() for page in doc])
                self.toc = self.generate_toc_pdf(file_path)
            elif file_type == "Text":
                with open(file_path, 'r', encoding="utf-8") as txt_file:
                    text = txt_file.read()
                toc_text = self.extract_toc_from_text(text)
                self.toc = [(1, line, 0) for line in toc_text.split('\n') if line.strip()]
            elif file_type == "JSON":
                with open(file_path, 'r', encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    text = json.dumps(data, indent=2)
                    self.toc = self.generate_toc_json(data)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return text, os.path.basename(file_path)
        except Exception as e:
            logging.error(f"Error processing {file_type} file: {str(e)}")
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

# Expose the upload_multiple_files function
def upload_multiple_files(file_type: str) -> int:
    return file_processor.upload_multiple_files(file_type)

if __name__ == "__main__":
    settings.load_settings()
    file_types = ["DOCX", "PDF", "Text", "JSON"]
    for file_type in file_types:
        upload_multiple_files(file_type)
