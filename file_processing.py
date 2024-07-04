import docx
import PyPDF2
import re
from tkinter import filedialog
from typing import Optional, List, Tuple
import logging
import os
from settings import settings
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileProcessor:
    def __init__(self):
        self.client = self.configure_api(settings.ollama_model)
        self.model = SentenceTransformer(settings.model_name)

    @staticmethod
    def configure_api(model_name: str) -> OpenAI:
        return OpenAI(base_url='http://localhost:11434/v1', api_key=model_name)

    def generate_table_of_contents(self, text: str, filename: str) -> str:
        # Remove page numbers and ellipsis characters
        cleaned_text = re.sub(r'\s*\d+\s*', ' ', text)  # Remove page numbers
        cleaned_text = re.sub(r'\.{3,}', '', cleaned_text)  # Remove ellipsis

        prompt = f"""
        Generate a detailed table of contents for the following text from the file '{filename}'. 
        Please follow these guidelines:
        1. Identify main chapters, sections, and subsections, focusing on the core content.
        2. Maintain the original structure and numbering of chapters and sections when present.
        3. DO NOT include page numbers.
        4. Preserve any detailed subheadings or bullet points that provide valuable information about the content.
        5. Use a hierarchical structure, indenting subsections appropriately.
        6. Include all relevant entries, even if the total number exceeds 15.
        7. Ignore about the author info, credits, legal information, release dates, and other non-core content.
        8. Format the table of contents to closely match the original structure, including any special formatting or symbols, but excluding ellipsis (...) characters.
        9. DO NOT invent or add any content that is not present in the given text. If you're unsure about any part, leave it out rather than guessing.
        10. If the text doesn't contain a clear structure or chapters, create a simple list of main topics or key points found in the text.
        11. When a chapter or section already has a descriptive name, use it exactly as it appears in the text. DO NOT summarize or modify existing chapter/section names.

        Here's the text:

        {cleaned_text[:3000]}...  

        Please provide a detailed and accurately structured table of contents based on these guidelines, using ONLY the information present in the given text.
        """

        response = self.client.chat.completions.create(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates detailed and accurate tables of contents, closely matching the original document structure. You never add information that isn't in the original text, and you preserve original chapter and section names exactly as they appear."},
                {"role": "user", "content": prompt}
            ],
            temperature=settings.temperature
        )

        return response.choices[0].message.content

    def upload_docx(self) -> Optional[Tuple[str, str]]:
        file_path = filedialog.askopenfilename(filetypes=[("DOCX Files", "*.docx")])
        if file_path:
            try:
                doc = docx.Document(file_path)
                text = " ".join([paragraph.text for paragraph in doc.paragraphs])
                return text, os.path.basename(file_path)
            except Exception as e:
                logging.error(f"Error processing DOCX file: {str(e)}")
        return None, None

    def upload_pdf(self) -> Optional[Tuple[str, str]]:
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

    def upload_txt(self) -> Optional[Tuple[str, str]]:
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r', encoding="utf-8") as txt_file:
                    text = txt_file.read()
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
                    text = json.dumps(data, indent=2)  # Convert JSON to formatted string
                    return text, os.path.basename(file_path)
            except Exception as e:
                logging.error(f"Error processing JSON file: {str(e)}")
        return None, None

    def handle_text_chunking(self, text: str) -> List[str]:
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

    def process_file(self, file_type: str) -> Optional[List[str]]:
        upload_functions = {
            "DOCX": self.upload_docx,
            "PDF": self.upload_pdf,
            "Text": self.upload_txt,
            "JSON": self.upload_json
        }
        
        if file_type not in upload_functions:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        text, filename = upload_functions[file_type]()
        if text:
            print(f"Processing {filename}...")
            chunks = self.handle_text_chunking(text)
            print("Generating table of contents...")
            toc = self.generate_table_of_contents(text, filename)
            print("Appending to db_content.txt...")
            self.append_to_db_content(filename, toc)
            return chunks
        return None

    def append_to_db_content(self, filename: str, table_of_contents: str):
        db_content_file = "db_content.txt"
        
        with open(db_content_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- {filename} ---\n")
            f.write("Table of Contents:\n")
            f.write(table_of_contents)
            f.write("\n\n")  # Add some separation between files

    def append_to_db(self, chunks: List[str], db_file: str = "db.txt"):
        total_chunks = len(chunks)
        with open(db_file, "a", encoding="utf-8") as f:
            for i, chunk in enumerate(tqdm(chunks, desc="Appending chunks", unit="chunk")):
                f.write(f"{chunk.strip()}\n")
                # Update progress every 10% or for each chunk if total_chunks < 10
                if total_chunks < 10 or (i + 1) % (total_chunks // 10) == 0:
                    print(f"Progress: {(i + 1) / total_chunks * 100:.1f}% ({i + 1}/{total_chunks} chunks)")
        print(f"Appended {total_chunks} chunks to {db_file}")

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
