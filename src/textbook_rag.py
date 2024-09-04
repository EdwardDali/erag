import os
import re
from typing import List, Tuple
from tqdm import tqdm
import glob

from src.settings import settings
from src.api_model import EragAPI
from src.look_and_feel import success, info, warning, error
from src.search_utils import SearchUtils
from sentence_transformers import SentenceTransformer
from src.embeddings_utils import load_or_compute_embeddings

class RagTextbookGenerator:
    def __init__(self, worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, manager_erag_api: EragAPI = None):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.manager_erag_api = manager_erag_api
        self.embedding_model = SentenceTransformer(settings.sentence_transformer_model)
        self.db_embeddings, self.db_indexes, self.db_content = self.load_embeddings()
        self.search_utils = SearchUtils(self.worker_erag_api, self.embedding_model, self.db_embeddings, self.db_content, None)  # Pass None for knowledge_graph
        self.output_folder = None
        self.improved_output_folder = None
        self.textbook_file = None
        self.improved_textbook_file = None

    def load_embeddings(self):
        embeddings, indexes, content = load_or_compute_embeddings(self.worker_erag_api, settings.db_file_path, settings.embeddings_file_path)
        return embeddings, indexes, content

    def get_rag_response(self, query: str, system_message: str, api: EragAPI) -> str:
        lexical_context, semantic_context, graph_context, text_context = self.search_utils.get_relevant_context(query, "")
        
        combined_context = f"""Lexical Search Results:
        {' '.join(lexical_context)}

        Semantic Search Results:
        {' '.join(semantic_context)}

        Knowledge Graph Context:
        {' '.join(graph_context)}

        Text Search Results:
        {' '.join(text_context)}"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuery: {query}"}
        ]

        try:
            response = api.chat(messages, temperature=settings.temperature)
            return response
        except Exception as e:
            error_message = f"Error in API call: {str(e)}"
            print(error(error_message))
            return f"I'm sorry, but I encountered an error while processing your request: {str(e)}"

    def generate_chapter_names(self, subject: str, num_chapters: int = 10) -> List[str]:
        system_message = "You are an expert in creating structured outlines for textbooks."
        query = f"Generate a list of {num_chapters} chapter titles for a textbook on {subject}. Each line should contain only the chapter title, numbered from 1 to {num_chapters}."

        response = self.get_rag_response(query, system_message, self.worker_erag_api)
        chapter_names = [line.strip() for line in response.split('\n') if line.strip()]
        return chapter_names[:num_chapters]

    def generate_subchapter_names(self, chapter_name: str, num_subchapters: int = 10) -> List[str]:
        system_message = "You are an expert in creating detailed outlines for textbook chapters."
        query = f"""Generate a list of {num_subchapters} subchapter titles for the chapter '{chapter_name}'. 
        Ensure these subchapters expand on the main chapter topic without repeating the chapter's title or main content.
        Each line should contain only the subchapter title, numbered from 1 to {num_subchapters}."""

        response = self.get_rag_response(query, system_message, self.worker_erag_api)
        subchapter_names = [line.strip() for line in response.split('\n') if line.strip()]
        return subchapter_names[:num_subchapters]

    def generate_content(self, title: str, is_subchapter: bool = False) -> str:
        content_type = "subchapter" if is_subchapter else "chapter"
        system_message = f"You are an expert in writing detailed and informative textbook content for {content_type}s."
        query = f"Write the content for the {content_type} titled '{title}'. Provide detailed explanations, examples, and ensure the content is comprehensive and educational."

        return self.get_rag_response(query, system_message, self.supervisor_erag_api)

    def sanitize_filename(self, filename):
        # Remove invalid characters and replace spaces with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        sanitized = sanitized.replace(' ', '_')
        # Ensure the filename is not too long
        return sanitized[:255]  # Maximum filename length in most file systems

    def save_content(self, content: str, filename: str, is_improved: bool = False):
        folder = self.improved_output_folder if is_improved else self.output_folder
        sanitized_filename = self.sanitize_filename(filename)
        file_path = os.path.join(folder, sanitized_filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(success(f"Saved content to {file_path}"))

    def run_rag_textbook_generator(self, subject: str):
        self.output_folder = os.path.join(settings.output_folder, f"{subject.replace(' ', '_').lower()}_rag_textbook")
        os.makedirs(self.output_folder, exist_ok=True)
        self.textbook_file = os.path.join(self.output_folder, f"{subject.replace(' ', '_').lower()}_rag_textbook.txt")

        print(info(f"Generating chapter names for {subject}..."))
        chapter_names = self.generate_chapter_names(subject, num_chapters=10)

        # Save table of contents
        with open(self.textbook_file, 'w', encoding='utf-8') as f:
            f.write(f"RAG Textbook: {subject}\n\n")
            f.write("Table of Contents\n\n")
            for chapter in chapter_names:
                f.write(f"{chapter}\n")
            f.write("\n")

        # Generate and save each chapter
        for chapter_num, chapter_name in enumerate(tqdm(chapter_names, desc="Generating chapters"), 1):
            print(info(f"Generating content for Chapter {chapter_num}: {chapter_name}"))
            chapter_content = self.generate_content(chapter_name)
            
            # Save chapter to individual file
            self.save_content(chapter_content, f"chapter_{chapter_num:02d}_{chapter_name.replace(' ', '_').lower()}.txt")
            
            # Append chapter to main textbook file
            with open(self.textbook_file, 'a', encoding='utf-8') as f:
                f.write(f"Chapter {chapter_num}: {chapter_name}\n\n")
                f.write(chapter_content)
                f.write("\n" + "=" * 50 + "\n\n")

        print(success(f"Basic RAG Textbook generation completed for {subject}. Check the output folder for results."))

        if self.supervisor_erag_api:
            self.expand_textbook(subject)

    def expand_textbook(self, subject: str):
        self.improved_output_folder = os.path.join(self.output_folder, "improved")
        os.makedirs(self.improved_output_folder, exist_ok=True)
        self.improved_textbook_file = os.path.join(self.improved_output_folder, f"{subject.replace(' ', '_').lower()}_improved_rag_textbook.txt")

        print(info(f"Expanding RAG Textbook for {subject}..."))

        # Create a new table of contents for the improved version
        with open(self.improved_textbook_file, 'w', encoding='utf-8') as f:
            f.write(f"Expanded RAG Textbook: {subject}\n\n")
            f.write("Table of Contents\n\n")

        # First, copy the original table of contents
        with open(self.textbook_file, 'r', encoding='utf-8') as original, open(self.improved_textbook_file, 'a', encoding='utf-8') as improved:
            for line in original:
                if line.strip() == "Table of Contents":
                    improved.write(line)
                    break
            for line in original:
                if line.strip().startswith("Chapter 1:"):
                    break
                improved.write(line)
            improved.write("\n" + "=" * 50 + "\n\n")

        # Now expand each chapter
        for chapter_num in tqdm(range(1, 11), desc="Expanding chapters"):
            chapter_file = os.path.join(self.output_folder, f"chapter_{chapter_num:02d}_*.txt")
            chapter_files = glob.glob(chapter_file)
            if not chapter_files:
                print(warning(f"Chapter {chapter_num} file not found. Skipping."))
                continue
            
            with open(chapter_files[0], 'r', encoding='utf-8') as f:
                original_content = f.read()

            # More robust chapter name extraction
            chapter_name = f"Chapter {chapter_num}"
            for line in original_content.split('\n'):
                if line.strip().startswith(f"Chapter {chapter_num}:"):
                    chapter_name = line.strip()
                    break
            
            print(info(f"Expanding content for {chapter_name}"))

            # Keep the original chapter content
            improved_content = original_content + "\n\n"

            # Generate and add subchapters
            subchapter_names = self.generate_subchapter_names(chapter_name, num_subchapters=10)
            
            for subchapter_num, subchapter_name in enumerate(tqdm(subchapter_names, desc=f"Generating subchapters for {chapter_name}"), 1):
                subchapter_content = self.generate_content(subchapter_name, is_subchapter=True)
                improved_content += f"{chapter_num}.{subchapter_num}. {subchapter_name}\n\n{subchapter_content}\n\n"

            # Final review by manager (if available)
            if self.manager_erag_api:
                system_message = "You are a senior textbook editor reviewing and finalizing expanded chapter content."
                query = f"Review and finalize the following expanded chapter content for our textbook on {subject}. Ensure quality, accuracy, and alignment with the overall textbook structure. Preserve the original high-level structure and introductory content."
                improved_content = self.get_rag_response(query + "\n\n" + improved_content, system_message, self.manager_erag_api)

            # Save improved chapter
            self.save_content(improved_content, f"improved_{chapter_name.replace(':', '')}.txt", is_improved=True)
            
            # Append improved chapter to the new main textbook file
            with open(self.improved_textbook_file, 'a', encoding='utf-8') as f:
                f.write(improved_content)
                f.write("\n" + "=" * 50 + "\n\n")

        print(success(f"Expanded RAG textbook generation completed. Improved file saved as {self.improved_textbook_file}"))


def run_rag_textbook_generator():
    self.subject = subject
    from src.api_model import create_erag_api
    api_type = input("Enter the API type (e.g., 'ollama', 'llama', 'groq'): ")
    worker_model = input("Enter the worker model name: ")
    supervisor_model = input("Enter the supervisor model name: ")
    use_manager = input("Use manager model? (y/n): ").lower() == 'y'
    subject = input("Enter the subject for the textbook: ")

    if not subject:
        print(error("No subject entered. Exiting textbook generation."))
        return

    worker_erag_api = create_erag_api(api_type, worker_model)
    supervisor_erag_api = create_erag_api(api_type, supervisor_model)
    manager_erag_api = create_erag_api(api_type, supervisor_model) if use_manager else None
    
    generator = RagTextbookGenerator(worker_erag_api, supervisor_erag_api, manager_erag_api)
    generator.run_rag_textbook_generator(subject)  # Pass the subject here

if __name__ == "__main__":
    run_rag_textbook_generator()