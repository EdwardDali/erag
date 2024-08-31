import os
import glob
from typing import List
from tqdm import tqdm

from src.settings import settings
from src.api_model import EragAPI
from src.look_and_feel import success, info, warning, error

class TextbookGenerator:
    def __init__(self, worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, subject: str):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.subject = subject
        self.output_folder = os.path.join(settings.output_folder, f"{subject.replace(' ', '_').lower()}_textbook")
        os.makedirs(self.output_folder, exist_ok=True)
        self.textbook_file = os.path.join(self.output_folder, f"{subject.replace(' ', '_').lower()}_textbook.txt")

    def save_chapter(self, chapter_number: int, chapter_name: str, content: str) -> str:
        filename = f"chapter_{chapter_number:02d}_{chapter_name.replace(' ', '_').lower()}.txt"
        filepath = os.path.join(self.output_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def generate_chapter_names(self, num_chapters: int = 10, is_subchapter: bool = False) -> List[str]:
        chapter_type = "sub-chapter" if is_subchapter else "chapter"
        prompt = f"""Generate a concise table of contents for {'a section of ' if is_subchapter else ''}a textbook on {self.subject}.
        Please provide a list of {num_chapters} {chapter_type} titles, adhering to these strict guidelines:
        1. Each line must contain ONLY ONE {chapter_type} title.
        2. Start each line with the {chapter_type} number, a period, and a space, followed by the {chapter_type} title.
        3. Do not include any further subdivisions, bullet points, or additional details.
        4. Ensure the {chapter_type}s cover the subject comprehensively and in a logical order.
        5. Keep {chapter_type} titles concise and descriptive.
        6. Do not include any introductory or explanatory text.

        Example format:
        1. Introduction to {self.subject}
        2. Fundamental Principles
        3. Key Concepts and Theories
        ...

        Now, generate the table of contents for {self.subject}, following these rules exactly:"""

        response = self.worker_erag_api.chat([{"role": "user", "content": prompt}])
        print(info(f"API Response for {chapter_type} names:\n{response}"))
        
        # Process the response to ensure it's in the correct format
        chapters = [line.strip() for line in response.split('\n') if line.strip() and '. ' in line]
        
        # Validate and clean up the chapter list
        valid_chapters = []
        for i, chapter in enumerate(chapters, 1):
            parts = chapter.split('. ', 1)
            if len(parts) == 2:
                valid_chapters.append(f"{i}. {parts[1]}")
            else:
                valid_chapters.append(f"{i}. {chapter}")
        
        # Ensure we have the correct number of chapters
        while len(valid_chapters) < num_chapters:
            valid_chapters.append(f"{len(valid_chapters) + 1}. Additional {'Sub-' if is_subchapter else ''}Chapter on {self.subject}")
        valid_chapters = valid_chapters[:num_chapters]
        
        return valid_chapters

    def generate_chapter_content(self, chapter_name: str, chapter_number: int, is_subchapter: bool = False) -> str:
        # Remove the chapter number from the chapter name
        clean_chapter_name = chapter_name.split('. ', 1)[1] if '. ' in chapter_name else chapter_name

        chapter_type = "Sub-chapter" if is_subchapter else "Chapter"
        word_count = "500-750" if is_subchapter else "1000-1500"

        prompt = f"""Write the content for {chapter_type} {chapter_number}: {clean_chapter_name} of our {self.subject} textbook.
        Follow these guidelines strictly:
        1. Begin with a brief introduction to the {chapter_type.lower()}'s topic.
        2. Divide the content into {2 if is_subchapter else '3-5'} main sections, each covering a key aspect of the {chapter_type.lower()}'s subject.
        3. Include relevant examples, explanations, and, where appropriate, simple diagrams or equations (described in text).
        4. Conclude with a summary of the {chapter_type.lower()}'s main points.
        5. Add {'2-3' if is_subchapter else '3-5'} review questions at the end.
        6. Aim for approximately {word_count} words for the entire {chapter_type.lower()}.
        7. Write in a clear, educational style suitable for a textbook.
        8. Do not include any meta-commentary or questions about how to proceed.
        9. Provide the full {chapter_type.lower()} content in your response.

        Generate the complete {chapter_type.lower()} content now:"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.worker_erag_api.chat([{"role": "user", "content": prompt}])
                
                # Basic validation of the response
                if len(response.split()) < 100 or "how would you like me to proceed" in response.lower():
                    raise ValueError("Generated content is too short or contains meta-commentary")

                chapter_content = f"{chapter_type} {chapter_number}: {clean_chapter_name}\n\n{response}\n\n"
                return chapter_content

            except Exception as e:
                print(warning(f"Error generating content for {chapter_type} {chapter_number} (Attempt {attempt + 1}/{max_retries}): {str(e)}"))
                if attempt == max_retries - 1:
                    print(error(f"Failed to generate content for {chapter_type} {chapter_number} after {max_retries} attempts."))
                    return f"{chapter_type} {chapter_number}: {clean_chapter_name}\n\nContent generation failed. Please refer to the textbook outline for this {chapter_type.lower()}'s topics.\n\n"

        # This should never be reached, but just in case:
        return f"{chapter_type} {chapter_number}: {clean_chapter_name}\n\nContent unavailable.\n\n"

    def generate_textbook(self) -> None:
        print(info(f"Generating chapter names for {self.subject}..."))
        chapter_names = self.generate_chapter_names()
        
        # Save table of contents
        with open(self.textbook_file, 'w', encoding='utf-8') as f:
            f.write(f"Textbook: {self.subject}\n\n")
            f.write("Table of Contents\n\n")
            for chapter in chapter_names:
                f.write(f"{chapter}\n")
            f.write("\n")

        print(success(f"Table of Contents saved to {self.textbook_file}"))

        # Generate and save each chapter
        for i, chapter in enumerate(tqdm(chapter_names, desc="Generating chapters"), 1):
            print(info(f"Generating content for {chapter}..."))
            chapter_content = self.generate_chapter_content(chapter, i)
            
            # Save chapter to individual file
            chapter_file = self.save_chapter(i, chapter.split('. ', 1)[1], chapter_content)
            print(success(f"Chapter {i} saved to {chapter_file}"))
            
            # Append chapter to main textbook file
            with open(self.textbook_file, 'a', encoding='utf-8') as f:
                f.write(chapter_content)
                f.write("=" * 50 + "\n\n")

            print(success(f"Chapter {i} appended to {self.textbook_file}"))

        print(success(f"Textbook generation completed. Main file saved as {self.textbook_file}"))
        print(success(f"Individual chapter files saved in {self.output_folder}"))

class SupervisorTextbookGenerator(TextbookGenerator):
    def __init__(self, worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, manager_erag_api: EragAPI, subject: str):
        super().__init__(worker_erag_api, supervisor_erag_api, subject)
        self.manager_erag_api = manager_erag_api
        self.improved_output_folder = os.path.join(self.output_folder, "improved")
        os.makedirs(self.improved_output_folder, exist_ok=True)
        self.improved_textbook_file = os.path.join(self.improved_output_folder, f"{subject.replace(' ', '_').lower()}_improved_textbook.txt")

    def improve_chapter(self, chapter_number: int, original_content: str) -> str:
        print(info(f"Improving Chapter {chapter_number}..."))

        prompt = f"""Review and improve the following chapter content for our textbook on {self.subject}. 
        Your task is to:
        1. Enhance the content with more depth and breadth.
        2. Ensure logical flow and coherence.
        3. Add any missing important information.
        4. Improve clarity and readability.
        5. Ensure appropriate depth and breadth of coverage.
        6. Maintain an educational and engaging tone.

        Original content:
        {original_content}

        Please provide the improved version of the chapter:"""

        improved_content = self.supervisor_erag_api.chat([{"role": "user", "content": prompt}])
        
        if self.manager_erag_api:
            manager_prompt = f"""Review and finalize the following improved chapter content for our textbook on {self.subject}.
            Your task is to:
            1. Ensure the content meets the highest standards of quality and accuracy.
            2. Make any final improvements or adjustments.
            3. Verify that the chapter aligns well with the overall textbook structure.

            Improved content:
            {improved_content}

            Please provide the final, manager-approved version of the chapter:"""
            
            improved_content = self.manager_erag_api.chat([{"role": "user", "content": manager_prompt}])

        return improved_content

    def save_improved_chapter(self, chapter_number: int, content: str) -> str:
        filename = f"improved_chapter_{chapter_number:02d}.txt"
        filepath = os.path.join(self.improved_output_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def generate_textbook(self) -> None:
        # First, generate the original textbook using the worker model
        super().generate_textbook()

        print(info(f"Original textbook generation completed. Now improving each chapter..."))

        # Find all chapter files
        chapter_files = sorted(glob.glob(os.path.join(self.output_folder, "chapter_*.txt")))
        
        # Create a new table of contents for the improved version
        with open(self.improved_textbook_file, 'w', encoding='utf-8') as f:
            f.write(f"Improved Textbook: {self.subject}\n\n")
            f.write("Table of Contents\n\n")

        # Improve each chapter
        for i, chapter_file in enumerate(tqdm(chapter_files, desc="Improving chapters"), 1):
            with open(chapter_file, 'r', encoding='utf-8') as f:
                original_content = f.read()

            print(info(f"Improving content for Chapter {i}..."))
            improved_content = self.improve_chapter(i, original_content)
            
            # Save improved chapter to a new file
            improved_chapter_file = self.save_improved_chapter(i, improved_content)
            print(success(f"Improved Chapter {i} saved to {improved_chapter_file}"))
            
            # Append improved chapter to the new main textbook file
            with open(self.improved_textbook_file, 'a', encoding='utf-8') as f:
                f.write(f"Chapter {i}\n\n")
                f.write(improved_content)
                f.write("\n" + "=" * 50 + "\n\n")

            print(success(f"Improved Chapter {i} appended to {self.improved_textbook_file}"))

        print(success(f"Enhanced textbook generation completed. Improved file saved as {self.improved_textbook_file}"))
        print(success(f"Individual improved chapter files saved in {self.improved_output_folder}"))

def run_create_textbook(worker_erag_api: EragAPI, supervisor_erag_api: EragAPI = None):
    try:
        # Get subject from console input
        subject = input("Enter the subject for the textbook: ")
        if not subject:
            print(error("No subject entered. Exiting textbook generation."))
            return

        # Create and run the appropriate TextbookGenerator
        if supervisor_erag_api:
            generator = SupervisorTextbookGenerator(worker_erag_api, supervisor_erag_api, subject)
        else:
            generator = TextbookGenerator(worker_erag_api, subject)
        
        generator.generate_textbook()

        print(info(f"Textbook generation completed for '{subject}'. Check the output folder for results."))
    except Exception as e:
        error_message = f"An error occurred during the textbook generation process: {str(e)}"
        print(error(error_message))

if __name__ == "__main__":
    from src.api_model import create_erag_api
    api_type = input("Enter the API type (e.g., 'ollama', 'llama', 'groq'): ")
    model = input("Enter the model name: ")
    use_supervisor = input("Use supervisor? (y/n): ").lower() == 'y'
    
    worker_erag_api = create_erag_api(api_type, model)
    supervisor_erag_api = create_erag_api(api_type, model) if use_supervisor else None
    
    run_create_textbook(worker_erag_api, supervisor_erag_api)