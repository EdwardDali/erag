import os
import re
from typing import List, Dict, Any
from tqdm import tqdm

from src.settings import settings
from src.api_model import create_erag_api
from src.look_and_feel import success, info, warning, error

class TextbookGenerator:
    def __init__(self, api_type: str, model: str, subject: str):
        self.erag_api = create_erag_api(api_type, model)
        self.subject = subject
        self.output_folder = os.path.join(settings.output_folder, f"{subject.replace(' ', '_').lower()}_textbook")
        os.makedirs(self.output_folder, exist_ok=True)
        self.textbook_file = os.path.join(self.output_folder, f"{subject.replace(' ', '_').lower()}_textbook.txt")

    def generate_chapter_names(self) -> List[str]:
        prompt = f"""Generate a concise table of contents for a textbook on {self.subject}.
        Please provide a list of 8-15 chapter titles, adhering to these strict guidelines:
        1. Each line must contain ONLY ONE chapter title.
        2. Start each line with the chapter number, a period, and a space, followed by the chapter title.
        3. Do not include any subchapters, bullet points, or additional details.
        4. Ensure the chapters cover the subject comprehensively and in a logical order.
        5. Keep chapter titles concise and descriptive.
        6. Do not include any introductory or explanatory text.

        Example format:
        1. Introduction to {self.subject}
        2. Fundamental Principles
        3. Key Concepts and Theories
        ...

        Now, generate the table of contents for {self.subject}, following these rules exactly:"""

        response = self.erag_api.chat([{"role": "user", "content": prompt}])
        print(info(f"API Response for chapter names:\n{response}"))
        
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
        
        # Ensure we have at least 8 chapters and no more than 15
        while len(valid_chapters) < 8:
            valid_chapters.append(f"{len(valid_chapters) + 1}. Additional Chapter on {self.subject}")
        valid_chapters = valid_chapters[:15]  # Limit to 15 chapters if more were generated
        
        return valid_chapters

    def generate_chapter_content(self, chapter_name: str, chapter_number: int) -> str:
        # Remove the chapter number from the chapter name
        clean_chapter_name = chapter_name.split('. ', 1)[1] if '. ' in chapter_name else chapter_name

        prompt = f"""Write the content for Chapter {chapter_number}: {clean_chapter_name} of our {self.subject} textbook.
        Follow these guidelines strictly:
        1. Begin with a brief introduction to the chapter's topic.
        2. Divide the content into 3-5 main sections, each covering a key aspect of the chapter's subject.
        3. Include relevant examples, explanations, and, where appropriate, simple diagrams or equations (described in text).
        4. Conclude with a summary of the chapter's main points.
        5. Add 3-5 review questions at the end.
        6. Aim for approximately 1000-1500 words for the entire chapter.
        7. Write in a clear, educational style suitable for a textbook.
        8. Do not include any meta-commentary or questions about how to proceed.
        9. Provide the full chapter content in your response.

        Generate the complete chapter content now:"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.erag_api.chat([{"role": "user", "content": prompt}])
                
                # Basic validation of the response
                if len(response.split()) < 200 or "how would you like me to proceed" in response.lower():
                    raise ValueError("Generated content is too short or contains meta-commentary")

                chapter_content = f"Chapter {chapter_number}: {clean_chapter_name}\n\n{response}\n\n"
                return chapter_content

            except Exception as e:
                print(warning(f"Error generating content for Chapter {chapter_number} (Attempt {attempt + 1}/{max_retries}): {str(e)}"))
                if attempt == max_retries - 1:
                    print(error(f"Failed to generate content for Chapter {chapter_number} after {max_retries} attempts."))
                    return f"Chapter {chapter_number}: {clean_chapter_name}\n\nContent generation failed. Please refer to the textbook outline for this chapter's topics.\n\n"

        # This should never be reached, but just in case:
        return f"Chapter {chapter_number}: {clean_chapter_name}\n\nContent unavailable.\n\n"

    def generate_textbook(self) -> None:
        print(info(f"Generating chapter names for {self.subject}..."))
        chapter_names = self.generate_chapter_names()
        
        with open(self.textbook_file, 'w', encoding='utf-8') as f:
            f.write(f"Textbook: {self.subject}\n\n")
            f.write("Table of Contents\n\n")
            for chapter in chapter_names:
                f.write(f"{chapter}\n")
            f.write("\n")

        print(success(f"Table of Contents saved to {self.textbook_file}"))

        for i, chapter in enumerate(tqdm(chapter_names, desc="Generating chapters"), 1):
            print(info(f"Generating content for {chapter}..."))
            try:
                chapter_content = self.generate_chapter_content(chapter, i)
                
                with open(self.textbook_file, 'a', encoding='utf-8') as f:
                    f.write(chapter_content)
                    f.write("=" * 50 + "\n\n")

                print(success(f"Chapter {i} appended to {self.textbook_file}"))
            except Exception as e:
                print(error(f"Failed to generate content for Chapter {i}: {str(e)}"))
                with open(self.textbook_file, 'a', encoding='utf-8') as f:
                    f.write(f"Chapter {i}: {chapter}\n\nContent generation failed. Please refer to the table of contents for this chapter's topic.\n\n")
                    f.write("=" * 50 + "\n\n")

        print(success(f"Textbook generation completed. File saved as {self.textbook_file}"))

def run_create_textbook(self):
    try:
        api_type = self.api_type_var.get()
        model = self.model_var.get()

        # Get subject from console input
        subject = input("Enter the subject for the textbook: ")
        if not subject:
            print(error("No subject entered. Exiting textbook generation."))
            return

        # Run the textbook generator in a separate thread
        threading.Thread(target=self._create_textbook_thread, args=(subject, api_type, model), daemon=True).start()

        print(info(f"Textbook generation started for '{subject}'. Check the console for progress."))
    except Exception as e:
        error_message = f"An error occurred while starting the textbook generation process: {str(e)}"
        print(error(error_message))
        messagebox.showerror("Error", error_message)

if __name__ == "__main__":
    print(warning("This module is not meant to be run directly. Import and use run_textbook_generator function in your main script."))