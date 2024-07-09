# -*- coding: utf-8 -*-

import sys
import os
import logging
from src.talk2doc import RAGSystem, ANSIColor  # Updated import
from src.search_utils import SearchUtils  # Updated import
from sentence_transformers import SentenceTransformer
from src.embeddings_utils import load_embeddings_and_data  # Updated import
from src.settings import settings  # Updated import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KnolCreator:
    def __init__(self, api_type: str):
        self.rag_system = RAGSystem(api_type)
        self.search_utils = self.rag_system.search_utils
        # Ensure output folder exists
        os.makedirs(settings.output_folder, exist_ok=True)

    def save_iteration(self, content: str, stage: str, subject: str):
        filename = f"knol_{subject.replace(' ', '_')}_{stage}.txt"
        file_path = os.path.join(settings.output_folder, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Saved {stage} knol as {file_path}")
        except IOError as e:
            logging.error(f"Error saving {stage} knol to file: {str(e)}")

    def create_knol(self, subject: str) -> str:
        system_message = f"""You are a knowledgeable assistant that creates structured, comprehensive, and detailed knowledge entries on specific subjects. When given a topic, provide an exhaustive explanation covering key aspects, sub-topics, and important details. Structure the information as follows:

Title: Knol: {subject}

1. [Main Topic 1]
1.1. [Subtopic 1.1]
• [Key point 1]
• [Key point 2]
• [Key point 3]
• [Key point 4]
• [Key point 5]

1.2. [Subtopic 1.2]
• [Relevant information 1]
• [Relevant information 2]
• [Relevant information 3]
• [Relevant information 4]
• [Relevant information 5]

2. [Main Topic 2]
...

Ensure that the structure is consistent and the information is detailed and accurate. Aim to provide at least 5 points for each subtopic, but you can include more if necessary. Stay strictly on the topic of {subject} and do not include information about unrelated subjects."""

        user_input = f"Create a structured, comprehensive knowledge entry about {subject}. Include main topics, subtopics, and at least 5 key points with detailed information for each subtopic. Focus exclusively on {subject}."

        content = self.rag_system.ollama_chat(user_input, system_message)
        self.save_iteration(content, "initial", subject)
        return content

    def improve_knol(self, knol: str, subject: str) -> str:
        system_message = f"""You are a knowledgeable assistant that improves and expands existing knowledge entries. Given a structured text representing a knol about {subject}, enhance it by adding more details, restructuring if necessary, and ensuring comprehensive coverage of the subject. Maintain the following structure:

Title: Knol: {subject}

1. [Main Topic]
1.1. [Subtopic]
• [Detailed point 1]
• [Detailed point 2]
• [Detailed point 3]
• [Detailed point 4]
• [Detailed point 5]
• [Detailed point 6]
• [Detailed point 7]

Feel free to add new topics, subtopics, or points, and reorganize the structure if it improves the overall quality and comprehensiveness of the knol. Aim to provide at least 7 points for each subtopic, but you can include more if necessary. Stay strictly on the topic of {subject} and do not include information about unrelated subjects."""

        user_input = f"Improve and expand the following knol about {subject} by adding more details, ensuring at least 7 points per subtopic, and restructuring if necessary. Focus exclusively on {subject}:\n\n{knol}"

        improved_content = self.rag_system.ollama_chat(user_input, system_message)
        self.save_iteration(improved_content, "improved", subject)
        return improved_content

    def generate_questions(self, knol: str, subject: str) -> str:
        system_message = f"""You are a knowledgeable assistant tasked with creating diverse and thought-provoking questions based on a given knowledge entry (knol) about {subject}. Generate {settings.num_questions} questions that cover different aspects of the knol, ranging from factual recall to critical thinking and analysis. Ensure the questions are clear, concise, and directly related to the content of the knol."""

        user_input = f"""Based on the following knol about {subject}, generate {settings.num_questions} diverse questions:

{knol}

Please provide {settings.num_questions} questions that:
1. Cover different aspects and topics from the knol but are not fully covered by the current content
2. Include a mix of question types (e.g., factual, analytical, comparative)
3. Are clear and concise
4. Are directly related to the content of the knol
5. Encourage critical thinking and deeper understanding of the subject

Format the output as a numbered list of questions."""

        questions = self.rag_system.ollama_chat(user_input, system_message)
        self.save_iteration(questions, "q", subject)
        return questions

    def answer_questions(self, questions: str, subject: str, knol: str) -> str:
        answers = []
        question_list = questions.split('\n')
        
        for question in question_list:
            if question.strip():
                print(f"Answering: {question}")
                
                system_message = f"""You are a knowledgeable assistant tasked with answering questions about {subject} based on the provided knol and any additional context. Use the information from both the knol and the additional context to provide accurate and comprehensive answers. If the information is not available in the provided content, state that you don't have enough information to answer accurately."""

                user_input = f"""Question: {question}

Knol Content:
{knol}

Please provide a comprehensive answer to the question using the information from the knol and any additional context provided. If you can't find relevant information to answer the question accurately, please state so."""

                answer = self.rag_system.ollama_chat(user_input, system_message)
                answers.append(f"Q: {question}\nA: {answer}\n")

        full_qa = "\n".join(answers)
        self.save_iteration(full_qa, "q_a", subject)
        return full_qa

    def create_final_knol(self, subject: str):
        improved_knol_filename = f"knol_{subject.replace(' ', '_')}_improved.txt"
        qa_filename = f"knol_{subject.replace(' ', '_')}_q_a.txt"
        final_knol_filename = f"knol_{subject.replace(' ', '_')}_final.txt"

        improved_knol_path = os.path.join(settings.output_folder, improved_knol_filename)
        qa_path = os.path.join(settings.output_folder, qa_filename)
        final_knol_path = os.path.join(settings.output_folder, final_knol_filename)

        try:
            # Read the improved knol content
            with open(improved_knol_path, "r", encoding="utf-8") as f:
                improved_knol_content = f.read()

            # Read the Q&A content
            with open(qa_path, "r", encoding="utf-8") as f:
                qa_content = f.read()

            # Combine the contents
            final_content = f"{improved_knol_content}\n\nQuestions and Answers:\n\n{qa_content}"

            # Write the final knol
            with open(final_knol_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            logging.info(f"Created final knol as {final_knol_path}")
            print(f"{ANSIColor.NEON_GREEN.value}Final knol created: {final_knol_path}{ANSIColor.RESET.value}")

        except IOError as e:
            logging.error(f"Error creating final knol: {str(e)}")
            print(f"{ANSIColor.PINK.value}Error creating final knol. See log for details.{ANSIColor.RESET.value}")

    def run_knol_creator(self):
        print(f"{ANSIColor.YELLOW.value}Welcome to the Knol Creation System. Type 'exit' to quit.{ANSIColor.RESET.value}")
        print(f"{ANSIColor.CYAN.value}All generated files will be saved in: {settings.output_folder}{ANSIColor.RESET.value}")

        while True:
            user_input = input(f"{ANSIColor.YELLOW.value}Which subject do you want to create a knol about? {ANSIColor.RESET.value}").strip()

            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using the Knol Creation System. Goodbye!{ANSIColor.RESET.value}")
                break

            if not user_input:
                print(f"{ANSIColor.PINK.value}Please enter a valid subject.{ANSIColor.RESET.value}")
                continue

            print(f"{ANSIColor.CYAN.value}Creating initial knol...{ANSIColor.RESET.value}")
            initial_knol = self.create_knol(user_input)

            print(f"{ANSIColor.CYAN.value}Improving and expanding the knol...{ANSIColor.RESET.value}")
            improved_knol = self.improve_knol(initial_knol, user_input)

            print(f"{ANSIColor.CYAN.value}Generating questions based on the improved knol...{ANSIColor.RESET.value}")
            questions = self.generate_questions(improved_knol, user_input)

            print(f"{ANSIColor.CYAN.value}Answering questions using RAG...{ANSIColor.RESET.value}")
            qa_pairs = self.answer_questions(questions, user_input, improved_knol)

            print(f"{ANSIColor.CYAN.value}Creating final knol...{ANSIColor.RESET.value}")
            self.create_final_knol(user_input)

            print(f"{ANSIColor.NEON_GREEN.value}Knol creation process completed.{ANSIColor.RESET.value}")
            print(f"{ANSIColor.CYAN.value}You can find the results in files with '_initial', '_improved', '_q', '_q_a', and '_final' suffixes.{ANSIColor.RESET.value}")

            # Print the improved knol content, questions, and answers
            print(f"\n{ANSIColor.NEON_GREEN.value}Improved Knol Content:{ANSIColor.RESET.value}")
            print(improved_knol)
            print(f"\n{ANSIColor.NEON_GREEN.value}Generated Questions and Answers:{ANSIColor.RESET.value}")
            print(qa_pairs)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        creator = KnolCreator(api_type)
        creator.run_knol_creator()
    else:
        print("Error: No API type provided.")
        print("Usage: python src/create_knol.py <api_type>")  # Updated usage instruction
        print("Available API types: ollama, llama")
        sys.exit(1)
