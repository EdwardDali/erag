# -*- coding: utf-8 -*-

import sys
import os
import logging
from src.talk2doc import RAGSystem
from src.search_utils import SearchUtils
from sentence_transformers import SentenceTransformer
from src.embeddings_utils import load_embeddings_and_data
from src.settings import settings
from src.api_model import configure_api, LlamaClient
from src.look_and_feel import (
    RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE,
    DUSTY_PINK, SAGE_GREEN,
    BOLD, UNDERLINE, RESET,
    error, success, warning, info, highlight,
    user_input as color_user_input,
    llm_response as color_llm_response
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KnolCreator:
    def __init__(self, api_type: str):
        self.api_type = api_type
        if api_type == "llama":
            self.client = LlamaClient()
        else:
            self.client = configure_api(api_type)
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

        if self.api_type == "llama":
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ]
            content = self.client.chat(messages, temperature=settings.temperature)
        else:
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

        if self.api_type == "llama":
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ]
            improved_content = self.client.chat(messages, temperature=settings.temperature)
        else:
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

        if self.api_type == "llama":
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ]
            questions = self.client.chat(messages, temperature=settings.temperature)
        else:
            questions = self.rag_system.ollama_chat(user_input, system_message)

        self.save_iteration(questions, "q", subject)
        return questions

    def answer_questions(self, questions: str, subject: str, knol: str) -> str:
        answers = []
        question_list = questions.split('\n')
        
        for question in question_list:
            if question.strip():
                print(info(f"Answering: {question}"))
                
                system_message = f"""You are a knowledgeable assistant tasked with answering questions about {subject} based on the provided knol and any additional context. Use the information from both the knol and the additional context to provide accurate and comprehensive answers. If the information is not available in the provided content, state that you don't have enough information to answer accurately."""

                user_input = f"""Question: {question}

Knol Content:
{knol}

Please provide a comprehensive answer to the question using the information from the knol and any additional context provided. If you can't find relevant information to answer the question accurately, please state so."""

                if self.api_type == "llama":
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_input}
                    ]
                    answer = self.client.chat(messages, temperature=settings.temperature)
                else:
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
            print(success(f"Final knol created: {final_knol_path}"))

        except IOError as e:
            logging.error(f"Error creating final knol: {str(e)}")
            print(error("Error creating final knol. See log for details."))

    def run_knol_creator(self):
        print(highlight("Welcome to the Knol Creation System. Type 'exit' to quit."))
        print(info(f"All generated files will be saved in: {settings.output_folder}"))

        while True:
            user_input = input(color_user_input("Which subject do you want to create a knol about? ")).strip()

            if user_input.lower() == 'exit':
                print(success("Thank you for using the Knol Creation System. Goodbye!"))
                break

            if not user_input:
                print(error("Please enter a valid subject."))
                continue

            print(info("Creating initial knol..."))
            initial_knol = self.create_knol(user_input)

            print(info("Improving and expanding the knol..."))
            improved_knol = self.improve_knol(initial_knol, user_input)

            print(info("Generating questions based on the improved knol..."))
            questions = self.generate_questions(improved_knol, user_input)

            print(info("Answering questions using RAG..."))
            qa_pairs = self.answer_questions(questions, user_input, improved_knol)

            print(info("Creating final knol..."))
            self.create_final_knol(user_input)

            print(success("Knol creation process completed."))
            print(info("You can find the results in files with '_initial', '_improved', '_q', '_q_a', and '_final' suffixes."))

            # Print the improved knol content, questions, and answers
            print(f"\n{highlight('Improved Knol Content:')}")
            print(color_llm_response(improved_knol))
            print(f"\n{highlight('Generated Questions and Answers:')}")
            print(color_llm_response(qa_pairs))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        creator = KnolCreator(api_type)
        creator.run_knol_creator()
    else:
        print(error("Error: No API type provided."))
        print(warning("Usage: python src/create_knol.py <api_type>"))
        print(info("Available API types: ollama, llama"))
        sys.exit(1)
