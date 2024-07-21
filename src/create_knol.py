# -*- coding: utf-8 -*-

import sys
import os
import re
import logging
from src.talk2doc import RAGSystem
from src.search_utils import SearchUtils
from src.settings import settings
from src.api_model import EragAPI, create_erag_api
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
    def __init__(self, worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, manager_erag_api: EragAPI = None):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.manager_erag_api = manager_erag_api
        self.rag_system = RAGSystem(self.worker_erag_api)
        self.search_utils = self.rag_system.search_utils
        os.makedirs(settings.output_folder, exist_ok=True)

    def save_iteration(self, content: str, stage: str, subject: str, iteration: int):
        filename = f"knol_{subject.replace(' ', '_')}_{stage}_iteration_{iteration}.txt"
        file_path = os.path.join(settings.output_folder, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Saved {stage} knol (iteration {iteration}) as {file_path}")
        except IOError as e:
            logging.error(f"Error saving {stage} knol to file: {str(e)}")

    def create_knol(self, subject: str, iteration: int, feedback: str = "") -> str:
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
        
        if feedback:
            user_input += f"\n\nPlease address the following feedback in your revision:\n{feedback}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]
        content = self.worker_erag_api.chat(messages, temperature=settings.temperature)
        
        self.save_iteration(content, "initial", subject, iteration)
        return content

    def improve_knol(self, knol: str, subject: str, iteration: int) -> str:
        supervisor_system_message = f"""You are a supervisory assistant tasked with improving and expanding an existing knowledge entry (knol) about {subject}. Your role is to enhance the knol by adding more details, restructuring if necessary, and ensuring comprehensive coverage of the subject. Pay special attention to:

        1. Accuracy and depth of information
        2. Completeness of coverage
        3. Clarity and coherence of presentation
        4. Logical structure and flow
        5. Appropriate depth of detail

        Maintain the following structure, but feel free to add, modify, or reorganize content as needed to improve the overall quality of the knol:

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

        Aim to provide at least 7 points for each subtopic, but you can include more if necessary. Stay strictly on the topic of {subject} and do not include information about unrelated subjects."""

        supervisor_user_input = f"Improve and expand the following knol about {subject} by adding more details, ensuring at least 7 points per subtopic, and restructuring if necessary. Focus exclusively on {subject}:\n\n{knol}"

        supervisor_messages = [
            {"role": "system", "content": supervisor_system_message},
            {"role": "user", "content": supervisor_user_input}
        ]
        improved_content = self.supervisor_erag_api.chat(supervisor_messages, temperature=settings.temperature)

        self.save_iteration(improved_content, "improved", subject, iteration)
        return improved_content

    def generate_questions(self, knol: str, subject: str, iteration: int) -> str:
        worker_system_message = f"""You are a knowledgeable assistant tasked with creating diverse and thought-provoking questions based on a given knowledge entry (knol) about {subject}. Generate {settings.num_questions} questions that cover different aspects of the knol, ranging from factual recall to critical thinking and analysis. Ensure the questions are clear, concise, and directly related to the content of the knol."""

        worker_user_input = f"""Based on the following knol about {subject}, generate {settings.num_questions} diverse questions:

        {knol}

        Please provide {settings.num_questions} questions that:
        1. Cover different aspects and topics from the knol but are not fully covered by the current content
        2. Include a mix of question types (e.g., factual, analytical, comparative)
        3. Are clear and concise
        4. Are directly related to the content of the knol
        5. Encourage critical thinking and deeper understanding of the subject

        Format the output as a numbered list of questions."""

        worker_messages = [
            {"role": "system", "content": worker_system_message},
            {"role": "user", "content": worker_user_input}
        ]
        worker_questions = self.worker_erag_api.chat(worker_messages, temperature=settings.temperature)

        # Now, let the supervisor refine the questions
        supervisor_system_message = f"""You are a supervisory assistant tasked with reviewing and refining a set of questions about {subject}. Your role is to ensure the questions are of high quality, diverse, and encourage deep understanding of the subject. Review the provided questions and make any necessary improvements or additions."""

        supervisor_user_input = f"""Please review and refine the following questions about {subject}:

        {worker_questions}

        Ensure that the questions:
        1. Are clear and well-formulated
        2. Cover a diverse range of aspects related to the subject
        3. Encourage critical thinking and deeper analysis
        4. Are relevant to the content of the knol
        5. Include a mix of question types (e.g., factual, analytical, comparative)

        Provide the refined list of questions, maintaining the numbered format."""

        supervisor_messages = [
            {"role": "system", "content": supervisor_system_message},
            {"role": "user", "content": supervisor_user_input}
        ]
        refined_questions = self.supervisor_erag_api.chat(supervisor_messages, temperature=settings.temperature)

        self.save_iteration(refined_questions, "q", subject, iteration)
        return refined_questions

    def answer_questions(self, questions: str, subject: str, knol: str, iteration: int) -> str:
        answers = []
        question_list = questions.split('\n')
        
        for question in question_list:
            if question.strip():
                print(info(f"Answering: {question}"))
                
                # Worker generates initial answer
                worker_system_message = f"""You are a knowledgeable assistant tasked with answering questions about {subject} based on the provided knol and any additional context. Use the information from both the knol and the additional context to provide accurate and comprehensive answers. If the information is not available in the provided content, state that you don't have enough information to answer accurately."""

                worker_user_input = f"""Question: {question}

                Knol Content:
                {knol}

                Please provide a comprehensive answer to the question using the information from the knol and any additional context provided. If you can't find relevant information to answer the question accurately, please state so."""

                worker_messages = [
                    {"role": "system", "content": worker_system_message},
                    {"role": "user", "content": worker_user_input}
                ]
                worker_answer = self.worker_erag_api.chat(worker_messages, temperature=settings.temperature)
                
                # Supervisor reviews and improves the answer
                supervisor_system_message = f"""You are a supervisory assistant tasked with reviewing and enhancing an answer about {subject}. Your role is to ensure the highest quality, accuracy, and comprehensiveness of the answer. Review the provided answer, and make any necessary enhancements, corrections, or additions. Pay special attention to:

                1. Accuracy of information
                2. Completeness of the answer
                3. Clarity and coherence
                4. Appropriate depth of detail
                5. Relevance to the question

                Provide an improved version of the answer that maintains its core content while enhancing its overall quality."""

                supervisor_user_input = f"""Question: {question}

                Original Answer: {worker_answer}

                Please review and enhance this answer, ensuring it meets the highest standards of quality, accuracy, and comprehensiveness."""

                supervisor_messages = [
                    {"role": "system", "content": supervisor_system_message},
                    {"role": "user", "content": supervisor_user_input}
                ]
                improved_answer = self.supervisor_erag_api.chat(supervisor_messages, temperature=settings.temperature)

                answers.append(f"Q: {question}\nA: {improved_answer}\n")

        full_qa = "\n".join(answers)
        self.save_iteration(full_qa, "q_a", subject, iteration)
        return full_qa

    def manager_review(self, knol: str, subject: str, iteration: int) -> tuple:
        if self.manager_erag_api is None:
            print(info("Manager review skipped as no manager model was selected."))
            return "Manager review skipped.", 10.0
        manager_system_message = f"""You are a managerial assistant tasked with critically reviewing and evaluating a knowledge entry (knol) about {subject}. Your role is to:

        1. Evaluate the overall quality, comprehensiveness, and accuracy of the knol
        2. Identify specific areas for improvement
        3. Pose questions that could enhance the knol's content
        4. Provide constructive feedback
        5. Rate the knol on a scale of 1 to 10

        Be thorough and critical in your evaluation, aiming to improve the knol to the highest possible standard. 
        Your review should be constructive but demanding, pointing out both strengths and weaknesses.

        IMPORTANT: You must provide a numerical grade between 1 and 10 at the end of your review, formatted as follows:
        GRADE: [Your numerical grade]

        This grade should reflect your critical evaluation, where:
        1-3: Poor quality, major revisions needed
        4-5: Below average, significant improvements required
        6-7: Average, several areas need improvement
        8-9: Good quality, minor improvements needed
        10: Excellent, meets the highest standards"""

        manager_user_input = f"""Please review the following knol about {subject} and provide:

        1. An overall evaluation (strengths and weaknesses)
        2. Specific areas for improvement
        3. Questions that could enhance the content
        4. Constructive feedback for the next iteration
        5. A rating on a scale of 1 to 10

        Remember to be critical and demanding in your evaluation. We are aiming for the highest possible quality.

        Knol Content:
        {knol}

        End your review with a numerical grade formatted as: GRADE: [Your numerical grade]"""

        manager_messages = [
            {"role": "system", "content": manager_system_message},
            {"role": "user", "content": manager_user_input}
        ]
        review = self.manager_erag_api.chat(manager_messages, temperature=settings.temperature)

        # Extract rating from the review
        grade_match = re.search(r'GRADE:\s*(\d+(?:\.\d+)?)', review)
        if grade_match:
            rating = float(grade_match.group(1))
        else:
            print(warning("No grade found in the manager's review. Assigning a default grade of 5."))
            rating = 5.0

        self.save_iteration(review, "manager_review", subject, iteration)
        return review, rating

    def create_final_knol(self, subject: str, iteration: int):
        improved_knol_filename = f"knol_{subject.replace(' ', '_')}_improved_iteration_{iteration}.txt"
        qa_filename = f"knol_{subject.replace(' ', '_')}_q_a_iteration_{iteration}.txt"
        final_knol_filename = f"knol_{subject.replace(' ', '_')}_final_iteration_{iteration}.txt"

        improved_knol_path = os.path.join(settings.output_folder, improved_knol_filename)
        qa_path = os.path.join(settings.output_folder, qa_filename)
        final_knol_path = os.path.join(settings.output_folder, final_knol_filename)

        try:
            with open(improved_knol_path, "r", encoding="utf-8") as f:
                improved_knol_content = f.read()

            with open(qa_path, "r", encoding="utf-8") as f:
                qa_content = f.read()

            final_content = f"{improved_knol_content}\n\nQuestions and Answers:\n\n{qa_content}"

            with open(final_knol_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            logging.info(f"Created final knol for iteration {iteration} as {final_knol_path}")
            print(success(f"Final knol for iteration {iteration} created: {final_knol_path}"))
            return final_content
        except IOError as e:
            logging.error(f"Error creating final knol: {str(e)}")
            print(error("Error creating final knol. See log for details."))
            return None

    def run_knol_creator(self):
        print(highlight("Welcome to the Knol Creation System. Type 'exit' to quit."))
        print(info(f"All generated files will be saved in: {settings.output_folder}"))
        print(info(f"Using Worker EragAPI with {self.worker_erag_api.api_type} backend and model: {self.worker_erag_api.model}"))
        print(info(f"Using Supervisor EragAPI with {self.supervisor_erag_api.api_type} backend and model: {self.supervisor_erag_api.model}"))
        if self.manager_erag_api:
            print(info(f"Using Manager EragAPI with {self.manager_erag_api.api_type} backend and model: {self.manager_erag_api.model}"))
        else:
            print(info("Manager model not selected. Manager review will be skipped."))

        while True:
            user_input = input(color_user_input("Which subject do you want to create a knol about? ")).strip()

            if user_input.lower() == 'exit':
                print(success("Thank you for using the Knol Creation System. Goodbye!"))
                break

            if not user_input:
                print(error("Please enter a valid subject."))
                continue

            iteration = 1
            feedback = ""
            while True:
                print(info(f"Starting iteration {iteration}..."))
                
                print(info("Creating initial knol..."))
                initial_knol = self.create_knol(user_input, iteration, feedback)

                print(info("Improving and expanding the knol..."))
                improved_knol = self.improve_knol(initial_knol, user_input, iteration)

                print(info("Generating questions based on the improved knol..."))
                questions = self.generate_questions(improved_knol, user_input, iteration)

                print(info("Answering questions using RAG..."))
                qa_pairs = self.answer_questions(questions, user_input, improved_knol, iteration)

                print(info("Creating final knol..."))
                final_knol = self.create_final_knol(user_input, iteration)

                if self.manager_erag_api:
                    print(info("Manager reviewing the final knol..."))
                    review, rating = self.manager_review(final_knol, user_input, iteration)

                    print(f"\n{highlight('Manager Review:')}")
                    print(color_llm_response(review))
                    print(f"\n{highlight(f'Manager Rating: {rating}/10')}")

                    if rating >= 8:
                        print(success(f"Knol creation process completed after {iteration} iterations."))
                        break
                    else:
                        print(warning(f"Manager rating below 8. Starting iteration {iteration + 1}..."))
                        feedback = review
                        iteration += 1
                else:
                    print(success(f"Knol creation process completed after {iteration} iterations."))
                    break

            print(info("You can find the results in files with '_initial', '_improved', '_q', '_q_a', and '_final' suffixes for each iteration."))

            # Print the final improved knol content, questions, and answers
            print(f"\n{highlight('Final Improved Knol Content:')}")
            print(color_llm_response(final_knol))

            print(f"\n{highlight('Generated Questions and Answers:')}")
            print(color_llm_response(qa_pairs))

            # Ask if the user wants to create another knol
            another = input(color_user_input("Do you want to create another knol? (yes/no): ")).strip().lower()
            if another != 'yes':
                print(success("Thank you for using the Knol Creation System. Goodbye!"))
                break


def main(worker_api_type: str, supervisor_api_type: str, manager_api_type: str):
    worker_erag_api = create_erag_api(worker_api_type)
    supervisor_erag_api = create_erag_api(supervisor_api_type)
    manager_erag_api = create_erag_api(manager_api_type)
    creator = KnolCreator(worker_erag_api, supervisor_erag_api, manager_erag_api)
    creator.run_knol_creator()

if __name__ == "__main__":
    if len(sys.argv) > 3:
        worker_api_type = sys.argv[1]
        supervisor_api_type = sys.argv[2]
        manager_api_type = sys.argv[3]
        main(worker_api_type, supervisor_api_type, manager_api_type)
    else:
        print(error("Error: Worker, Supervisor, and Manager API types not provided."))
        print(warning("Usage: python src/create_knol.py <worker_api_type> <supervisor_api_type> <manager_api_type>"))
        print(info("Available API types: ollama, llama, groq"))
        sys.exit(1)
