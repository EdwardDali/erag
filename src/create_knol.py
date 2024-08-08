# -*- coding: utf-8 -*-

import sys
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import logging
from collections import deque
import networkx as nx
import json
from src.settings import settings
from src.api_model import EragAPI, create_erag_api
from src.look_and_feel import success, info, warning, error, colorize, MAGENTA, RESET, user_input as color_user_input, llm_response as color_llm_response

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SearchUtils:
    def __init__(self, db_content):
        self.db_content = db_content
        self.vectorizer = TfidfVectorizer()
        self.db_vectors = self.vectorizer.fit_transform(self.db_content)

    def get_relevant_context(self, query, conversation_context):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.db_vectors).flatten()
        top_k_indices = similarities.argsort()[-settings.top_k:][::-1]
        
        lexical_context = [self.db_content[i] for i in top_k_indices]
        semantic_context = lexical_context  # In this simplified version, lexical and semantic are the same
        graph_context = []  # Placeholder for graph context
        text_context = conversation_context[-5:]  # Last 5 conversation entries
        
        return lexical_context, semantic_context, graph_context, text_context

class KnolCreator:
    def __init__(self, worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, manager_erag_api: EragAPI = None):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.manager_erag_api = manager_erag_api
        self.db_content = self.load_db_content()
        self.conversation_history = []
        self.new_entries = []
        self.conversation_context = deque(maxlen=settings.conversation_context_size * 2)
        self.knowledge_graph = self.load_knowledge_graph()
        self.search_utils = SearchUtils(self.db_content)
        self.output_folder = None
        os.makedirs(settings.output_folder, exist_ok=True)

    def load_db_content(self):
        if os.path.exists(settings.db_file_path):
            with open(settings.db_file_path, "r", encoding='utf-8') as db_file:
                return db_file.readlines()
        return []

    def load_knowledge_graph(self):
        try:
            graph_path = os.path.join(settings.output_folder, os.path.basename(settings.knowledge_graph_file_path))
            if os.path.exists(graph_path):
                with open(graph_path, 'r') as f:
                    graph_data = json.load(f)
                G = nx.node_link_graph(graph_data)
                logging.info(success(f"Successfully loaded knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."))
                return G
            else:
                logging.warning(warning(f"Knowledge graph file {graph_path} not found. Initializing empty graph."))
                return nx.Graph()
        except Exception as e:
            logging.error(error(f"Failed to load knowledge graph: {str(e)}"))
            return nx.Graph()

    def get_response(self, query: str, system_message: str, api: EragAPI) -> str:
        lexical_context, semantic_context, graph_context, text_context = self.search_utils.get_relevant_context(query, list(self.conversation_context))
        
        combined_context = f"""Conversation Context:\n{' '.join(self.conversation_context)}

Lexical Search Results:
{' '.join(lexical_context)}

Semantic Search Results:
{' '.join(semantic_context)}

Knowledge Graph Context:
{' '.join(graph_context)}

Text Search Results:
{' '.join(text_context)}"""

        messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history,
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuery: {query}"}
        ]

        try:
            response = api.chat(messages, temperature=settings.temperature)
            logging.info(success(f"Generated response for query: {query[:50]}..."))
            return response
        except Exception as e:
            error_message = f"Error in API call: {str(e)}"
            logging.error(error(error_message))
            return f"I'm sorry, but I encountered an error while processing your request: {str(e)}"

    def save_iteration(self, content: str, stage: str, subject: str, iteration: int):
        if self.output_folder is None:
            # Create a new subfolder for this knol creation process
            self.output_folder = os.path.join(settings.output_folder, f"knol_{subject.replace(' ', '_')}")
            os.makedirs(self.output_folder, exist_ok=True)
            logging.info(success(f"Created output folder: {self.output_folder}"))

        filename = f"knol_{subject.replace(' ', '_')}_{stage}_iteration_{iteration}.txt"
        file_path = os.path.join(self.output_folder, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(success(f"Saved {stage} knol (iteration {iteration}) as {file_path}"))
        except IOError as e:
            logging.error(error(f"Error saving {stage} knol to file: {str(e)}"))

    def create_knol(self, subject: str, iteration: int, previous_knol: str = "", manager_review: str = "") -> str:
        system_message = f"""You are a knowledgeable assistant that creates structured, comprehensive, and detailed knowledge entries on specific subjects. When given a topic, provide an exhaustive explanation covering key aspects, sub-topics, and important details. Structure the information as follows:

Title: Knol: {subject}

1. [Main Topic 1]
1.1. [Subtopic 1.1]
- [Key point 1]
- [Key point 2]
...

Ensure that the structure is consistent and the information is detailed and accurate. Aim to provide at least 5 points for each subtopic, but you can include more if necessary. Stay strictly on the topic of {subject} and do not include information about unrelated subjects."""

        if iteration == 1:
            query = f"Create a structured, comprehensive knowledge entry about {subject}. Include main topics, subtopics, and at least 5 key points with detailed information for each subtopic. Focus exclusively on {subject}."
        else:
            query = f"""Improve and expand the following knowledge entry about {subject}, taking into account the manager's review. 
            Your task is to create a new, enhanced version of the knol, not just to provide comments.
            Add more details, ensure at least 5 points per subtopic, and restructure if necessary. 
            Focus on addressing the specific feedback and areas of improvement mentioned in the review.
            Incorporate all the valuable information from the previous version while expanding and improving upon it.

Previous Knol:
{previous_knol}

Manager's Review:
{manager_review}

Please provide a completely updated and improved version of the knol, incorporating all valuable information from the previous version and addressing the manager's feedback."""

        content = self.get_response(query, system_message, self.worker_erag_api)
        self.save_iteration(content, "initial", subject, iteration)
        return content

    def improve_knol(self, knol: str, subject: str, iteration: int, manager_review: str = "") -> str:
        system_message = f"""You are a supervisory assistant tasked with improving and expanding an existing knowledge entry (knol) about {subject}. Your role is to enhance the knol by adding more details, restructuring if necessary, and ensuring comprehensive coverage of the subject. Pay special attention to:

1. Accuracy and depth of information
2. Completeness of coverage
3. Clarity and coherence of presentation
4. Logical structure and flow
5. Appropriate depth of detail

Maintain the following structure, but feel free to add, modify, or reorganize content as needed to improve the overall quality of the knol:

Title: Knol: {subject}

1. [Main Topic]
1.1. [Subtopic]
- [Detailed point 1]
- [Detailed point 2]
...

Aim to provide at least 7 points for each subtopic, but you can include more if necessary. Stay strictly on the topic of {subject} and do not include information about unrelated subjects."""

        query = f"""Improve and expand the following knol about {subject} by adding more details, ensuring at least 7 points per subtopic, and restructuring if necessary. Focus exclusively on {subject}.

Your task is to create a new, enhanced version of the knol, not just to provide comments.
Incorporate all the valuable information from the previous version while expanding and improving upon it.

Original Knol:
{knol}

"""
        if manager_review:
            query += f"""
Please pay special attention to the following manager's review from the previous iteration and address the points raised:

Manager's Review:
{manager_review}

Ensure that you create a fully updated and improved version of the knol, addressing all the feedback and suggestions provided in the manager's review.
"""

        improved_content = self.get_response(query, system_message, self.supervisor_erag_api)
        self.save_iteration(improved_content, "improved", subject, iteration)
        return improved_content

    def generate_questions(self, knol: str, subject: str, iteration: int) -> str:
        system_message = f"""You are a knowledgeable assistant tasked with creating diverse and thought-provoking questions based on a given knowledge entry (knol) about {subject}. Generate {settings.num_questions} questions that cover different aspects of the knol, ranging from factual recall to critical thinking and analysis. Ensure the questions are clear, concise, and directly related to the content of the knol."""

        query = f"""Based on the following knol about {subject}, generate {settings.num_questions} diverse questions:

{knol}

Please provide {settings.num_questions} questions that:
1. Cover different aspects and topics from the knol
2. Include a mix of question types (e.g., factual, analytical, comparative)
3. Are clear and concise
4. Are directly related to the content of the knol
5. Encourage critical thinking and deeper understanding of the subject

Format the output as a numbered list of questions."""

        questions = self.get_response(query, system_message, self.worker_erag_api)
        self.save_iteration(questions, "q", subject, iteration)
        return questions

    def answer_questions(self, questions: str, subject: str, knol: str, iteration: int) -> str:
        answers = []
        question_list = questions.split('\n')
        
        for question in question_list:
            if question.strip():
                print(info(f"Answering: {question}"))
                
                system_message = f"""You are a knowledgeable assistant tasked with answering questions about {subject} based on the provided knol and any additional context. Use the information from both the knol and the additional context to provide accurate and comprehensive answers. If the information is not available in the provided content, state that you don't have enough information to answer accurately."""

                query = f"""Question: {question}

Please provide a comprehensive answer to the question using the information from the knol and the additional context provided. If you can't find relevant information to answer the question accurately, please state so.

Knol Content:
{knol}"""

                answer = self.get_response(query, system_message, self.supervisor_erag_api)
                answers.append(f"Q: {question}\nA: {answer}\n")

        full_qa = "\n".join(answers)
        self.save_iteration(full_qa, "q_a", subject, iteration)
        return full_qa

    def manager_review(self, knol: str, subject: str, iteration: int) -> Tuple[str, float]:
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
        GRADE: [Your numerical grade]/10

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

        End your review with a numerical grade formatted as: GRADE: [Your numerical grade]/10"""

        manager_messages = [
            {"role": "system", "content": manager_system_message},
            {"role": "user", "content": manager_user_input}
        ]
        review = self.manager_erag_api.chat(manager_messages, temperature=settings.temperature)

        # Extract rating from the review
        grade_match = re.search(r'GRADE:\s*(\d+(?:\.\d+)?)\s*\/\s*(\d+)', review)
        if grade_match:
            grade = float(grade_match.group(1))
            scale = int(grade_match.group(2))
            
            if scale == 5:
                rating = (grade / 5) * 10
            elif scale == 10:
                rating = grade
            else:
                print(warning(f"Unexpected rating scale: {scale}. Treating as a 10-point scale."))
                rating = grade
            
            print(info(f"Original grade: {grade}/{scale}, Converted rating: {rating}/10"))
        else:
            print(warning("No grade found in the manager's review. Assigning a default grade of 5."))
            rating = 5.0

        self.save_iteration(review, "manager_review", subject, iteration)
        return review, rating

    def create_final_knol(self, subject: str, iteration: int):
        improved_knol_filename = f"knol_{subject.replace(' ', '_')}_improved_iteration_{iteration}.txt"
        qa_filename = f"knol_{subject.replace(' ', '_')}_q_a_iteration_{iteration}.txt"
        final_knol_filename = f"knol_{subject.replace(' ', '_')}_final_iteration_{iteration}.txt"

        improved_knol_path = os.path.join(self.output_folder, improved_knol_filename)
        qa_path = os.path.join(self.output_folder, qa_filename)
        final_knol_path = os.path.join(self.output_folder, final_knol_filename)

        try:
            with open(improved_knol_path, "r", encoding="utf-8") as f:
                improved_knol_content = f.read()

            with open(qa_path, "r", encoding="utf-8") as f:
                qa_content = f.read()

            final_content = f"{improved_knol_content}\n\nQuestions and Answers:\n\n{qa_content}"

            with open(final_knol_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            logging.info(success(f"Created final knol for iteration {iteration} as {final_knol_path}"))
            return final_content
        except IOError as e:
            logging.error(error(f"Error creating final knol: {str(e)}"))
            return None

    def run_knol_creator(self):
        print(info("Welcome to the Knol Creation System. Type 'exit' to quit."))
        
        while True:
            user_input = input(color_user_input("Which subject do you want to create a knol about? ")).strip()

            if user_input.lower() == 'exit':
                print(success("Thank you for using the Knol Creation System. Goodbye!"))
                break

            if not user_input:
                print(error("Please enter a valid subject."))
                continue

            # Reset the output folder for each new knol
            self.output_folder = None

            iteration = 1
            previous_knol = ""
            previous_review = ""

            while True:
                print(info(f"Creating knol about {user_input} (Iteration {iteration})..."))
                initial_knol = self.create_knol(user_input, iteration, previous_knol, previous_review)

                print(info("Improving and expanding the knol..."))
                improved_knol = self.improve_knol(initial_knol, user_input, iteration, previous_review)

                print(info("Generating questions based on the improved knol..."))
                questions = self.generate_questions(improved_knol, user_input, iteration)

                print(info("Answering questions..."))
                qa_pairs = self.answer_questions(questions, user_input, improved_knol, iteration)

                print(info("Manager review in progress..."))
                manager_review, rating = self.manager_review(improved_knol, user_input, iteration)
                
                print(f"\n{success('Manager Review:')}")
                print(color_llm_response(manager_review))
                print(f"\n{success(f'Manager Rating: {rating:.1f}/10')}")

                if rating >= 8.0:
                    print(success(f"Knol creation process completed for {user_input} with a satisfactory rating."))
                    break
                else:
                    print(info(f"The knol did not meet the required standard (8.0). Current rating: {rating:.1f}/10"))
                    print(info("Proceeding with the next iteration..."))
                    iteration += 1
                    previous_knol = improved_knol
                    previous_review = manager_review

            print(info("Creating final knol..."))
            final_knol = self.create_final_knol(user_input, iteration)

            print(success(f"Knol creation process completed for {user_input}."))
            print(info(f"You can find the results in the folder: {self.output_folder}"))

            print(f"\n{success('Final Improved Knol Content:')}")
            print(color_llm_response(improved_knol))

            print(f"\n{success('Generated Questions and Answers:')}")
            print(color_llm_response(qa_pairs))

def main(api_type: str):
    worker_erag_api = create_erag_api(api_type)
    supervisor_erag_api = create_erag_api(api_type)
    manager_erag_api = create_erag_api(api_type)
    creator = KnolCreator(worker_erag_api, supervisor_erag_api, manager_erag_api)
    creator.run_knol_creator()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        main(api_type)
    else:
        print(error("Error: API type not provided."))
        print(warning("Usage: python src/create_knol.py <api_type>"))
        print(info("Available API types: ollama, llama, groq"))
        sys.exit(1)
