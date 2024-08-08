import sys
import logging
from enum import Enum
from src.settings import settings
import re
import os
from src.api_model import EragAPI, create_erag_api
from src.look_and_feel import success, info, warning, error, highlight

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RouteQuery:
    def __init__(self, erag_api: EragAPI):
        self.erag_api = erag_api
        # Ensure output folder exists
        os.makedirs(settings.output_folder, exist_ok=True)
        self.db_content = self.load_db_content()
        


    def load_db_content(self):
        logging.info("Loading database content...")
        db_content_path = os.path.join(settings.output_folder, 'db_content.txt')
        try:
            with open(db_content_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logging.info(f"Successfully loaded database content from {db_content_path}")
            logging.info(f"Loaded db_content (first 100 characters): {content[:100]}...")
            return content
        except FileNotFoundError:
            logging.error(f"{db_content_path} not found")
            print(warning("Warning: db_content.txt not found. Please upload and process documents first."))
            print(info("You can do this by:"))
            print("1. Using the 'Upload' buttons in the main tab to process documents")
            print("2. Clicking 'Execute Embeddings' to generate the necessary files for using Doc RAG")
            print("3. Optionally, creating a knowledge graph using 'Create Knowledge Graph'")
            return ""

    def evaluate_query(self, query: str) -> dict:
        logging.info(f"Evaluating query: {query}")
        
        db_content = self.load_db_content()
        
        system_message = """You are an intelligent query router with expertise in analyzing user queries and database content. Your task is to determine the most appropriate way to handle a given query based on the available information and the capabilities of our systems. 

        IMPORTANT: The database content provided is a Table of Contents (TOC) summary, not the full content. The presence of relevant entries in this TOC is a strong indicator that detailed information is available in the local documents.

        Follow these steps:

        1. Carefully analyze the query and the provided TOC summary.
        2. Determine the relevance of the database content to the query:
           - High: If the TOC mentions topics directly related to the query. This strongly suggests that relevant detailed information is available locally.
           - Low: If the TOC does not contain any entries related to the query topic.

        3. Assess the complexity of the query:
           - Simple: If the query can be answered with a straightforward explanation or fact.
           - Deep: If the query requires a comprehensive analysis, multiple aspects, or extensive information.

        4. Based on your evaluation, recommend one of the following options:
    First check if there is relevant information in the TOC; if yes chose A or B. If there is no relevant information in TOC chose C or D. For complex query not in the TOC choose D.
           A. talk2doc (RAGSystem): 
              - in TOC: yes
              - simple: yes

           B. create_knol (KnolCreator):
              - in TOC: yes
              - simple: no

           C. web_rag (WebRAG):
              - in TOC: no
              - simple: yes

           D. web_sum (WebSum):
              - in TOC: no
              - simple: no

        5. Provide a detailed explanation for your recommendation, referencing specific parts of the query or TOC that influenced your decision. Consider the unique capabilities of each system in your reasoning.

        Format your response as follows:
        Relevance: [high/low]
        Complexity: [simple/deep]
        Recommendation: [A/B/C/D]
        Explanation: [Your detailed explanation here]"""

        user_message = f"Query: {query}\n\nDatabase Content Summary:\n{db_content}\n\nPlease evaluate the query and the database content summary, then provide your recommendation."
        
        logging.info(f"User message sent to LLM: {user_message[:100]}...")  # Log the first 100 characters of the user message

        try:
            logging.info("Sending request to LLM...")
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            llm_response = self.erag_api.chat(messages, temperature=settings.temperature)
            
            print(info("LLM-generated response:"))
            print(llm_response)
            print(info("Parsing LLM response..."))
            
            parsed_response = self.parse_evaluation(llm_response)
            
            return parsed_response
        except Exception as e:
            logging.error(f"Error in LLM response: {str(e)}. Using default routing.")
            return {
                "recommendation": "C",
                "relevance": "low",
                "complexity": "simple",
                "explanation": f"Default routing due to error in LLM response: {str(e)}"
            }

    def parse_evaluation(self, response: str) -> dict:
        lines = response.split('\n')
        evaluation = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key == 'relevance':
                    evaluation['relevance'] = 'high' if 'high' in value.lower() else 'low'
                elif key == 'complexity':
                    evaluation['complexity'] = 'deep' if 'deep' in value.lower() else 'simple'
                elif key == 'recommendation':
                    # Extract the letter before any additional text
                    match = re.match(r'([A-D])', value.upper())
                    evaluation['recommendation'] = match.group(1) if match else 'C'
                elif key == 'explanation':
                    evaluation['explanation'] = value

        # Ensure all keys are present
        for key in ['relevance', 'complexity', 'recommendation', 'explanation']:
            if key not in evaluation:
                evaluation[key] = 'N/A'

        return evaluation

    def load_component(self, component_name: str):
        logging.info(f"Loading component: {component_name}")
        if component_name == 'talk2doc':
            from src.talk2doc import RAGSystem
            return RAGSystem(self.erag_api)
        elif component_name == 'create_knol':
            from src.create_knol import KnolCreator
            return KnolCreator(self.erag_api, self.erag_api)  # Using the same API for both worker and supervisor
        elif component_name == 'web_rag':
            from src.web_rag import WebRAG
            return WebRAG(self.erag_api)
        elif component_name == 'web_sum':
            from src.web_sum import WebSum
            return WebSum(self.erag_api)
        else:
            raise ValueError(f"Unknown component: {component_name}")


    def route_query(self, query: str, evaluation: dict):
        print(info("Routing decision:"))
        print(f"Relevance: {evaluation['relevance']}")
        print(f"Complexity: {evaluation['complexity']}")
        print(f"Recommendation: {evaluation['recommendation']}")
        print(f"Explanation: {evaluation['explanation']}")

        system_to_load = {
            'A': 'talk2doc',
            'B': 'create_knol',
            'C': 'web_rag',
            'D': 'web_sum'
        }.get(evaluation['recommendation'], 'web_rag')

        print(info(f"Loading system: {system_to_load}"))
        component = self.load_component(system_to_load)
        print(success(f"System loaded: {system_to_load}"))

        if system_to_load == 'talk2doc':
            response = component.get_response(query)
        elif system_to_load == 'create_knol':
            component.run_knol_creator()
            response = "Knol creation process completed. Please check the generated files for results."
        elif system_to_load in ['web_rag', 'web_sum']:
            response = component.search_and_process(query)
        else:
            response = "Error: Invalid system selected."

        print(success("Response:"))
        print(response)

    def run(self):
        print(highlight("Welcome to the Query Routing System. Type 'exit' to quit."))
        print(info(f"Using output folder: {settings.output_folder}"))

        # Check if db_content.txt exists and is not empty
        db_content = self.load_db_content()
        if not db_content:
            print(error("Error: Cannot start the Query Routing System. Please upload and process documents first."))
            return

        while True:
            user_input = input(info("Enter your query: ")).strip()

            if user_input.lower() == 'exit':
                print(success("Thank you for using the Query Routing System. Goodbye!"))
                break

            if not user_input:
                print(warning("Please enter a valid query."))
                continue

            print(info("Evaluating your query..."))
            evaluation = self.evaluate_query(user_input)

            self.route_query(user_input, evaluation)

def main(api_type: str):
    erag_api = create_erag_api(api_type)
    route_query = RouteQuery(erag_api)
    route_query.run()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        main(api_type)
    else:
        print(error("No API type provided."))
        print(warning("Usage: python src/route_query.py <api_type>"))
        print(info("Available API types: ollama, llama"))
        sys.exit(1)
