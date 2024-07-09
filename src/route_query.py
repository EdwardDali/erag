import sys
import logging
from openai import OpenAI
from enum import Enum
from src.settings import settings
import re
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ANSIColor(Enum):
    PINK = '\033[95m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    NEON_GREEN = '\033[92m'
    RESET = '\033[0m'

class RouteQuery:
    def __init__(self, api_type: str, client: OpenAI):
        self.api_type = api_type
        self.client = client
        # Ensure output folder exists
        os.makedirs(settings.output_folder, exist_ok=True)

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
            print(f"\n{ANSIColor.PINK.value}Warning: db_content.txt not found. Please upload and process documents first.{ANSIColor.RESET.value}")
            print("You can do this by:")
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
        Explanation: [Your detailed explanation here]

        Remember:
        - The presence of relevant topics in the TOC is a strong indicator to use local systems (A or B). Only recommend web-based options (C or D) if the TOC clearly lacks relevant entries.
        - For queries about current events or rapidly changing information not likely to be in our local database, you may consider web-based options even if some related content exists in the TOC.
        - Always explain your reasoning, especially when choosing between talk2doc and create_knol for locally available information."""

        user_message = f"Query: {query}\n\nDatabase Content Summary:\n{db_content}\n\nPlease evaluate the query and the database content summary, then provide your recommendation."
        
        logging.info(f"User message sent to LLM: {user_message[:100]}...")  # Log the first 100 characters of the user message

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        try:
            logging.info("Sending request to LLM...")
            response = self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=messages,
                temperature=settings.temperature
            )
            
            llm_response = response.choices[0].message.content
            
            print(f"\n{ANSIColor.CYAN.value}LLM-generated response:{ANSIColor.RESET.value}")
            print(llm_response)
            print(f"\n{ANSIColor.YELLOW.value}Parsing LLM response...{ANSIColor.RESET.value}")
            
            parsed_response = self.parse_evaluation(llm_response)
            
            return parsed_response
        except Exception as e:
            logging.error(f"Error in LLM response: {str(e)}. Using default routing.")
            return {
                "recommendation": "C",
                "relevance": "low",
                "complexity": "simple",
                "explanation": "Default routing due to error in LLM response."
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
            from src.talk2doc import RAGSystem  # Updated import
            return RAGSystem(self.api_type)
        elif component_name == 'create_knol':
            from src.create_knol import KnolCreator  # Updated import
            return KnolCreator(self.api_type)
        elif component_name == 'web_rag':
            from src.web_rag import WebRAG  # Updated import
            return WebRAG(self.api_type)
        elif component_name == 'web_sum':
            from src.web_sum import WebSum  # Updated import
            return WebSum(self.api_type)
        else:
            raise ValueError(f"Unknown component: {component_name}")

    def route_query(self, query: str, evaluation: dict):
        print(f"{ANSIColor.CYAN.value}Routing decision:{ANSIColor.RESET.value}")
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

        print(f"\n{ANSIColor.YELLOW.value}Loading system: {system_to_load}{ANSIColor.RESET.value}")
        component = self.load_component(system_to_load)
        print(f"{ANSIColor.NEON_GREEN.value}System loaded: {system_to_load}{ANSIColor.RESET.value}")

        if system_to_load == 'talk2doc':
            response = component.ollama_chat(query, "You are a helpful assistant. Please respond to the user's query based on the available context.")
        elif system_to_load == 'create_knol':
            component.run_knol_creator()
            response = "Knol creation process completed. Please check the generated files for results."
        elif system_to_load in ['web_rag', 'web_sum']:
            response = component.search_and_process(query)
        else:
            response = "Error: Invalid system selected."

        print(f"\n{ANSIColor.NEON_GREEN.value}Response:{ANSIColor.RESET.value}")
        print(response)

    def run(self):
        print(f"{ANSIColor.YELLOW.value}Welcome to the Query Routing System. Type 'exit' to quit.{ANSIColor.RESET.value}")
        print(f"{ANSIColor.CYAN.value}Using output folder: {settings.output_folder}{ANSIColor.RESET.value}")

        # Check if db_content.txt exists and is not empty
        db_content = self.load_db_content()
        if not db_content:
            print(f"{ANSIColor.PINK.value}Error: Cannot start the Query Routing System. Please upload and process documents first.{ANSIColor.RESET.value}")
            return

        while True:
            user_input = input(f"\n{ANSIColor.YELLOW.value}Enter your query: {ANSIColor.RESET.value}").strip()

            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using the Query Routing System. Goodbye!{ANSIColor.RESET.value}")
                break

            if not user_input:
                print(f"{ANSIColor.PINK.value}Please enter a valid query.{ANSIColor.RESET.value}")
                continue

            print(f"{ANSIColor.CYAN.value}Evaluating your query...{ANSIColor.RESET.value}")
            evaluation = self.evaluate_query(user_input)

            self.route_query(user_input, evaluation)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        from src.api_model import configure_api  # Updated import
        client = configure_api(api_type)
        route_query = RouteQuery(api_type, client)
        route_query.run()
    else:
        print("Error: No API type provided.")
        print("Usage: python src/route_query.py <api_type>")  # Updated usage instruction
        print("Available API types: ollama, llama")
        sys.exit(1)
