import sys
import logging
import time
from openai import OpenAI
from enum import Enum
from talk2doc import RAGSystem
from create_knol import KnolCreator
from web_rag import WebRAG
from web_sum import WebSum
from settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ANSIColor(Enum):
    PINK = '\033[95m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    NEON_GREEN = '\033[92m'
    RESET = '\033[0m'

class RouteQuery:
    def __init__(self, api_type: str):
        self.api_type = api_type
        self.client = self.configure_api(api_type)
        self.current_system = None
        self.talk2doc = RAGSystem(api_type)
        self.create_knol = KnolCreator(api_type)
        self.web_rag = WebRAG(api_type)
        self.web_sum = WebSum(api_type)

    @staticmethod
    def configure_api(api_type: str) -> OpenAI:
        if api_type == "ollama":
            return OpenAI(base_url='http://localhost:11434/v1', api_key=settings.ollama_model)
        elif api_type == "llama":
            return OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        else:
            raise ValueError("Invalid API type")

    def load_db_content(self):
        try:
            with open(settings.db_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"db_content.txt not found at {settings.db_file_path}")
            return ""

    def evaluate_query(self, query: str, db_content: str) -> dict:
        system_message = """You are an intelligent query router. Your task is to analyze the given query and the provided database content summary to determine the most appropriate way to handle the query. The database content summary contains a table of contents for documents accessible by the system. Evaluate the following:

        1. Relevance: Does the database content summary indicate that relevant information is available for the query? (high/low)
        2. Complexity: Is the query asking for a simple answer or a deep dive? (simple/deep)

        Based on your evaluation, recommend one of the following options:
        A. talk2doc (for relevant content and simple queries)
        B. create_knol (for relevant content and deep dive queries)
        C. web_rag (for non-relevant content and simple queries)
        D. web_sum (for non-relevant content and deep dive queries)

        Provide your recommendation in the following format:
        Relevance: [high/low]
        Complexity: [simple/deep]
        Recommendation: [A/B/C/D]
        Explanation: [brief explanation of your choice]"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Query: {query}\n\nDatabase Content Summary:\n{db_content}\n\nPlease evaluate the query and the database content summary, then provide your recommendation."}
        ]

        try:
            response = self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=messages,
                temperature=settings.temperature
            )
            return self.parse_evaluation(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error in LLM response: {str(e)}. Using default routing.")
            return {
                "recommendation": "C",
                "relevance": "low",
                "complexity": "simple",
                "explanation": "Default routing due to error."
            }

    def parse_evaluation(self, response: str) -> dict:
        lines = response.split('\n')
        evaluation = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip().lower()
                if key == 'relevance':
                    evaluation['relevance'] = 'high' if 'high' in value else 'low'
                elif key == 'complexity':
                    evaluation['complexity'] = 'deep' if 'deep' in value else 'simple'
                elif key == 'recommendation':
                    evaluation['recommendation'] = value.upper() if value.upper() in ['A', 'B', 'C', 'D'] else 'C'
                elif key == 'explanation':
                    evaluation['explanation'] = value

        # Ensure all keys are present
        for key in ['relevance', 'complexity', 'recommendation', 'explanation']:
            if key not in evaluation:
                evaluation[key] = 'N/A'

        return evaluation

    def execute_query(self, query: str, recommendation: str):
        print(f"\n{ANSIColor.CYAN.value}Executing query with {self.current_system}: {query}{ANSIColor.RESET.value}")
        
        try:
            if self.current_system == "talk2doc":
                response = self.talk2doc.ollama_chat(query, "You are a helpful assistant. Please respond to the user's query based on the available context.")
            elif self.current_system == "create_knol":
                self.create_knol.run_knol_creator()
                response = "Knol creation process completed. Please check the generated files for results."
            elif self.current_system == "web_rag":
                response = self.web_rag.search_and_process(query)
            elif self.current_system == "web_sum":
                response = self.web_sum.search_and_process(query)
            else:
                response = "Error: Invalid system selected."
        except Exception as e:
            logging.error(f"Error executing query: {str(e)}")
            response = f"An error occurred while processing your query: {str(e)}"

        print(f"\n{ANSIColor.NEON_GREEN.value}Response:{ANSIColor.RESET.value}")
        print(response)

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
        time.sleep(1)  # Simulate loading time
        print(f"{ANSIColor.NEON_GREEN.value}System loaded: {system_to_load}{ANSIColor.RESET.value}")
        
        self.current_system = system_to_load
        self.execute_query(query, evaluation['recommendation'])

    def run(self):
        print(f"{ANSIColor.YELLOW.value}Welcome to the Query Routing System. Type 'exit' to quit.{ANSIColor.RESET.value}")

        while True:
            user_input = input(f"\n{ANSIColor.YELLOW.value}Enter your query: {ANSIColor.RESET.value}").strip()

            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using the Query Routing System. Goodbye!{ANSIColor.RESET.value}")
                break

            if not user_input:
                print(f"{ANSIColor.PINK.value}Please enter a valid query.{ANSIColor.RESET.value}")
                continue

            print(f"{ANSIColor.CYAN.value}Evaluating your query...{ANSIColor.RESET.value}")
            db_content = self.load_db_content()
            evaluation = self.evaluate_query(user_input, db_content)

            self.route_query(user_input, evaluation)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        route_query = RouteQuery(api_type)
        route_query.run()
    else:
        print("Error: No API type provided.")
        print("Usage: python route_query.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
