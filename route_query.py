import sys
import logging
from openai import OpenAI
from enum import Enum
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

    @staticmethod
    def configure_api(api_type: str) -> OpenAI:
        if api_type == "ollama":
            return OpenAI(base_url='http://localhost:11434/v1', api_key=settings.ollama_model)
        elif api_type == "llama":
            return OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        else:
            raise ValueError("Invalid API type")

    def load_db_content(self):
        logging.info("Loading database content...")
        try:
            with open('db_content.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            logging.info("Successfully loaded database content from db_content.txt")
            logging.info(f"Loaded db_content (first 100 characters): {content[:100]}...")
            return content
        except FileNotFoundError:
            logging.error("db_content.txt not found")
            return ""

    def evaluate_query(self, query: str) -> dict:
        logging.info(f"Evaluating query: {query}")
        
        db_content = self.load_db_content()
        
        system_message = """You are an intelligent query router. Your task is to analyze the given query and the provided database content summary (which is a table of contents) to determine the most appropriate way to handle the query. Base your decision ONLY on the information provided in the query and the database content summary.

        Evaluate the following:
        1. Relevance: Does the database content summary indicate that relevant information is available for the query? (high/low)
        2. Complexity: Is the query asking for a simple answer or a deep dive? (simple/deep)

        Based on your evaluation, recommend one of the following options:
        A. talk2doc (for relevant content and simple queries)
        B. create_knol (for relevant content and deep dive queries)
        C. web_rag (for non-relevant content and simple queries)
        D. web_sum (for non-relevant content and deep dive queries)

        Format your response as follows:
        Query: [Repeat the exact query here]
        Relevance: [high/low]
        Complexity: [simple/deep]
        Recommendation: [A/B/C/D]
        Explanation: [Your detailed explanation here]

        Remember: Only use the information provided in the query and database content summary for your decision."""

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
            
            if parsed_response['query'].lower() != query.lower():
                logging.warning(f"LLM response query '{parsed_response['query']}' doesn't match the input query '{query}'. Using default routing.")
                return {
                    "query": query,
                    "recommendation": "C",
                    "relevance": "low",
                    "complexity": "simple",
                    "explanation": "Default routing due to LLM response mismatch."
                }
            
            return parsed_response
        except Exception as e:
            logging.error(f"Error in LLM response: {str(e)}. Using default routing.")
            return {
                "query": query,
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
                if key == 'query':
                    evaluation['query'] = value
                elif key == 'relevance':
                    evaluation['relevance'] = 'high' if 'high' in value.lower() else 'low'
                elif key == 'complexity':
                    evaluation['complexity'] = 'deep' if 'deep' in value.lower() else 'simple'
                elif key == 'recommendation':
                    evaluation['recommendation'] = value.upper() if value.upper() in ['A', 'B', 'C', 'D'] else 'C'
                elif key == 'explanation':
                    evaluation['explanation'] = value

        for key in ['query', 'relevance', 'complexity', 'recommendation', 'explanation']:
            if key not in evaluation:
                evaluation[key] = 'N/A'

        return evaluation

    def load_component(self, component_name: str):
        logging.info(f"Loading component: {component_name}")
        if component_name == 'talk2doc':
            from talk2doc import RAGSystem
            return RAGSystem(self.api_type)
        elif component_name == 'create_knol':
            from create_knol import KnolCreator
            return KnolCreator(self.api_type)
        elif component_name == 'web_rag':
            from web_rag import WebRAG
            return WebRAG(self.api_type)
        elif component_name == 'web_sum':
            from web_sum import WebSum
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
        route_query = RouteQuery(api_type)
        route_query.run()
    else:
        print("Error: No API type provided.")
        print("Usage: python route_query.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
