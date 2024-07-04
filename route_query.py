import sys
import logging
from talk2doc import RAGSystem, ANSIColor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RouteQuery:
    def __init__(self, api_type: str):
        self.rag_system = RAGSystem(api_type)

    def route_query(self, query: str) -> str:
        system_message = """You are an intelligent query router. Your task is to analyze the given query and determine the most appropriate way to handle it. You have the following options:

        1. Web Search: For queries that require up-to-date information or general knowledge not specific to any domain.
        2. Document Search: For queries that are likely to be answered by searching through existing documents in the system.
        3. Knowledge Graph: For queries that involve relationships between entities or concepts.
        4. Calculation: For queries that require mathematical calculations or data analysis.
        5. Code Generation: For queries that ask for code snippets or programming-related information.

        Provide your recommendation in the following format:
        Recommendation: [chosen option]
        Explanation: [brief explanation of why you chose this option]
        Next steps: [suggested actions or modifications to the query to best utilize the chosen option]"""

        user_input = f"Route the following query: {query}"

        response = self.rag_system.ollama_chat(user_input, system_message)
        return response

    def run(self):
        print(f"{ANSIColor.YELLOW.value}Welcome to the Query Routing System. Type 'exit' to quit.{ANSIColor.RESET.value}")

        while True:
            user_input = input(f"{ANSIColor.YELLOW.value}Enter your query: {ANSIColor.RESET.value}").strip()

            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using the Query Routing System. Goodbye!{ANSIColor.RESET.value}")
                break

            if not user_input:
                print(f"{ANSIColor.PINK.value}Please enter a valid query.{ANSIColor.RESET.value}")
                continue

            print(f"{ANSIColor.CYAN.value}Routing your query...{ANSIColor.RESET.value}")
            result = self.route_query(user_input)

            print(f"\n{ANSIColor.NEON_GREEN.value}Query Routing Result:{ANSIColor.RESET.value}")
            print(result)

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
