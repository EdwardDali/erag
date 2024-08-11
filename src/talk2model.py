# Standard library imports
import sys
from typing import List, Dict

# Local imports
from src.api_model import create_erag_api, EragAPI
from src.look_and_feel import error, info, llm_response, success, user_input
from src.settings import settings

class Talk2Model:
    def __init__(self, erag_api: EragAPI):
        self.erag_api = erag_api
        self.model = erag_api.model

    def run(self):
        print(info(f"EragAPI initialized with {self.erag_api.api_type} backend."))
        print(info(f"Talking to {self.model} using EragAPI. Type 'exit' to end the conversation."))
        
        while True:
            user_prompt = input(user_input("You: "))
            if user_prompt.lower() == 'exit':
                print(success("Thank you for using Talk2Model. Goodbye!"))
                break
            
            response = self.get_model_response(user_prompt)
            print(llm_response(f"Model: {response}"))

    def get_model_response(self, user_prompt: str) -> str:
        try:
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_prompt}
            ]
            return self.erag_api.chat(messages, temperature=settings.temperature)
        except Exception as e:
            return error(f"An error occurred: {str(e)}")

def main():
    if len(sys.argv) != 3:
        print(error("Usage: python src/talk2model.py <api_type> <model>"))
        sys.exit(1)
    
    api_type = sys.argv[1]
    model = sys.argv[2]
    talk2model = Talk2Model(api_type, model)
    talk2model.run()

if __name__ == "__main__":
    main()