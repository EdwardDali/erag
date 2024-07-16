from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import success, info, warning, error, highlight, user_input, llm_response

class Talk2Model:
    def __init__(self, erag_api: EragAPI, model: str):
        self.erag_api = erag_api
        self.model = model

    def run(self):
        print(info(f"EragAPI initialized with {self.erag_api.api_type} backend."))
        print(info(f"Talking to {self.model} using EragAPI (backed by {self.erag_api.api_type}). Type 'exit' to end the conversation."))
        
        while True:
            user_prompt = input(user_input("You: "))
            if user_prompt.lower() == 'exit':
                print(success("Thank you for using Talk2Model. Goodbye!"))
                break
            
            response = self.get_model_response(user_prompt)
            print(llm_response(f"Model: {response}"))

    def get_model_response(self, user_prompt):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_prompt}
            ]
            response = self.erag_api.chat(messages, temperature=settings.temperature)
            return response
        except Exception as e:
            return error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(error("Usage: python src/talk2model.py <api_type> <model>"))
        sys.exit(1)
    
    api_type = sys.argv[1]
    model = sys.argv[2]
    erag_api = EragAPI(api_type)
    talk2model = Talk2Model(erag_api, model)
    talk2model.run()
