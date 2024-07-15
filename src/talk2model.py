from src.api_model import configure_api, LlamaClient
from src.settings import settings
from src.look_and_feel import success, info, warning, error, highlight, user_input, llm_response

class Talk2Model:
    def __init__(self, api_type, model):
        self.api_type = api_type
        self.model = model
        if api_type == "llama":
            self.client = LlamaClient()
        else:
            self.client = configure_api(api_type)

    def run(self):
        print(info(f"Talking to {self.model} using {self.api_type} API. Type 'exit' to end the conversation."))
        
        while True:
            user_prompt = input(user_input("You: "))
            if user_prompt.lower() == 'exit':
                print(success("Thank you for using Talk2Model. Goodbye!"))
                break
            
            response = self.get_model_response(user_prompt)
            print(llm_response(f"Model: {response}"))

    def get_model_response(self, user_prompt):
        try:
            if self.api_type == "llama":
                response = self.client.chat([
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": user_prompt}
                ], temperature=settings.temperature)
            else:
                response = self.client.chat.completions.create(
                    model=settings.ollama_model if self.api_type == "ollama" else settings.llama_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=settings.temperature
                ).choices[0].message.content
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
    talk2model = Talk2Model(api_type, model)
    talk2model.run()
