from src.api_model import configure_api, LlamaClient
from src.settings import settings
from src.color_scheme import Colors, colorize
import colorama

# Initialize colorama
colorama.init(autoreset=True)

class Talk2Model:
    def __init__(self, api_type, model):
        self.api_type = api_type
        self.model = model
        if api_type == "llama":
            self.client = LlamaClient()
        else:
            self.client = configure_api(api_type)

    def run(self):
        print(colorize(f"Talking to {self.model} using {self.api_type} API. Type 'exit' to end the conversation.", Colors.INFO))
        
        while True:
            user_input = input(colorize("You: ", Colors.INFO)).strip()
            if user_input.lower() == 'exit':
                print(colorize("Thank you for using Talk2Model. Goodbye!", Colors.SUCCESS))
                break
            
            response = self.get_model_response(user_input)
            print(colorize("Model: ", Colors.SUCCESS) + response)

    def get_model_response(self, user_input):
        try:
            if self.api_type == "llama":
                response = self.client.chat([
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": user_input}
                ], temperature=settings.temperature)
            else:
                response = self.client.chat.completions.create(
                    model=settings.ollama_model if self.api_type == "ollama" else settings.llama_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=settings.temperature
                ).choices[0].message.content
            return response
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(colorize("Usage: python src/talk2model.py <api_type> <model>", Colors.ERROR))
        sys.exit(1)
    
    api_type = sys.argv[1]
    model = sys.argv[2]
    talk2model = Talk2Model(api_type, model)
    talk2model.run()
