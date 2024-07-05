from api_model import configure_api
from settings import settings
from talk2doc import ANSIColor

class Talk2Model:
    def __init__(self, api_type, model):
        self.api_type = api_type
        self.model = model
        self.client = configure_api(api_type)

    def run(self):
        print(f"{ANSIColor.YELLOW.value}Talking to {self.model} using {self.api_type} API. Type 'exit' to end the conversation.{ANSIColor.RESET.value}")
        
        while True:
            user_input = input(f"{ANSIColor.YELLOW.value}You: {ANSIColor.RESET.value}").strip()
            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using Talk2Model. Goodbye!{ANSIColor.RESET.value}")
                break
            
            response = self.get_model_response(user_input)
            print(f"{ANSIColor.NEON_GREEN.value}Model: {response}{ANSIColor.RESET.value}")

    def get_model_response(self, user_input):
        try:
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
        print("Usage: python talk2model.py <api_type> <model>")
        sys.exit(1)
    
    api_type = sys.argv[1]
    model = sys.argv[2]
    talk2model = Talk2Model(api_type, model)
    talk2model.run()
