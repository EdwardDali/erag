import subprocess
from openai import OpenAI
from src.settings import settings
import requests
from src.look_and_feel import error, success, warning, info

class EragAPI:
    def __init__(self, api_type):
        self.api_type = api_type
        if api_type == "ollama":
            self.client = OpenAI(base_url='http://localhost:11434/v1', api_key=settings.ollama_model)
        elif api_type == "llama":
            self.client = LlamaClient()
        else:
            raise ValueError(error("Invalid API type"))

    def chat(self, messages, temperature=0.7, max_tokens=None):
        try:
            if self.api_type == "llama":
                response = self.client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            else:
                response = self.client.chat.completions.create(
                    model=settings.ollama_model if self.api_type == "ollama" else settings.llama_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                ).choices[0].message.content
            return response
        except Exception as e:
            return error(f"An error occurred: {str(e)}")

    def complete(self, prompt, temperature=0.7, max_tokens=None):
        try:
            if self.api_type == "llama":
                response = self.client.complete(prompt, temperature=temperature, max_tokens=max_tokens)
            else:
                response = self.client.completions.create(
                    model=settings.ollama_model if self.api_type == "ollama" else settings.llama_model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                ).choices[0].text
            return response
        except Exception as e:
            return error(f"An error occurred: {str(e)}")

def get_available_models(api_type):
    if api_type == "ollama":
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            models = result.stdout.strip().split('\n')[1:]  # Skip the header
            # Filter out 'failed' and 'NAME' entries
            return [model.split()[0] for model in models if model.split()[0] not in ['failed', 'NAME']]
        except subprocess.CalledProcessError:
            print(error("Error running 'ollama list' command"))
            return []
    elif api_type == "llama":
        try:
            response = requests.get("http://localhost:8080/v1/models")
            if response.status_code == 200:
                models = response.json()['data']
                return [model['id'] for model in models]
            else:
                print(error(f"Error fetching models from llama.cpp server: {response.status_code}"))
                return []
        except requests.RequestException as e:
            print(error(f"Error connecting to llama.cpp server: {e}"))
            return []
    else:
        return []

def update_settings(settings, api_type, model):
    if api_type == "ollama":
        settings.update_setting("ollama_model", model)
    elif api_type == "llama":
        settings.update_setting("llama_model", model)
    settings.apply_settings()
    print(success(f"Settings updated. Using {model} with {api_type} API."))

class LlamaClient:
    def __init__(self, base_url='http://localhost:8080/v1'):
        self.base_url = base_url

    def chat(self, messages, temperature=0.7, max_tokens=None):
        url = f"{self.base_url}/chat/completions"
        data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(error(f"Error from llama.cpp server: {response.status_code} - {response.text}"))

    def complete(self, prompt, temperature=0.7, max_tokens=None):
        url = f"{self.base_url}/completions"
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['text']
        else:
            raise Exception(error(f"Error from llama.cpp server: {response.status_code} - {response.text}"))

# Factory function to create EragAPI instance
def create_erag_api(api_type):
    return EragAPI(api_type)
