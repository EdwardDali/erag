import subprocess
from openai import OpenAI
from src.settings import settings
import requests

def get_available_models(api_type):
    if api_type == "ollama":
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            models = result.stdout.strip().split('\n')[1:]  # Skip the header
            # Filter out 'failed' and 'NAME' entries
            return [model.split()[0] for model in models if model.split()[0] not in ['failed', 'NAME']]
        except subprocess.CalledProcessError:
            print("Error running 'ollama list' command")
            return []
    elif api_type == "llama":
        try:
            response = requests.get("http://localhost:8080/v1/models")
            if response.status_code == 200:
                models = response.json()['data']
                return [model['id'] for model in models]
            else:
                print(f"Error fetching models from llama.cpp server: {response.status_code}")
                return []
        except requests.RequestException as e:
            print(f"Error connecting to llama.cpp server: {e}")
            return []
    else:
        return []

def update_settings(settings, api_type, model):
    if api_type == "ollama":
        settings.update_setting("ollama_model", model)
    elif api_type == "llama":
        settings.update_setting("llama_model", model)
    settings.apply_settings()
    print(f"Settings updated. Using {model} with {api_type} API.")

def configure_api(api_type: str) -> OpenAI:
    if api_type == "ollama":
        return OpenAI(base_url='http://localhost:11434/v1', api_key=settings.ollama_model)
    elif api_type == "llama":
        return OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
    else:
        raise ValueError("Invalid API type")

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
            raise Exception(f"Error from llama.cpp server: {response.status_code} - {response.text}")

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
            raise Exception(f"Error from llama.cpp server: {response.status_code} - {response.text}")
