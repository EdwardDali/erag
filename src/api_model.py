# Standard library imports
import os
import subprocess

# Third-party imports
import google.generativeai as genai
import requests
import torch
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Local imports
from src.look_and_feel import error, success, warning, info
from src.settings import settings

load_dotenv()

class EragAPI:
    def __init__(self, api_type, model=None, embedding_class=None, embedding_model=None, reranker_model=None):
        self.api_type = api_type
        self.model = model or settings.get_default_model(api_type)
        self.embedding_class = embedding_class or settings.get_default_embedding_class()
        self.embedding_model = embedding_model or settings.get_default_embedding_model()
        self.reranker_model = reranker_model or settings.reranker_model

        clients = {
            "ollama": lambda: OpenAI(base_url='http://localhost:11434/v1', api_key='ollama'),
            "llama": LlamaClient,
            "groq": lambda: GroqClient(self.model),
            "gemini": lambda: GeminiClient(self.model)
        }
        self.client = clients.get(api_type, lambda: ValueError(error("Invalid API type")))()

        embedding_clients = {
            "ollama": lambda: OpenAI(base_url='http://localhost:11434/v1', api_key='ollama'),
            "sentence_transformers": lambda: SentenceTransformer(self.embedding_model)
        }
        self.embedding_client = embedding_clients.get(self.embedding_class, lambda: ValueError(f"Invalid embedding class: {self.embedding_class}"))()

    def _encode(self, texts):
        print(info(f"Starting embedding process for {len(texts)} texts"))
        if self.embedding_class == "ollama":
            embeddings = [self.embedding_client.embeddings.create(model=self.embedding_model, input=text).data[0].embedding for text in texts]
            return torch.tensor(embeddings)
        return self.embedding_client.encode(texts)

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        try:
            if self.api_type == "gemini":
                model = genai.GenerativeModel(self.model)
                
                # Format messages for Gemini API
                formatted_messages = []
                for message in messages:
                    if message['role'] == 'system':
                        # Prepend system messages to the user's first message
                        formatted_messages.append({"role": "user", "parts": [{"text": f"System: {message['content']}"}]})
                    elif message['role'] in ['user', 'assistant']:
                        formatted_messages.append({"role": message['role'], "parts": [{"text": message['content']}]})
                
                # If there are no user messages, add an empty one to avoid API errors
                if not any(msg['role'] == 'user' for msg in formatted_messages):
                    formatted_messages.append({"role": "user", "parts": [{"text": " "}]})
                
                response = model.generate_content(
                    formatted_messages,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                    stream=stream
                )
                return response if stream else response.text
            elif self.api_type in ["llama", "groq"]:
                return self.client.chat(messages, temperature=temperature, max_tokens=max_tokens, stream=stream)
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=stream
            )
            return response if stream else response.choices[0].message.content
        except Exception as e:
            return error(f"An error occurred: {str(e)}")

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        try:
            if self.api_type in ["llama", "groq", "gemini"]:
                return self.client.complete(prompt, temperature=temperature, max_tokens=max_tokens, stream=stream)
            response = self.client.completions.create(
                model=self.model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, stream=stream
            )
            return response if stream else response.choices[0].text
        except Exception as e:
            return error(f"An error occurred: {str(e)}")

def update_settings(settings, api_type, model):
    setting_map = {"ollama": "ollama_model", "llama": "llama_model", "groq": "groq_model", "gemini": "gemini_model"}
    if api_type in setting_map:
        settings.update_setting(setting_map[api_type], model)
        settings.apply_settings()
        print(success(f"Settings updated. Using {model} with {api_type} backend."))
    else:
        print(error(f"Unknown API type: {api_type}"))

class LlamaClient:
    def __init__(self, base_url='http://localhost:8080/v1'):
        self.base_url = base_url

    def _request(self, endpoint, data):
        response = requests.post(f"{self.base_url}/{endpoint}", json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content' if endpoint == 'chat/completions' else 'text']
        raise Exception(error(f"Error from llama.cpp server: {response.status_code} - {response.text}"))

    def chat(self, messages, temperature=0.7, max_tokens=None):
        return self._request('chat/completions', {"messages": messages, "temperature": temperature, "max_tokens": max_tokens})

    def complete(self, prompt, temperature=0.7, max_tokens=None):
        return self._request('completions', {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens})

    def get_models(self):
        try:
            return [model['id'] for model in requests.get(f"{self.base_url}/models").json()['data']]
        except requests.RequestException as e:
            raise Exception(error(f"Error connecting to llama.cpp server: {e}"))

class GroqClient:
    def __init__(self, model=None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY") or ValueError(error("GROQ_API_KEY not found in .env file")))
        self.model = model or self.get_default_model()

    def get_default_model(self):
        try:
            return self.client.models.list().data[0].id
        except Exception:
            return None

    def _request(self, method, **kwargs):
        try:
            return method(model=self.model, **kwargs)
        except Exception as e:
            raise Exception(error(f"Error from Groq API: {str(e)}"))

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        completion = self._request(self.client.chat.completions.create, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=stream)
        return completion if stream else completion.choices[0].message.content

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        completion = self._request(self.client.completions.create, prompt=prompt, temperature=temperature, max_tokens=max_tokens, stream=stream)
        return completion if stream else completion.choices[0].text

    def stream_chat(self, messages, temperature=1, max_tokens=1024):
        for chunk in self._request(self.client.chat.completions.create, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True):
            yield chunk.choices[0].delta.content or ""

class GeminiClient:
    def __init__(self, model=None):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY") or ValueError(error("GEMINI_API_KEY not found in .env file")))
        self.model = model or self.get_default_model()

    def get_default_model(self):
        try:
            return next((model.name for model in genai.list_models() if 'generateContent' in model.supported_generation_methods), None)
        except Exception:
            return None

    def _request(self, messages=None, prompt=None, temperature=0.7, max_tokens=None, stream=False):
        model = genai.GenerativeModel(self.model)
        config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        
        if messages:
            chat = model.start_chat(history=[])
            for message in messages:
                if message['role'] == 'user':
                    response = chat.send_message(message['content'], generation_config=config)
                    return self._stream_response(response) if stream else response.text
            return error("No user message found in the conversation.")
        
        response = model.generate_content(prompt, generation_config=config)
        return self._stream_response(response) if stream else response.text

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        return self._request(messages=messages, temperature=temperature, max_tokens=max_tokens, stream=stream)

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        return self._request(prompt=prompt, temperature=temperature, max_tokens=max_tokens, stream=stream)

    @staticmethod
    def _stream_response(response):
        for chunk in response:
            yield chunk.text

def get_available_models(api_type, server_manager=None):
    model_fetchers = {
        "ollama": lambda: [model.split()[0] for model in subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout.strip().split('\n')[1:] if model.split()[0] not in ['failed', 'NAME']],
        "llama": lambda: server_manager.get_gguf_models() if server_manager else [],
        "groq": lambda: [model.id for model in Groq(api_key=os.getenv("GROQ_API_KEY")).models.list().data],
        "gemini": lambda: [model.name for model in genai.list_models() if 'generateContent' in model.supported_generation_methods]
    }
    
    try:
        return model_fetchers.get(api_type, lambda: [])()
    except Exception as e:
        print(error(f"Error fetching models for {api_type}: {str(e)}"))
        return []

def create_erag_api(api_type, model=None, embedding_class=None, embedding_model=None, reranker_model=None):
    embedding_class = embedding_class or settings.get_default_embedding_class()
    embedding_model = embedding_model or settings.get_default_embedding_model(embedding_class)
    reranker_model = reranker_model or settings.reranker_model
    return EragAPI(api_type, model, embedding_class, embedding_model, reranker_model)