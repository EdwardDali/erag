import subprocess
from openai import OpenAI
from src.settings import settings
import requests
from src.look_and_feel import error, success, warning, info
import os
from dotenv import load_dotenv
from groq.types import Model, ModelDeleted, ModelListResponse
from groq import Groq
from dotenv import load_dotenv
import vertexai
import google.generativeai as genai
import numpy as np
import logging
import warnings
from urllib3.exceptions import InsecureRequestWarning
from sentence_transformers import SentenceTransformer


# Load environment variables from .env file
load_dotenv()

class EragAPI:
    def __init__(self, api_type, model=None, embedding_class=None, embedding_model=None):
        self.api_type = api_type
        self.model = model
        self.embedding_class = embedding_class or settings.get_default_embedding_class()
        self.embedding_model = embedding_model or settings.get_default_embedding_model()

        if api_type == "ollama":
            self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
            self.model = model or settings.ollama_model
        elif api_type == "llama":
            self.client = LlamaClient()
            self.model = model or settings.llama_model
        elif api_type == "groq":
            self.client = GroqClient(model)
            self.model = self.client.model
        elif api_type == "gemini":
            self.client = GeminiClient(model)
            self.model = self.client.model
        else:
            raise ValueError(error("Invalid API type"))
        
        # Embedding model initialization
        if self.embedding_class == "ollama":
            self.embedding_client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        elif self.embedding_class == "sentence_transformers":
            self.embedding_client = SentenceTransformer(self.embedding_model)
        else:
            raise ValueError(error(f"Invalid embedding class: {self.embedding_class}"))

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        try:
            if self.api_type == "llama":
                response = self.client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            elif self.api_type == "groq":
                response = self.client.chat(messages, temperature=temperature, max_tokens=max_tokens, stream=stream)
            elif self.api_type == "gemini":
                response = self.client.chat(messages, temperature=temperature, max_tokens=max_tokens, stream=stream)
            else:  # ollama
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
                if not stream:
                    response = response.choices[0].message.content
            return response
        except Exception as e:
            return error(f"An error occurred: {str(e)}")

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        try:
            if self.api_type == "llama":
                response = self.client.complete(prompt, temperature=temperature, max_tokens=max_tokens)
            elif self.api_type == "groq":
                response = self.client.complete(prompt, temperature=temperature, max_tokens=max_tokens, stream=stream)
            elif self.api_type == "gemini":
                response = self.client.complete(prompt, temperature=temperature, max_tokens=max_tokens, stream=stream)
            else:  # ollama
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
                if not stream:
                    response = response.choices[0].text
            return response
        except Exception as e:
            return error(f"An error occurred: {str(e)}")
        
    

    def encode(self, texts):
        if self.embedding_class == "ollama":
            return self._encode_ollama(texts)
        elif self.embedding_class == "sentence_transformers":
            return self._encode_sentence_transformers(texts)
        
    def _encode_ollama(self, texts):
        embeddings = []
        batch_size = len(texts)
        
        print(info(f"Starting embedding process for {batch_size} texts"))
        
        # Suppress HTTP request logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        
        for i, text in enumerate(texts, 1):
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            
            if i % 25 == 0 or i == batch_size:  # Print progress every 25 batches or at the end
                print(info(f"Processed {i}/{batch_size} texts"))
        
        print(info(f"Embedding process completed for {batch_size} texts"))
        
        # Reset logging levels
        logging.getLogger("httpx").setLevel(logging.NOTSET)
        
        return np.array(embeddings)
    
    def _encode_sentence_transformers(self, texts):
            print(info(f"Computing embeddings for {len(texts)} texts using Sentence Transformers model: {self.embedding_model}"))
            embeddings = self.embedding_client.encode(texts)
            print(info(f"Embeddings shape: {embeddings.shape}"))
            return embeddings

def update_settings(settings, api_type, model):
        if api_type == "ollama":
            settings.update_setting("ollama_model", model)
        elif api_type == "llama":
            settings.update_setting("llama_model", model)
        elif api_type == "groq":
            settings.update_setting("groq_model", model)
        elif api_type == "gemini":
            settings.update_setting("gemini_model", model)
        else:
            print(error(f"Unknown API type: {api_type}"))
            return

        settings.apply_settings()
        print(success(f"Settings updated. Using {model} with {api_type} backend."))

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

    def get_models(self):
        url = f"{self.base_url}/models"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                models = response.json()['data']
                return [model['id'] for model in models]
            else:
                raise Exception(error(f"Error fetching models: {response.status_code} - {response.text}"))
        except requests.RequestException as e:
            raise Exception(error(f"Error connecting to llama.cpp server: {e}"))

class GroqClient:
    def __init__(self, model=None):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(error("GROQ_API_KEY not found in .env file"))
        self.client = Groq()
        self.model = model or self.get_default_model()

    def get_default_model(self):
        try:
            models = self.client.models.list()
            return models.data[0].id if models.data else None
        except Exception:
            return None

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return completion  # Return the stream object
            else:
                return completion.choices[0].message.content
        except Exception as e:
            raise Exception(error(f"Error from Groq API: {str(e)}"))

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        try:
            completion = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return completion  # Return the stream object
            else:
                return completion.choices[0].text
        except Exception as e:
            raise Exception(error(f"Error from Groq API: {str(e)}"))

    def stream_chat(self, messages, temperature=1, max_tokens=1024):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in completion:
                yield chunk.choices[0].delta.content or ""
        except Exception as e:
            raise Exception(error(f"Error streaming from Groq API: {str(e)}"))

class GeminiClient:
    def __init__(self, model=None):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(error("GEMINI_API_KEY not found in .env file"))
        genai.configure(api_key=self.api_key)
        self.model = model or self.get_default_model()

    def get_default_model(self):
        try:
            models = genai.list_models()
            return next((model.name for model in models if 'generateContent' in model.supported_generation_methods), None)
        except Exception:
            return None

    def chat(self, messages, temperature=0.7, max_tokens=None, stream=False):
        try:
            model = genai.GenerativeModel(self.model)
            chat = model.start_chat(history=[])
            
            for message in messages:
                if message['role'] == 'user':
                    response = chat.send_message(message['content'], generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ))
                    
                    if stream:
                        return self._stream_response(response)
                    else:
                        return response.text
            
            return error("No user message found in the conversation.")
        except Exception as e:
            raise Exception(error(f"Error from Gemini API: {str(e)}"))

    def complete(self, prompt, temperature=0.7, max_tokens=None, stream=False):
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ))
            
            if stream:
                return self._stream_response(response)
            else:
                return response.text
        except Exception as e:
            raise Exception(error(f"Error from Gemini API: {str(e)}"))

    def _stream_response(self, response):
        for chunk in response:
            yield chunk.text

def get_available_models(api_type, server_manager=None):
    if api_type == "ollama":
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            models = result.stdout.strip().split('\n')[1:]  # Skip the header
            return [model.split()[0] for model in models if model.split()[0] not in ['failed', 'NAME']]
        except subprocess.CalledProcessError:
            print(error("Error running 'ollama list' command"))
            return []
    elif api_type == "llama":
        if server_manager:
            return server_manager.get_gguf_models()
        else:
            print(error("Server manager not provided for llama models"))
            return []
    elif api_type == "groq":
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print(error("GROQ_API_KEY not found in .env file"))
                return []
            
            client = Groq(api_key=api_key)
            models: ModelListResponse = client.models.list()
            
            return [model.id for model in models.data if isinstance(model, Model)]
        except Exception as e:
            print(error(f"Error fetching models from Groq API: {str(e)}"))
            return []
    elif api_type == "gemini":
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print(error("GEMINI_API_KEY not found in .env file"))
                return []
            
            genai.configure(api_key=api_key)
            models = genai.list_models()
            
            return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
        except Exception as e:
            print(error(f"Error fetching models from Gemini API: {str(e)}"))
            return []
    else:
        return []

# Factory function to create EragAPI instance
def create_erag_api(api_type, model=None, embedding_class=None, embedding_model=None):
    if model is None:
        model = settings.get_default_model(api_type)
    if embedding_class is None:
        embedding_class = settings.get_default_embedding_class()
    if embedding_model is None:
        embedding_model = settings.get_default_embedding_model()
    return EragAPI(api_type, model, embedding_class, embedding_model)
