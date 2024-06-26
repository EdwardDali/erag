import sys
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging
from collections import deque
from embeddings_utils import load_embeddings_and_data
from enum import Enum
from openai import OpenAI
import networkx as nx
import json
from search_utils import SearchUtils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ANSIColor(Enum):
    PINK = '\033[95m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    NEON_GREEN = '\033[92m'
    RESET = '\033[0m'

class RAGSystem:
    def __init__(self, api_type: str):
        self.client = self.configure_api(api_type)
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.db_embeddings, _, _ = self.load_embeddings()
        self.db_content = self.load_db_content()
        self.conversation_history: List[Dict[str, str]] = []
        self.new_entries: List[str] = []
        self.conversation_context: deque = deque(maxlen=self.CONVERSATION_CONTEXT_SIZE * 2)
        self.knowledge_graph = self.load_knowledge_graph()
        self.search_utils = SearchUtils(self.model, self.db_embeddings, self.db_content, self.knowledge_graph)

    # Class variables (settings)
    MAX_HISTORY_LENGTH = 5
    EMBEDDINGS_FILE = "db_embeddings.pt"
    DB_FILE = "db.txt"
    MODEL_NAME = "all-MiniLM-L6-v2"
    OLLAMA_MODEL = "phi3:instruct"
    UPDATE_THRESHOLD = 10
    CONVERSATION_CONTEXT_SIZE = 3
    KNOWLEDGE_GRAPH_FILE = "knowledge_graph.json"
    RESULTS_FILE = "results.txt"
    TEMPERATURE = 0.1

    @staticmethod
    def configure_api(api_type: str) -> OpenAI:
        if api_type == "ollama":
            return OpenAI(base_url='http://localhost:11434/v1', api_key='phi3:instruct')
        elif api_type == "llama":
            return OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        else:
            raise ValueError("Invalid API type")

    def load_embeddings(self):
        embeddings, indexes, content = load_embeddings_and_data(self.EMBEDDINGS_FILE)
        if embeddings is None or indexes is None or content is None:
            logging.error(f"Failed to load data from {self.EMBEDDINGS_FILE}. Make sure the file exists and is properly formatted.")
            return torch.tensor([]), torch.tensor([]), []
        return embeddings, indexes, content
    
    def load_db_content(self):
        if os.path.exists(self.DB_FILE):
            with open(self.DB_FILE, "r", encoding='utf-8') as db_file:
                return db_file.readlines()
        return []

    def load_knowledge_graph(self):
        try:
            if not os.path.exists(self.KNOWLEDGE_GRAPH_FILE):
                logging.warning(f"Knowledge graph file {self.KNOWLEDGE_GRAPH_FILE} not found.")
                return nx.Graph()

            with open(self.KNOWLEDGE_GRAPH_FILE, 'r') as file:
                graph_data = json.load(file)
            
            G = nx.node_link_graph(graph_data)
            logging.info(f"Successfully loaded knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            return G
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse knowledge graph JSON: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load knowledge graph: {str(e)}")
        
        return nx.Graph()

    def ollama_chat(self, user_input: str, system_message: str) -> str:
        lexical_context, semantic_context, graph_context, text_context = self.search_utils.get_relevant_context(user_input, list(self.conversation_context))
        
        lexical_str = "\n".join(lexical_context)
        semantic_str = "\n".join(semantic_context)
        graph_str = "\n".join(graph_context)
        text_str = "\n".join(text_context)

        combined_context = f"""Conversation Context:\n{' '.join(self.conversation_context)}

Lexical Search Results:
{lexical_str}

Semantic Search Results:
{semantic_str}

Knowledge Graph Context:
{graph_str}

Text Search Results:
{text_str}"""

        logging.info(f"Combined context pulled: {combined_context[:200]}...")

        messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history,
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {user_input}\n\nPlease prioritize the Conversation Context when answering, followed by the most relevant information from either the lexical, semantic, knowledge graph, or text search results. If none of the provided context is relevant, you can answer based on your general knowledge."}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.OLLAMA_MODEL,
                messages=messages,
                temperature=self.TEMPERATURE
            ).choices[0].message.content

            # Save debug results
            self.save_debug_results(user_input, lexical_context, semantic_context, graph_context, text_context, response)

            return response
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your request."

    def save_debug_results(self, user_input: str, lexical_context: List[str], 
                           semantic_context: List[str], 
                           graph_context: List[str], 
                           text_context: List[str],
                           response: str):
        with open(self.RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"User Input: {user_input}\n\n")
            f.write("Lexical Search Results:\n")
            for i, content in enumerate(lexical_context, 1):
                f.write(f"{i}. {content}\n")
            f.write("\nSemantic Search Results:\n")
            for i, content in enumerate(semantic_context, 1):
                f.write(f"{i}. {content}\n")
            f.write("\nGraph Context Results:\n")
            for i, content in enumerate(graph_context, 1):
                f.write(f"{i}. {content}\n")
            f.write("\nText Search Results:\n")
            for i, content in enumerate(text_context, 1):
                f.write(f"{i}. {content}\n")
            f.write("\nCombined Response:\n")
            f.write(f"{response}\n")
            f.write("\n" + "="*50 + "\n\n")

    def run(self):
        system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Prioritize the most recent conversation context when answering questions, but also consider other relevant information if necessary. If the given context doesn't provide a suitable answer, rely on your general knowledge."

        print(f"{ANSIColor.YELLOW.value}Welcome to the RAG system. Type 'exit' to quit or 'clear' to clear conversation history.{ANSIColor.RESET.value}")

        while True:
            user_input = input(f"{ANSIColor.YELLOW.value}Ask a question about your documents: {ANSIColor.RESET.value}").strip()

            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using the RAG system. Goodbye!{ANSIColor.RESET.value}")
                self.update_embeddings()
                break
            elif user_input.lower() == 'clear':
                self.conversation_history.clear()
                self.conversation_context.clear()
                print(f"{ANSIColor.CYAN.value}Conversation history and context cleared.{ANSIColor.RESET.value}")
                continue

            if not user_input:
                print(f"{ANSIColor.PINK.value}Please enter a valid question.{ANSIColor.RESET.value}")
                continue

            response = self.ollama_chat(user_input, system_message)
            print(f"{ANSIColor.NEON_GREEN.value}Response: \n\n{response}{ANSIColor.RESET.value}")

            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            self.update_conversation_context(user_input, response)

            if len(self.conversation_history) > self.MAX_HISTORY_LENGTH * 2:
                self.conversation_history = self.conversation_history[-self.MAX_HISTORY_LENGTH * 2:]

            self.append_to_db(self.conversation_history[-2:])

    def update_conversation_context(self, user_input: str, assistant_response: str):
        self.conversation_context.append(user_input)
        self.conversation_context.append(assistant_response)

    def append_to_db(self, new_entries: List[Dict[str, str]]):
        with open(self.DB_FILE, "a", encoding="utf-8") as db_file:
            for entry in new_entries:
                db_file.write(f"{entry['role']}: {entry['content']}\n")
        self.new_entries.extend([entry['content'] for entry in new_entries])
        
        # Also update db_content
        self.db_content.extend([f"{entry['role']}: {entry['content']}" for entry in new_entries])

    def update_embeddings(self):
        if len(self.new_entries) >= self.UPDATE_THRESHOLD:
            logging.info("Updating embeddings...")
            new_embeddings = self.model.encode(self.new_entries, convert_to_tensor=True, show_progress_bar=False)
            self.db_embeddings = torch.cat([self.db_embeddings, new_embeddings], dim=0)
            
            # Save updated embeddings
            data_to_save = {
                'embeddings': self.db_embeddings,
                'indexes': torch.arange(len(self.db_content)),
                'content': self.db_content
            }
            torch.save(data_to_save, self.EMBEDDINGS_FILE)
            
            self.new_entries.clear()
            logging.info("Embeddings updated successfully.")

# Setter functions for updating settings
def set_max_history_length(value: int):
    RAGSystem.MAX_HISTORY_LENGTH = value

def set_conversation_context_size(value: int):
    RAGSystem.CONVERSATION_CONTEXT_SIZE = value

def set_update_threshold(value: int):
    RAGSystem.UPDATE_THRESHOLD = value

def set_ollama_model(value: str):
    RAGSystem.OLLAMA_MODEL = value

def set_temperature(value: float):
    RAGSystem.TEMPERATURE = value

def set_embeddings_file(value: str):
    RAGSystem.EMBEDDINGS_FILE = value

def set_db_file(value: str):
    RAGSystem.DB_FILE = value

def set_model_name(value: str):
    RAGSystem.MODEL_NAME = value

def set_knowledge_graph_file(value: str):
    RAGSystem.KNOWLEDGE_GRAPH_FILE = value

def set_results_file(value: str):
    RAGSystem.RESULTS_FILE = value

def main(api_type: str):
    rag_system = RAGSystem(api_type)
    rag_system.run()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        main(api_type)
    else:
        print("Error: No API type provided.")
        print("Usage: python run_model.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
