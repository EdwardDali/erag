import sys
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Union, Tuple, Optional
import logging
import re
from collections import Counter
from embeddings_utils import load_or_compute_embeddings, compute_and_save_embeddings
from enum import Enum
from openai import OpenAI


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ANSIColor(Enum):
    PINK = '\033[95m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    NEON_GREEN = '\033[92m'
    RESET = '\033[0m'

class RAGSystem:
    MAX_HISTORY_LENGTH = 5
    EMBEDDINGS_FILE = "db_embeddings.pt"
    DB_FILE = "db.txt"
    MODEL_NAME = "all-MiniLM-L6-v2"
    OLLAMA_MODEL = "phi3:instruct"
    UPDATE_THRESHOLD = 10  # Number of new entries before updating embeddings

    def append_to_db(self, new_messages: List[Dict[str, str]]):
        with open(self.DB_FILE, "a", encoding='utf-8') as file:
            for msg in new_messages:
                entry = f"{msg['role']}: {msg['content']}"
                file.write(entry + "\n")
                self.new_entries.append(entry)
                self.db_content.append(entry)  # Update db_content in memory
        
        if len(self.new_entries) >= self.UPDATE_THRESHOLD:
            self.update_embeddings()

    def update_embeddings(self):
        if not self.new_entries:
            return

        logging.info(f"Updating embeddings with {len(self.new_entries)} new entries")
        new_embeddings = self.model.encode(self.new_entries, convert_to_tensor=True)
        
        if isinstance(self.db_embeddings, np.ndarray):
            self.db_embeddings = torch.from_numpy(self.db_embeddings)
        
        self.db_embeddings = torch.cat([self.db_embeddings, new_embeddings], dim=0)
        
        # Save updated embeddings
        torch.save({'embeddings': self.db_embeddings, 'indexes': torch.arange(len(self.db_content))}, self.EMBEDDINGS_FILE)
        
        self.new_entries.clear()
        logging.info("Embeddings updated successfully")

    def __init__(self, api_type: str):
        self.client = self.configure_api(api_type)
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.db_content = self.load_db_content()
        self.db_embeddings, _ = load_or_compute_embeddings(self.model)
        self.conversation_history: List[Dict[str, str]] = []
        self.new_entries: List[str] = []

    @staticmethod
    def configure_api(api_type: str) -> OpenAI:
        if api_type == "ollama":
            return OpenAI(base_url='http://localhost:11434/v1', api_key='phi3:instruct')
        elif api_type == "llama":
            return OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        else:
            raise ValueError("Invalid API type")

    def load_db_content(self) -> List[str]:
        if os.path.exists(self.DB_FILE):
            with open(self.DB_FILE, "r", encoding='utf-8') as db_file:
                return db_file.readlines()
        return []

    @staticmethod
    def lexical_search(query: str, db_content: List[str], top_k: int = 5) -> List[str]:
        query_words = set(re.findall(r'\w+', query.lower()))
        
        overlap_scores = []
        for context in db_content:
            context_words = set(re.findall(r'\w+', context.lower()))
            overlap = len(query_words.intersection(context_words))
            overlap_scores.append(overlap)
        
        top_indices = sorted(range(len(overlap_scores)), key=lambda i: overlap_scores[i], reverse=True)[:top_k]
        
        return [db_content[i].strip() for i in top_indices]

    def semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        if isinstance(self.db_embeddings, np.ndarray):
            self.db_embeddings = torch.from_numpy(self.db_embeddings)
        
        input_embedding = self.model.encode([query], convert_to_tensor=True)
        cos_scores = util.cos_sim(input_embedding, self.db_embeddings)[0]
        top_indices = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))[1].tolist()
        
        return [self.db_content[idx].strip() for idx in top_indices]

    def get_relevant_context(self, user_input: str, top_k: int = 5) -> Tuple[List[str], List[str]]:
        logging.info(f"DB Embeddings type: {type(self.db_embeddings)}")
        logging.info(f"DB Embeddings shape: {self.db_embeddings.shape if hasattr(self.db_embeddings, 'shape') else 'No shape attribute'}")
        logging.info(f"DB Content length: {len(self.db_content)}")
        
        if isinstance(self.db_embeddings, np.ndarray):
            self.db_embeddings = torch.from_numpy(self.db_embeddings)
        
        if self.db_embeddings.numel() == 0 or len(self.db_content) == 0:
            logging.warning("DB Embeddings or DB Content is empty")
            return [], []
        
        lexical_results = self.lexical_search(user_input, self.db_content, top_k)
        semantic_results = self.semantic_search(user_input, top_k)
        
        logging.info(f"Number of lexical results: {len(lexical_results)}")
        logging.info(f"Number of semantic results: {len(semantic_results)}")
        
        return lexical_results, semantic_results

    def ollama_chat(self, user_input: str, system_message: str) -> str:
        lexical_context, semantic_context = self.get_relevant_context(user_input)
        
        lexical_str = "\n".join(lexical_context) if lexical_context else "No relevant lexical context found."
        semantic_str = "\n".join(semantic_context) if semantic_context else "No relevant semantic context found."
        
        context_str = f"Lexical Search Results:\n{lexical_str}\n\nSemantic Search Results:\n{semantic_str}"
        
        logging.info(f"Context pulled: {context_str[:200]}...")

        messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history,
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_input}\n\nPlease use the most relevant information from either the lexical or semantic search results to answer the question. If neither set of results is relevant, you can say so and answer based on your general knowledge."}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.OLLAMA_MODEL,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your request."

    def append_to_db(self, new_messages: List[Dict[str, str]]):
        with open(self.DB_FILE, "a", encoding='utf-8') as file:
            for msg in new_messages:
                entry = f"{msg['role']}: {msg['content']}"
                file.write(entry + "\n")
                self.new_entries.append(entry)
        
        if len(self.new_entries) >= self.UPDATE_THRESHOLD:
            self.update_embeddings()

    def update_embeddings(self):
        if not self.new_entries:
            return

        logging.info(f"Updating embeddings with {len(self.new_entries)} new entries")
        new_embeddings = self.model.encode(self.new_entries, convert_to_tensor=True)
        
        if isinstance(self.db_embeddings, np.ndarray):
            self.db_embeddings = torch.from_numpy(self.db_embeddings)
        
        self.db_embeddings = torch.cat([self.db_embeddings, new_embeddings], dim=0)
        self.db_content.extend(self.new_entries)
        
        # Save updated embeddings
        torch.save({'embeddings': self.db_embeddings, 'indexes': torch.arange(len(self.db_content))}, self.EMBEDDINGS_FILE)
        
        self.new_entries.clear()
        logging.info("Embeddings updated successfully")

    def run(self):
        system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Not all info provided in context is useful. Reply with 'I don't see any relevant info in the context' if the given text does not provide the correct answer. Try to provide a comprehensive answer considering also what you can deduce from information provided as well as what you know already."

        print(f"{ANSIColor.YELLOW.value}Welcome to the RAG system. Type 'exit' to quit or 'clear' to clear conversation history.{ANSIColor.RESET.value}")

        while True:
            user_input = input(f"{ANSIColor.YELLOW.value}Ask a question about your documents: {ANSIColor.RESET.value}").strip()
            
            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using the RAG system. Goodbye!{ANSIColor.RESET.value}")
                self.update_embeddings()  # Final update before exiting
                break
            elif user_input.lower() == 'clear':
                self.conversation_history.clear()
                print(f"{ANSIColor.CYAN.value}Conversation history cleared.{ANSIColor.RESET.value}")
                continue

            if not user_input:
                print(f"{ANSIColor.PINK.value}Please enter a valid question.{ANSIColor.RESET.value}")
                continue

            response = self.ollama_chat(user_input, system_message)
            print(f"{ANSIColor.NEON_GREEN.value}Response: \n\n{response}{ANSIColor.RESET.value}")

            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})

            # Keep only the last MAX_HISTORY_LENGTH exchanges
            if len(self.conversation_history) > self.MAX_HISTORY_LENGTH * 2:
                self.conversation_history = self.conversation_history[-self.MAX_HISTORY_LENGTH * 2:]

            self.append_to_db(self.conversation_history[-2:])

def main(api_type: str):
    rag_system = RAGSystem(api_type)
    rag_system.run()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        main(api_type)
    else:
        print("Error: No API type provided.")
        sys.exit(1)
