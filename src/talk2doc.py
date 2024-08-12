import sys
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging
from collections import deque
from src.embeddings_utils import load_embeddings_and_data
import networkx as nx
import json
from src.search_utils import SearchUtils
from src.settings import settings
from src.api_model import EragAPI
from src.look_and_feel import success, info, warning, error, colorize, MAGENTA, RESET

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGSystem:
    def __init__(self, erag_api: EragAPI):
        self.erag_api = erag_api
        self.embedding_model = SentenceTransformer(settings.sentence_transformer_model)
        self.db_embeddings, _, _ = self.load_embeddings()
        self.db_content = self.load_db_content()
        self.conversation_history = []
        self.new_entries = []
        self.conversation_context = deque(maxlen=settings.conversation_context_size * 2)
        self.knowledge_graph = self.load_knowledge_graph()
        self.search_utils = SearchUtils(self.embedding_model, self.db_embeddings, self.db_content, self.knowledge_graph)

    def load_embeddings(self):
        embeddings, indexes, content = load_embeddings_and_data(settings.embeddings_file_path)
        if embeddings is None or indexes is None or content is None:
            logging.error(f"Failed to load data from {settings.embeddings_file_path}. Make sure the file exists and is properly formatted.")
            return torch.tensor([]), torch.tensor([]), []
        return embeddings, indexes, content
    
    def load_db_content(self):
        if os.path.exists(settings.db_file_path):
            with open(settings.db_file_path, "r", encoding='utf-8') as db_file:
                return db_file.readlines()
        return []

    def load_knowledge_graph(self):
        try:
            if not os.path.exists(settings.knowledge_graph_file_path):
                logging.warning(f"Knowledge graph file {settings.knowledge_graph_file_path} not found.")
                return nx.Graph()

            with open(settings.knowledge_graph_file_path, 'r') as file:
                graph_data = json.load(file)
            
            G = nx.node_link_graph(graph_data)
            logging.info(f"Successfully loaded knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            return G
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse knowledge graph JSON: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load knowledge graph: {str(e)}")
        
        return nx.Graph()

    def get_response(self, query: str) -> str:
        system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Prioritize the most recent conversation context when answering questions, but also consider other relevant information if necessary. If the given context doesn't provide a suitable answer, rely on your general knowledge."

        lexical_results, semantic_results, graph_results, text_results = self.search_utils.get_relevant_context(query, list(self.conversation_context))
        
        # Combine all contexts for re-ranking
        all_contexts = lexical_results + semantic_results + graph_results + text_results

        # Re-rank the combined contexts
        reranked_contexts = self.search_utils.rerank_results(query, all_contexts, settings.rerank_top_k)

        combined_context = f"""Conversation Context:\n{' '.join(self.conversation_context)}

    Relevant Context (Re-ranked):
    {' '.join(reranked_contexts)}"""

        messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history,
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {query}\n\nPlease prioritize the Conversation Context when answering, followed by the most relevant information from the re-ranked context. If none of the provided context is relevant, you can answer based on your general knowledge."}
        ]

        try:
            response = self.erag_api.chat(messages, temperature=settings.temperature)
            print(success(f"Generated response for query: {query[:50]}..."))
            self.save_debug_results(query, lexical_results, semantic_results, graph_results, text_results, reranked_contexts, response)
            return response
        except Exception as e:
            error_message = f"Error in API call: {str(e)}"
            print(error(error_message))
            logging.error(error_message)
            return f"I'm sorry, but I encountered an error while processing your request: {str(e)}"

    def run(self):
        print(warning("Welcome to the RAG system. Type 'exit' to quit or 'clear' to clear conversation history."))

        while True:
            user_input = input(success("Ask a question about your documents: ")).strip()

            if user_input.lower() == 'exit':
                print(warning("Thank you for using the RAG system. Goodbye!"))
                self.update_embeddings()
                break
            elif user_input.lower() == 'clear':
                self.conversation_history.clear()
                self.conversation_context.clear()
                print(info("Conversation history and context cleared."))
                continue

            if not user_input:
                print(error("Please enter a valid question."))
                continue

            response = self.get_response(user_input)
            print(colorize("Response: \n\n", MAGENTA) + f"{response}{RESET}")

            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            self.update_conversation_context(user_input, response)

            if len(self.conversation_history) > settings.max_history_length * 2:
                self.conversation_history = self.conversation_history[-settings.max_history_length * 2:]

            self.append_to_db(self.conversation_history[-2:])

    def update_conversation_context(self, user_input: str, assistant_response: str):
        self.conversation_context.append(user_input)
        self.conversation_context.append(assistant_response)

    def append_to_db(self, new_entries: List[Dict[str, str]]):
        with open(settings.db_file_path, "a", encoding="utf-8") as db_file:
            for entry in new_entries:
                db_file.write(f"{entry['role']}: {entry['content']}\n")
        self.new_entries.extend([entry['content'] for entry in new_entries])
        
        # Also update db_content
        self.db_content.extend([f"{entry['role']}: {entry['content']}" for entry in new_entries])

    def update_embeddings(self):
        if len(self.new_entries) >= settings.update_threshold:
            logging.info("Updating embeddings...")
            new_embeddings = self.embedding_model.encode(self.new_entries, convert_to_tensor=True, show_progress_bar=False)
            self.db_embeddings = torch.cat([self.db_embeddings, new_embeddings], dim=0)
            
            # Save updated embeddings
            data_to_save = {
                'embeddings': self.db_embeddings,
                'indexes': torch.arange(len(self.db_content)),
                'content': self.db_content
            }
            torch.save(data_to_save, settings.embeddings_file_path)
            
            self.new_entries.clear()
            logging.info("Embeddings updated successfully.")

    def save_debug_results(self, user_input: str, 
                        lexical_results: List[str],
                        semantic_results: List[str],
                        graph_results: List[str],
                        text_results: List[str],
                        reranked_results: List[str],
                        response: str):
        with open(settings.results_file_path, "a", encoding='utf-8') as f:
            f.write(f"User Input: {user_input}\n\n")
            
            f.write("Original Search Results:\n")
            f.write("Lexical Search Results:\n")
            for i, content in enumerate(lexical_results, 1):
                f.write(f"{i}. {content}\n")
            f.write("\nSemantic Search Results:\n")
            for i, content in enumerate(semantic_results, 1):
                f.write(f"{i}. {content}\n")
            f.write("\nGraph Context Results:\n")
            for i, content in enumerate(graph_results, 1):
                f.write(f"{i}. {content}\n")
            f.write("\nText Search Results:\n")
            for i, content in enumerate(text_results, 1):
                f.write(f"{i}. {content}\n")
            
            f.write("\nRe-ranked Results:\n")
            for i, content in enumerate(reranked_results, 1):
                f.write(f"{i}. {content}\n")
            
            f.write("\nCombined Response:\n")
            f.write(f"{response}\n")
            f.write("\n" + "="*50 + "\n\n")

def main(api_type: str):
    erag_api = EragAPI(api_type)
    rag_system = RAGSystem(erag_api)
    rag_system.run()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        main(api_type)
    else:
        print(error("No API type provided."))
        print(warning("Usage: python src/talk2doc.py <api_type>"))
        print(info("Available API types: ollama, llama, sentence_transformer"))
        sys.exit(1)
