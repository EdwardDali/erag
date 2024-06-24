import sys
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Union, Tuple, Optional
import logging
import re
from collections import Counter, deque
from embeddings_utils import load_or_compute_embeddings, compute_and_save_embeddings
from enum import Enum
from openai import OpenAI
import json
import networkx as nx
import spacy

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
    UPDATE_THRESHOLD = 10
    CONVERSATION_CONTEXT_SIZE = 3
    KNOWLEDGE_GRAPH_FILE = "knowledge_graph.json"

    def __init__(self, api_type: str):
        self.client = self.configure_api(api_type)
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.db_content = self.load_db_content()
        self.db_embeddings, _ = load_or_compute_embeddings(self.model)
        self.conversation_history: List[Dict[str, str]] = []
        self.new_entries: List[str] = []
        self.conversation_context: deque = deque(maxlen=self.CONVERSATION_CONTEXT_SIZE * 2)
        self.knowledge_graph = self.load_knowledge_graph()
        self.nlp = spacy.load("en_core_web_sm")
        
    def load_knowledge_graph(self) -> nx.Graph:
        try:
            with open(self.KNOWLEDGE_GRAPH_FILE, 'r') as file:
                graph_data = json.load(file)
            return nx.node_link_graph(graph_data)
        except Exception as e:
            logging.error(f"Failed to load knowledge graph: {str(e)}")
            return nx.Graph()

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

    def get_graph_context(self, query: str, top_k: int = 5) -> List[str]:
        if not self.knowledge_graph:
            return ["Knowledge graph is not available"]

        query_entities = set(self.extract_entities(query))
        node_scores = []

        for node, data in self.knowledge_graph.nodes(data=True):
            if data.get('type') == 'document':
                contained_entities = set(self.knowledge_graph.neighbors(node))
                entity_overlap = len(query_entities.intersection(contained_entities))
                text_similarity = self.compute_similarity(query, data.get('text', ''))
                score = entity_overlap + text_similarity
                node_scores.append((node, score))

        top_nodes = sorted(node_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        context = []
        for node, _ in top_nodes:
            doc_text = self.knowledge_graph.nodes[node].get('text', '')
            related_entities = list(self.knowledge_graph.neighbors(node))
            entity_info = []
            for entity in related_entities:
                relations = self.knowledge_graph[node][entity].get('relation', '')
                entity_info.append(f"{entity} ({relations})")
            context.append(f"Document: {doc_text}\nRelated Entities: {', '.join(entity_info)}")

        return context

    def extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]

    def compute_similarity(self, text1: str, text2: str) -> float:
        # Disable progress bar for encoding
        with torch.no_grad():
            embedding1 = self.model.encode([text1], convert_to_tensor=True, show_progress_bar=False)
            embedding2 = self.model.encode([text2], convert_to_tensor=True, show_progress_bar=False)
        return util.pytorch_cos_sim(embedding1, embedding2).item()

    def semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        if isinstance(self.db_embeddings, np.ndarray):
            self.db_embeddings = torch.from_numpy(self.db_embeddings)

        # Disable progress bar for encoding
        with torch.no_grad():
            input_embedding = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        cos_scores = util.cos_sim(input_embedding, self.db_embeddings)[0]
        top_indices = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))[1].tolist()

        return [self.db_content[idx].strip() for idx in top_indices]

    def get_relevant_context(self, user_input: str, top_k: int = 5) -> Tuple[List[str], List[str], List[str]]:
        logging.info(f"DB Embeddings shape: {self.db_embeddings.shape if hasattr(self.db_embeddings, 'shape') else 'No shape attribute'}")
        logging.info(f"DB Content length: {len(self.db_content)}")

        if isinstance(self.db_embeddings, np.ndarray):
            self.db_embeddings = torch.from_numpy(self.db_embeddings)

        if self.db_embeddings.numel() == 0 or len(self.db_content) == 0:
            logging.warning("DB Embeddings or DB Content is empty")
            return [], [], []

        search_query = " ".join(list(self.conversation_context) + [user_input])

        lexical_results = self.lexical_search(search_query, self.db_content, top_k)
        semantic_results = self.semantic_search(search_query, top_k)
        graph_results = self.get_graph_context(search_query, top_k)

        conversation_context = list(self.conversation_context)
        lexical_results = conversation_context + [r for r in lexical_results if r not in conversation_context]
        semantic_results = conversation_context + [r for r in semantic_results if r not in conversation_context]

        logging.info(f"Number of lexical results: {len(lexical_results)}")
        logging.info(f"Number of semantic results: {len(semantic_results)}")
        logging.info(f"Number of graph results: {len(graph_results)}")

        return lexical_results[:top_k], semantic_results[:top_k], graph_results[:top_k]

    def ollama_chat(self, user_input: str, system_message: str) -> Tuple[str, str]:
        lexical_context, semantic_context, graph_context = self.get_relevant_context(user_input)
        
        lexical_str = "\n".join(lexical_context)
        semantic_str = "\n".join(semantic_context)
        graph_str = "\n".join(graph_context)

        standard_context = f"Conversation Context:\n{' '.join(self.conversation_context)}\n\nLexical Search Results:\n{lexical_str}\n\nSemantic Search Results:\n{semantic_str}"
        graph_context_str = f"Knowledge Graph Context:\n{graph_str}"

        logging.info(f"Standard context pulled: {standard_context[:200]}...")
        logging.info(f"Graph context pulled: {graph_context_str[:200]}...")

        standard_messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history,
            {"role": "user", "content": f"Context:\n{standard_context}\n\nQuestion: {user_input}\n\nPlease prioritize the Conversation Context when answering, followed by the most relevant information from either the lexical or semantic search results. If none of the provided context is relevant, you can answer based on your general knowledge."}
        ]

        graph_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{graph_context_str}\n\nQuestion: {user_input}\n\nPlease provide an answer based on the information from the Knowledge Graph Context. If the context doesn't provide relevant information, state that and provide a general answer."}
        ]

        try:
            standard_response = self.client.chat.completions.create(
                model=self.OLLAMA_MODEL,
                messages=standard_messages,
                temperature=0.1
            ).choices[0].message.content

            graph_response = self.client.chat.completions.create(
                model=self.OLLAMA_MODEL,
                messages=graph_messages,
                temperature=0.1
            ).choices[0].message.content

            return standard_response, graph_response
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your request.", "Error in processing graph-based response."

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

            standard_response, graph_response = self.ollama_chat(user_input, system_message)
            print(f"{ANSIColor.NEON_GREEN.value}Standard Response: \n\n{standard_response}{ANSIColor.RESET.value}")
            print(f"\n{ANSIColor.CYAN.value}Graph-based Response: \n\n{graph_response}{ANSIColor.RESET.value}")

            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": standard_response})
            self.update_conversation_context(user_input, standard_response)

            if len(self.conversation_history) > self.MAX_HISTORY_LENGTH * 2:
                self.conversation_history = self.conversation_history[-self.MAX_HISTORY_LENGTH * 2:]

            self.append_to_db(self.conversation_history[-2:])

    def update_conversation_context(self, user_input: str, assistant_response: str):
        self.conversation_context.append(user_input)
        self.conversation_context.append(assistant_response)

    def append_to_db(self, new_entries: List[Dict[str, str]]):
        with open(self.DB_FILE, "a", encoding='utf-8') as db_file:
            for entry in new_entries:
                db_file.write(f"{entry['role']}: {entry['content']}\n")
        self.new_entries.extend([entry['content'] for entry in new_entries])

    def update_embeddings(self):
        if len(self.new_entries) >= self.UPDATE_THRESHOLD:
            logging.info("Updating embeddings...")
            new_embeddings = self.model.encode(self.new_entries, convert_to_tensor=True)
            self.db_embeddings = torch.cat([self.db_embeddings, new_embeddings], dim=0)
            compute_and_save_embeddings(self.db_content + self.new_entries, self.model, self.EMBEDDINGS_FILE)
            self.new_entries.clear()
            logging.info("Embeddings updated successfully.")

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
        
