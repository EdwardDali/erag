import sys
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Union, Tuple, Optional
import logging
import re
from collections import Counter, deque
from embeddings_utils import load_embeddings_and_data
from enum import Enum
from openai import OpenAI
import networkx as nx
import spacy
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ANSIColor(Enum):
    PINK = '\033[95m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    NEON_GREEN = '\033[92m'
    RESET = '\033[0m'

FAMILY_RELATIONS = [
    "sister", "sisters", "brother", "brothers", "father", "mother", "parent", "parents",
    "son", "sons", "daughter", "daughters", "husband", "wife", "spouse",
    "grandfather", "grandmother", "grandparent", "grandparents",
    "grandson", "granddaughter", "grandchild", "grandchildren",
    "uncle", "aunt", "cousin", "cousins", "niece", "nephew"
]

class RAGSystem:
    MAX_HISTORY_LENGTH = 5
    EMBEDDINGS_FILE = "db_embeddings.pt"
    DB_FILE = "db.txt"
    MODEL_NAME = "all-MiniLM-L6-v2"
    OLLAMA_MODEL = "phi3:instruct"
    UPDATE_THRESHOLD = 10
    CONVERSATION_CONTEXT_SIZE = 3
    KNOWLEDGE_GRAPH_FILE = "knowledge_graph.json"
    RESULTS_FILE = "results.txt"

    def __init__(self, api_type: str):
        self.client = self.configure_api(api_type)
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.db_embeddings, _, _ = self.load_embeddings()
        self.db_content = self.load_db_content()
        self.conversation_history: List[Dict[str, str]] = []
        self.new_entries: List[str] = []
        self.conversation_context: deque = deque(maxlen=self.CONVERSATION_CONTEXT_SIZE * 2)
        self.knowledge_graph = self.load_knowledge_graph()
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def configure_api(api_type: str) -> OpenAI:
        if api_type == "ollama":
            return OpenAI(base_url='http://localhost:11434/v1', api_key='phi3:instruct')
        elif api_type == "llama":
            return OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        else:
            raise ValueError("Invalid API type")

    def load_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        embeddings, indexes, content = load_embeddings_and_data(self.EMBEDDINGS_FILE)
        if embeddings is None or indexes is None or content is None:
            logging.error(f"Failed to load data from {self.EMBEDDINGS_FILE}. Make sure the file exists and is properly formatted.")
            return torch.tensor([]), torch.tensor([]), []
        return embeddings, indexes, content
    
    def load_db_content(self) -> List[str]:
        if os.path.exists(self.DB_FILE):
          with open(self.DB_FILE, "r", encoding='utf-8') as db_file:
            return db_file.readlines()
        return []

    def load_knowledge_graph(self) -> nx.Graph:
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

    def lexical_search(self, query: str, top_k: int = 5) -> List[int]:
        query_words = set(re.findall(r'\w+', query.lower()))
        overlap_scores = []
        for context in self.db_content:
            context_words = set(re.findall(r'\w+', context.lower()))
            overlap = len(query_words.intersection(context_words))
            overlap_scores.append(overlap)
        return sorted(range(len(overlap_scores)), key=lambda i: overlap_scores[i], reverse=True)[:top_k]

    def semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        if isinstance(self.db_embeddings, np.ndarray):
           self.db_embeddings = torch.from_numpy(self.db_embeddings)
        with torch.no_grad():
           input_embedding = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        cos_scores = util.cos_sim(input_embedding, self.db_embeddings)[0]
        top_indices = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))[1].tolist()
        return [self.db_content[idx].strip() for idx in top_indices]

    def get_graph_context(self, query: str, top_k: int = 5) -> List[str]:
        if not self.knowledge_graph.nodes():
            logging.warning("Knowledge graph is empty or not available.")
            return ["Knowledge graph is not available or empty."]

        query_entities = set(self.extract_entities(query))
        node_scores = []

        for node, data in self.knowledge_graph.nodes(data=True):
            if data.get('type') == 'entity':
                entity_name = node
                entity_type = data.get('entity_type', 'Unknown')
                connected_docs = [n for n in self.knowledge_graph.neighbors(node) if n.startswith('doc_')]
                family_relations = [
                    (node, edge_data['relation'], neighbor) 
                    for neighbor, edge_data in self.knowledge_graph[node].items() 
                    if 'relation' in edge_data and edge_data['relation'] in FAMILY_RELATIONS
                ]
                
                relevance_score = 1 if entity_name.lower() in query.lower() else 0
                relevance_score += len(connected_docs) * 0.1  # More connected docs = more relevant
                relevance_score += len(family_relations) * 0.2  # Family relations are more relevant
                
                node_scores.append((node, entity_type, connected_docs, family_relations, relevance_score))

        top_entities = sorted(node_scores, key=lambda x: x[4], reverse=True)[:top_k]
        
        if not top_entities:
            return ["No relevant information found in the knowledge graph."]

        context = []
        for entity, entity_type, connected_docs, family_relations, _ in top_entities:
            entity_info = f"Entity: {entity} (Type: {entity_type})"
            doc_contexts = []
            for doc in connected_docs[:3]:  # Limit to top 3 connected documents per entity
                doc_text = self.knowledge_graph.nodes[doc].get('text', '')
                doc_contexts.append(doc_text)
            
            related_entities = [n for n in self.knowledge_graph.neighbors(entity) if self.knowledge_graph.nodes[n]['type'] == 'entity']
            related_info = f"Related Entities: {', '.join(related_entities[:5])}"  # Limit to top 5 related entities
            
            family_info = "Family Relations:"
            for rel in family_relations:
                if rel[0] == entity:
                    family_info += f"\n{entity} is the {rel[1]} of {rel[2]}"
                else:
                    family_info += f"\n{rel[0]} is the {rel[1]} of {entity}"
            
            context_text = f"{entity_info}\n{related_info}\n{family_info}\nRelevant Documents:\n" + "\n".join(doc_contexts)
            context.append(context_text)

        return context

    def get_relevant_context(self, user_input: str, top_k: int = 5) -> Tuple[List[str], List[str], List[str], List[str]]:
        logging.info(f"DB Embeddings shape: {self.db_embeddings.shape if hasattr(self.db_embeddings, 'shape') else 'No shape attribute'}")
        logging.info(f"DB Content length: {len(self.db_content)}")

        if isinstance(self.db_embeddings, np.ndarray):
          self.db_embeddings = torch.from_numpy(self.db_embeddings)

        if self.db_embeddings.numel() == 0 or len(self.db_content) == 0:
          logging.warning("DB Embeddings or DB Content is empty")
          return [], [], [], []

        search_query = " ".join(list(self.conversation_context) + [user_input])

        lexical_indices = self.lexical_search(search_query, top_k)
        semantic_results = self.semantic_search(search_query, top_k)
        graph_results = self.get_graph_context(search_query, top_k)
        text_results = self.text_search(search_query, top_k)

        lexical_results = [self.db_content[i].strip() for i in lexical_indices]

        conversation_context = list(self.conversation_context)
        lexical_results = conversation_context + [r for r in lexical_results if r not in conversation_context]
        semantic_results = conversation_context + [r for r in semantic_results if r not in conversation_context]

        logging.info(f"Number of lexical results: {len(lexical_results)}")
        logging.info(f"Number of semantic results: {len(semantic_results)}")
        logging.info(f"Number of graph results: {len(graph_results)}")
        logging.info(f"Number of text search results: {len(text_results)}")

        return lexical_results[:top_k], semantic_results[:top_k], graph_results[:top_k], text_results[:top_k]

    def text_search(self, query: str, top_k: int = 5) -> List[str]:
        query_terms = query.lower().split()
        results = []
        for content in self.db_content:
            if any(term in content.lower() for term in query_terms):
                results.append(content.strip())
        return sorted(results, key=lambda x: sum(term in x.lower() for term in query_terms), reverse=True)[:top_k]

    def extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]

    def compute_similarity(self, text1: str, text2: str) -> float:
        with torch.no_grad():
            embedding1 = self.model.encode([text1], convert_to_tensor=True, show_progress_bar=False)
            embedding2 = self.model.encode([text2], convert_to_tensor=True, show_progress_bar=False)
        return util.pytorch_cos_sim(embedding1, embedding2).item()

    def ollama_chat(self, user_input: str, system_message: str) -> str:
        lexical_context, semantic_context, graph_context, text_context = self.get_relevant_context(user_input)
        
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
                temperature=0.1
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

def main(api_type: str):
    rag_system = RAGSystem(api_type)
    rag_system.run()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        main(api_type)
    else:
        print("Error: No API type provided.")
        print("Usage: python localrag14.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
