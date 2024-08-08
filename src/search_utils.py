import re
import numpy as np
import logging
from typing import List, Tuple
import networkx as nx
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from src.settings import settings
import os
import json
from src.look_and_feel import success, info, warning, error

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

class SearchUtils:
    def __init__(self, model, db_embeddings=None, db_content=None, knowledge_graph=None):
        self.model = model
        
        # Load db_embeddings and db_content
        if db_embeddings is not None and db_content is not None:
            self.db_embeddings = db_embeddings
            self.db_content = db_content
        else:
            embeddings_path = os.path.join(settings.output_folder, os.path.basename(settings.embeddings_file_path))
            if os.path.exists(embeddings_path):
                loaded_data = np.load(embeddings_path, allow_pickle=True).item()
                self.db_embeddings = loaded_data['embeddings']
                self.db_content = loaded_data['content']
                logging.info(info(f"Loaded embeddings and content from {embeddings_path}"))
            else:
                logging.error(error(f"Embeddings file not found: {embeddings_path}"))
                self.db_embeddings = np.array([])
                self.db_content = []

        # Load knowledge graph
        if knowledge_graph is not None:
            self.knowledge_graph = knowledge_graph
        else:
            graph_path = os.path.join(settings.output_folder, os.path.basename(settings.knowledge_graph_file_path))
            if os.path.exists(graph_path):
                with open(graph_path, 'r') as f:
                    graph_data = json.load(f)
                self.knowledge_graph = nx.node_link_graph(graph_data)
                logging.info(info(f"Loaded knowledge graph from {graph_path}"))
            else:
                logging.error(error(f"Knowledge graph file not found: {graph_path}"))
                self.knowledge_graph = nx.Graph()

    def lexical_search(self, query: str) -> List[str]:
        if not settings.enable_lexical_search:
            return []
        
        query_words = set(re.findall(r'\w+', query.lower()))
        overlap_scores = []
        for context in self.db_content:
            context_words = set(re.findall(r'\w+', context.lower()))
            overlap = len(query_words.intersection(context_words))
            overlap_scores.append(overlap)
        top_indices = sorted(range(len(overlap_scores)), key=lambda i: overlap_scores[i], reverse=True)[:settings.top_k]
        return [str(self.db_content[i].strip()) for i in top_indices]

    def semantic_search(self, query: str) -> List[str]:
        if not settings.enable_semantic_search:
            return []
        
        if hasattr(self.model, 'encode'):
            # For models with encode method
            input_embedding = self.model.encode([query])
            cos_scores = np.dot(self.db_embeddings, input_embedding.T).flatten()
            top_indices = cos_scores.argsort()[-settings.top_k:][::-1]
        else:
            # Fallback to keyword matching
            query_words = set(query.lower().split())
            scores = [sum(word in content.lower() for word in query_words) for content in self.db_content]
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:settings.top_k]
        
        return [str(self.db_content[idx].strip()) for idx in top_indices]

    def get_graph_context(self, query: str) -> List[str]:
        if not settings.enable_graph_search or not self.knowledge_graph.nodes():
            return []

        query_entities = set(self.extract_entities(query))
        node_scores = []

        for node, data in self.knowledge_graph.nodes(data=True):
            if data.get('type') == 'entity':
                entity_name = node
                entity_type = data.get('entity_type', 'Unknown')
                confidence = data.get('confidence', 0.5)  # Default confidence if not set
                connected_chunks = [n for n in self.knowledge_graph.neighbors(node) if self.knowledge_graph.nodes[n]['type'] == 'chunk']
                connected_docs = set([self.get_parent_document(chunk) for chunk in connected_chunks])
                
                relevance_score = 1 if entity_name.lower() in query.lower() else 0
                relevance_score += len(connected_chunks) * 0.1
                relevance_score += len(connected_docs) * 0.2
                relevance_score *= confidence  # Adjust relevance based on confidence
                
                if relevance_score >= settings.entity_relevance_threshold:
                    node_scores.append((node, entity_type, connected_chunks, relevance_score))

        top_entities = sorted(node_scores, key=lambda x: x[3], reverse=True)[:settings.top_k]
        
        context = []
        for entity, entity_type, connected_chunks, _ in top_entities:
            entity_info = f"Entity: {entity} (Type: {entity_type})"
            chunk_contexts = []
            for chunk in connected_chunks[:3]:
                chunk_text = self.knowledge_graph.nodes[chunk].get('text', '')
                chunk_contexts.append(chunk_text)
            
            related_entities = [n for n in self.knowledge_graph.neighbors(entity) if self.knowledge_graph.nodes[n]['type'] == 'entity']
            related_info = f"Related Entities: {', '.join(related_entities[:5])}"
            
            context_text = f"{entity_info}\n{related_info}\nRelevant Chunks:\n" + "\n".join(chunk_contexts)
            context.append(context_text)

        return context

    def text_search(self, query: str) -> List[str]:
        if not settings.enable_text_search:
            return []
        
        query_terms = query.lower().split()
        results = []
        for content in self.db_content:
            if any(term in content.lower() for term in query_terms):
                results.append(content.strip())
        return sorted(results, key=lambda x: sum(term in x.lower() for term in query_terms), reverse=True)[:settings.top_k]

    def extract_entities(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entities.append(' '.join(c[0] for c in chunk))
        return entities

    def get_parent_document(self, chunk_node: str) -> str:
        for neighbor in self.knowledge_graph.neighbors(chunk_node):
            if self.knowledge_graph.nodes[neighbor]['type'] == 'document':
                return neighbor
        return None

    def get_relevant_context(self, user_input: str, conversation_context: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
        logging.info(info(f"DB Embeddings shape: {self.db_embeddings.shape if hasattr(self.db_embeddings, 'shape') else 'No shape attribute'}"))
        logging.info(info(f"DB Content length: {len(self.db_content)}"))

        if self.db_embeddings.size == 0 or len(self.db_content) == 0:
            logging.warning(warning("DB Embeddings or DB Content is empty"))
            return [], [], [], []

        search_query = " ".join(list(conversation_context) + [user_input])

        lexical_results = self.lexical_search(search_query)
        semantic_results = self.semantic_search(search_query)
        graph_results = self.get_graph_context(search_query)
        text_results = self.text_search(search_query)

        logging.info(success(f"Number of lexical results: {len(lexical_results)}"))
        logging.info(success(f"Number of semantic results: {len(semantic_results)}"))
        logging.info(success(f"Number of graph results: {len(graph_results)}"))
        logging.info(success(f"Number of text search results: {len(text_results)}"))

        return lexical_results, semantic_results, graph_results, text_results