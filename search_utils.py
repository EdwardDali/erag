import re
import torch
from sentence_transformers import util
import logging
from typing import List, Tuple
import networkx as nx
import spacy
from settings import settings

import re
import torch
from sentence_transformers import util
import logging
from typing import List, Tuple
import networkx as nx
import spacy
from settings import settings

class SearchUtils:
    def __init__(self, model, db_embeddings, db_content, knowledge_graph):
        self.model = model
        self.db_embeddings = db_embeddings
        self.db_content = db_content
        self.knowledge_graph = knowledge_graph
        self.nlp = spacy.load(settings.nlp_model)

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
        
        if isinstance(self.db_embeddings, torch.Tensor):
            self.db_embeddings = self.db_embeddings.numpy()
        
        if hasattr(self.model, 'encode'):
            # For SentenceTransformer models
            with torch.no_grad():
                input_embedding = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False)
            cos_scores = util.cos_sim(input_embedding, torch.from_numpy(self.db_embeddings))[0]
            top_indices = torch.topk(cos_scores, k=min(settings.top_k, len(cos_scores)))[1].tolist()
        else:
            # For LLM models (Ollama, Llama), we'll use a simple keyword matching as a fallback
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
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]

    def get_parent_document(self, chunk_node: str) -> str:
        for neighbor in self.knowledge_graph.neighbors(chunk_node):
            if self.knowledge_graph.nodes[neighbor]['type'] == 'document':
                return neighbor
        return None

    def get_relevant_context(self, user_input: str, conversation_context: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
        logging.info(f"DB Embeddings shape: {self.db_embeddings.shape if hasattr(self.db_embeddings, 'shape') else 'No shape attribute'}")
        logging.info(f"DB Content length: {len(self.db_content)}")

        if isinstance(self.db_embeddings, torch.Tensor):
            self.db_embeddings = self.db_embeddings.numpy()

        if self.db_embeddings.size == 0 or len(self.db_content) == 0:
            logging.warning("DB Embeddings or DB Content is empty")
            return [], [], [], []

        search_query = " ".join(list(conversation_context) + [user_input])

        lexical_results = self.lexical_search(search_query)
        semantic_results = self.semantic_search(search_query)
        graph_results = self.get_graph_context(search_query)
        text_results = self.text_search(search_query)

        logging.info(f"Number of lexical results: {len(lexical_results)}")
        logging.info(f"Number of semantic results: {len(semantic_results)}")
        logging.info(f"Number of graph results: {len(graph_results)}")
        logging.info(f"Number of text search results: {len(text_results)}")

        return lexical_results, semantic_results, graph_results, text_results
