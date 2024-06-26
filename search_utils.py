import re
import torch
from sentence_transformers import util
import logging
from typing import List, Tuple
import networkx as nx
import spacy

class SearchUtils:
    def __init__(self, model, db_embeddings, db_content, knowledge_graph):
        self.model = model
        self.db_embeddings = db_embeddings
        self.db_content = db_content
        self.knowledge_graph = knowledge_graph
        self.nlp = spacy.load("en_core_web_sm")

        # Default settings
        self.top_k = 5
        self.entity_relevance_threshold = 0.5
        self.lexical_weight = 1.0
        self.semantic_weight = 1.0
        self.graph_weight = 1.0
        self.text_weight = 1.0
        self.enable_lexical_search = True
        self.enable_semantic_search = True
        self.enable_graph_search = True
        self.enable_text_search = True

    def lexical_search(self, query: str) -> List[str]:
        if not self.enable_lexical_search:
            return []
        
        query_words = set(re.findall(r'\w+', query.lower()))
        overlap_scores = []
        for context in self.db_content:
            context_words = set(re.findall(r'\w+', context.lower()))
            overlap = len(query_words.intersection(context_words))
            overlap_scores.append(overlap)
        top_indices = sorted(range(len(overlap_scores)), key=lambda i: overlap_scores[i], reverse=True)[:self.top_k]
        return [str(self.db_content[i].strip()) for i in top_indices]

    def semantic_search(self, query: str) -> List[str]:
        if not self.enable_semantic_search:
            return []
        
        if isinstance(self.db_embeddings, torch.Tensor):
            self.db_embeddings = self.db_embeddings.numpy()
        with torch.no_grad():
            input_embedding = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        cos_scores = util.cos_sim(input_embedding, torch.from_numpy(self.db_embeddings))[0]
        top_indices = torch.topk(cos_scores, k=min(self.top_k, len(cos_scores)))[1].tolist()
        return [str(self.db_content[idx].strip()) for idx in top_indices]

    def get_graph_context(self, query: str) -> List[str]:
        if not self.enable_graph_search or not self.knowledge_graph.nodes():
            return []

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
                relevance_score += len(connected_docs) * 0.1
                relevance_score += len(family_relations) * 0.2
                
                if relevance_score >= self.entity_relevance_threshold:
                    node_scores.append((node, entity_type, connected_docs, family_relations, relevance_score))

        top_entities = sorted(node_scores, key=lambda x: x[4], reverse=True)[:self.top_k]
        
        if not top_entities:
            return []

        context = []
        for entity, entity_type, connected_docs, family_relations, _ in top_entities:
            entity_info = f"Entity: {entity} (Type: {entity_type})"
            doc_contexts = []
            for doc in connected_docs[:3]:
                doc_text = self.knowledge_graph.nodes[doc].get('text', '')
                doc_contexts.append(doc_text)
            
            related_entities = [n for n in self.knowledge_graph.neighbors(entity) if self.knowledge_graph.nodes[n]['type'] == 'entity']
            related_info = f"Related Entities: {', '.join(related_entities[:5])}"
            
            family_info = "Family Relations:"
            for rel in family_relations:
                if rel[0] == entity:
                    family_info += f"\n{entity} is the {rel[1]} of {rel[2]}"
                else:
                    family_info += f"\n{rel[0]} is the {rel[1]} of {entity}"
            
            context_text = f"{entity_info}\n{related_info}\n{family_info}\nRelevant Documents:\n" + "\n".join(doc_contexts)
            context.append(context_text)

        return context

    def text_search(self, query: str) -> List[str]:
        if not self.enable_text_search:
            return []
        
        query_terms = query.lower().split()
        results = []
        for content in self.db_content:
            if any(term in content.lower() for term in query_terms):
                results.append(content.strip())
        return sorted(results, key=lambda x: sum(term in x.lower() for term in query_terms), reverse=True)[:self.top_k]

    def extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]

    def get_relevant_context(self, user_input: str, conversation_context: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
        logging.info(f"DB Embeddings shape: {self.db_embeddings.shape if hasattr(self.db_embeddings, 'shape') else 'No shape attribute'}")
        logging.info(f"DB Content length: {len(self.db_content)}")

        if isinstance(self.db_embeddings, torch.Tensor):
            self.db_embeddings = self.db_embeddings.numpy()

        if self.db_embeddings.size == 0 or len(self.db_content) == 0:
            logging.warning("DB Embeddings or DB Content is empty")
            return [], [], [], []

        search_query = " ".join(list(conversation_context) + [user_input])

        lexical_results = self.lexical_search(search_query) if self.enable_lexical_search else []
        semantic_results = self.semantic_search(search_query) if self.enable_semantic_search else []
        graph_results = self.get_graph_context(search_query) if self.enable_graph_search else []
        text_results = self.text_search(search_query) if self.enable_text_search else []

        # Apply weights
        lexical_results = [(str(result), self.lexical_weight) for result in lexical_results]
        semantic_results = [(str(result), self.semantic_weight) for result in semantic_results]
        graph_results = [(str(result), self.graph_weight) for result in graph_results]
        text_results = [(str(result), self.text_weight) for result in text_results]

        # Combine all results and sort by weight
        all_results = lexical_results + semantic_results + graph_results + text_results
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for result, weight in all_results:
            if result not in seen:
                unique_results.append((result, weight))
                seen.add(result)

        # Separate results back into their categories
        lexical_results = [result for result, weight in unique_results if (result, self.lexical_weight) in lexical_results][:self.top_k]
        semantic_results = [result for result, weight in unique_results if (result, self.semantic_weight) in semantic_results][:self.top_k]
        graph_results = [result for result, weight in unique_results if (result, self.graph_weight) in graph_results][:self.top_k]
        text_results = [result for result, weight in unique_results if (result, self.text_weight) in text_results][:self.top_k]

        # Ensure all results are strings
        lexical_results = [str(result) for result in lexical_results]
        semantic_results = [str(result) for result in semantic_results]
        graph_results = [str(result) for result in graph_results]
        text_results = [str(result) for result in text_results]

        logging.info(f"Number of lexical results: {len(lexical_results)}")
        logging.info(f"Number of semantic results: {len(semantic_results)}")
        logging.info(f"Number of graph results: {len(graph_results)}")
        logging.info(f"Number of text search results: {len(text_results)}")

        return lexical_results, semantic_results, graph_results, text_results

# Setter functions for updating settings
def set_top_k(value: int):
    SearchUtils.top_k = value

def set_entity_relevance_threshold(value: float):
    SearchUtils.entity_relevance_threshold = value

def set_search_weights(lexical_weight: float, semantic_weight: float, graph_weight: float, text_weight: float):
    SearchUtils.lexical_weight = lexical_weight
    SearchUtils.semantic_weight = semantic_weight
    SearchUtils.graph_weight = graph_weight
    SearchUtils.text_weight = text_weight

def set_search_toggles(enable_lexical: bool, enable_semantic: bool, enable_graph: bool, enable_text: bool):
    SearchUtils.enable_lexical_search = enable_lexical
    SearchUtils.enable_semantic_search = enable_semantic
    SearchUtils.enable_graph_search = enable_graph
    SearchUtils.enable_text_search = enable_text

FAMILY_RELATIONS = [
    "sister", "sisters", "brother", "brothers", "father", "mother", "parent", "parents",
    "son", "sons", "daughter", "daughters", "husband", "wife", "spouse",
    "grandfather", "grandmother", "grandparent", "grandparents",
    "grandson", "granddaughter", "grandchild", "grandchildren",
    "uncle", "aunt", "cousin", "cousins", "niece", "nephew"
]
