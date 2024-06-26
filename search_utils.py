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

    def lexical_search(self, query: str, top_k: int = 5) -> List[int]:
        query_words = set(re.findall(r'\w+', query.lower()))
        overlap_scores = []
        for context in self.db_content:
            context_words = set(re.findall(r'\w+', context.lower()))
            overlap = len(query_words.intersection(context_words))
            overlap_scores.append(overlap)
        return sorted(range(len(overlap_scores)), key=lambda i: overlap_scores[i], reverse=True)[:top_k]

    def semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        if isinstance(self.db_embeddings, torch.Tensor):
            self.db_embeddings = self.db_embeddings.numpy()
        with torch.no_grad():
            input_embedding = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        cos_scores = util.cos_sim(input_embedding, torch.from_numpy(self.db_embeddings))[0]
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
                relevance_score += len(connected_docs) * 0.1
                relevance_score += len(family_relations) * 0.2
                
                node_scores.append((node, entity_type, connected_docs, family_relations, relevance_score))

        top_entities = sorted(node_scores, key=lambda x: x[4], reverse=True)[:top_k]
        
        if not top_entities:
            return ["No relevant information found in the knowledge graph."]

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

    def get_relevant_context(self, user_input: str, conversation_context: List[str], top_k: int = 5) -> Tuple[List[str], List[str], List[str], List[str]]:
        logging.info(f"DB Embeddings shape: {self.db_embeddings.shape if hasattr(self.db_embeddings, 'shape') else 'No shape attribute'}")
        logging.info(f"DB Content length: {len(self.db_content)}")

        if isinstance(self.db_embeddings, torch.Tensor):
            self.db_embeddings = self.db_embeddings.numpy()

        if self.db_embeddings.size == 0 or len(self.db_content) == 0:
            logging.warning("DB Embeddings or DB Content is empty")
            return [], [], [], []

        search_query = " ".join(list(conversation_context) + [user_input])

        lexical_indices = self.lexical_search(search_query, top_k)
        semantic_results = self.semantic_search(search_query, top_k)
        graph_results = self.get_graph_context(search_query, top_k)
        text_results = self.text_search(search_query, top_k)

        lexical_results = [self.db_content[i].strip() for i in lexical_indices]

        lexical_results = conversation_context + [r for r in lexical_results if r not in conversation_context]
        semantic_results = conversation_context + [r for r in semantic_results if r not in conversation_context]

        logging.info(f"Number of lexical results: {len(lexical_results)}")
        logging.info(f"Number of semantic results: {len(semantic_results)}")
        logging.info(f"Number of graph results: {len(graph_results)}")
        logging.info(f"Number of text search results: {len(text_results)}")

        return lexical_results[:top_k], semantic_results[:top_k], graph_results[:top_k], text_results[:top_k]

FAMILY_RELATIONS = [
    "sister", "sisters", "brother", "brothers", "father", "mother", "parent", "parents",
    "son", "sons", "daughter", "daughters", "husband", "wife", "spouse",
    "grandfather", "grandmother", "grandparent", "grandparents",
    "grandson", "granddaughter", "grandchild", "grandchildren",
    "uncle", "aunt", "cousin", "cousins", "niece", "nephew"
]
