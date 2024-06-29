import spacy
import networkx as nx
import json
from sentence_transformers import SentenceTransformer, util
import torch
import logging
from typing import List, Tuple, Dict
from nltk.tokenize import sent_tokenize
import nltk
from embeddings_utils import load_embeddings_and_data

nltk.download('punkt', quiet=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for settings
NLP_MODEL = "en_core_web_sm"
SIMILARITY_THRESHOLD = 0.7
MIN_ENTITY_OCCURRENCE = 1
KNOWLEDGE_GRAPH_FILE = "knowledge_graph.json"
EMBEDDINGS_FILE = "db_embeddings.pt"
CHUNK_SIZE = 5000  # Increased chunk size

# Initialize NLP model
nlp = spacy.load(NLP_MODEL)

def set_nlp_model(new_model: str):
    global NLP_MODEL, nlp
    NLP_MODEL = new_model
    nlp = spacy.load(NLP_MODEL)
    logging.info(f"NLP model set to {NLP_MODEL}")

def set_graph_settings(similarity_threshold: float, min_entity_occurrence: int):
    global SIMILARITY_THRESHOLD, MIN_ENTITY_OCCURRENCE
    SIMILARITY_THRESHOLD = similarity_threshold
    MIN_ENTITY_OCCURRENCE = min_entity_occurrence
    logging.info(f"Graph settings updated: Similarity Threshold={SIMILARITY_THRESHOLD}, "
                 f"Min Entity Occurrence={MIN_ENTITY_OCCURRENCE}")

def set_knowledge_graph_file(new_file: str):
    global KNOWLEDGE_GRAPH_FILE
    KNOWLEDGE_GRAPH_FILE = new_file
    logging.info(f"Knowledge graph file set to {KNOWLEDGE_GRAPH_FILE}")

def set_embeddings_file(new_file: str):
    global EMBEDDINGS_FILE
    EMBEDDINGS_FILE = new_file
    logging.info(f"Embeddings file set to {EMBEDDINGS_FILE}")

def preprocess_text(text: str) -> str:
    return ' '.join(text.split())

def extract_entities_with_confidence(text: str) -> List[Tuple[str, str, float]]:
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        # Calculate a simple confidence score based on entity length and label
        confidence = min(1.0, (len(ent.text) / 10) * (1 if ent.label_ in ['PERSON', 'ORG', 'GPE'] else 0.7))
        entities.append((ent.text, ent.label_, confidence))
    return entities

def chunk_document(document: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    sentences = sent_tokenize(document)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_networkx_graph(data: List[str], embeddings: torch.Tensor) -> nx.Graph:
    G = nx.Graph()
    entity_count = {}
    
    for doc_idx, document in enumerate(data):
        doc_node = f"doc_{doc_idx}"
        G.add_node(doc_node, type='document', text=document)
        
        chunks = chunk_document(document)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_node = f"doc_{doc_idx}_chunk_{chunk_idx}"
            G.add_node(chunk_node, type='chunk', text=chunk)
            G.add_edge(doc_node, chunk_node, relation='contains')
            
            entities = extract_entities_with_confidence(chunk)
            for entity, entity_type, confidence in entities:
                entity_count[entity] = entity_count.get(entity, 0) + 1
                if entity_count[entity] >= MIN_ENTITY_OCCURRENCE:
                    if not G.has_node(entity):
                        G.add_node(entity, type='entity', entity_type=entity_type, confidence=confidence)
                    G.add_edge(chunk_node, entity, relation='contains', confidence=confidence)
    
    # Add semantic similarity edges between document nodes
    doc_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'document']
    for i, node1 in enumerate(doc_nodes):
        for j in range(i+1, len(doc_nodes)):
            node2 = doc_nodes[j]
            similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            if similarity > SIMILARITY_THRESHOLD:
                G.add_edge(node1, node2, relation='similar', weight=similarity, confidence=similarity)
    
    return G

def save_graph_json(G: nx.Graph, file_path: str):
    graph_data = nx.node_link_data(G)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(graph_data, file, indent=2)

def create_knowledge_graph():
    embeddings, _, content = load_embeddings_and_data(EMBEDDINGS_FILE)
    
    if embeddings is None or content is None:
        logging.error(f"Failed to load data from {EMBEDDINGS_FILE}")
        return None

    G = create_networkx_graph(content, embeddings)
    save_graph_json(G, KNOWLEDGE_GRAPH_FILE)
    logging.info(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    logging.info(f"Graph saved as {KNOWLEDGE_GRAPH_FILE}")
    return G

if __name__ == "__main__":
    create_knowledge_graph()
