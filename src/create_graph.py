import networkx as nx
import json
import numpy as np
import logging
from typing import List, Tuple, Dict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import nltk
from src.embeddings_utils import load_embeddings_and_data
from src.settings import settings
from src.api_model import EragAPI, create_erag_api
import os
from tqdm import tqdm
from src.look_and_feel import BLUE, GREEN, RESET, error, success, warning, info

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_graph_settings(similarity_threshold: float, min_entity_occurrence: int):
    settings.similarity_threshold = similarity_threshold
    settings.min_entity_occurrence = min_entity_occurrence
    logging.info(info(f"Graph settings updated: Similarity Threshold={settings.similarity_threshold}, "
                 f"Min Entity Occurrence={settings.min_entity_occurrence}"))

def preprocess_text(text: str) -> str:
    return ' '.join(text.split())

def extract_entities_with_confidence(text: str) -> List[Tuple[str, str, float]]:
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    chunks = ne_chunk(pos_tags)
    
    entities = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entity_text = ' '.join(c[0] for c in chunk)
            entity_type = chunk.label()
            confidence = min(1.0, (len(entity_text) / 10) * (1 if entity_type in ['PERSON', 'ORGANIZATION', 'GPE'] else 0.7))
            entities.append((entity_text, entity_type, confidence))
    
    return entities

def chunk_document(document: str) -> List[str]:
    sentences = sent_tokenize(document)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= settings.graph_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def create_networkx_graph(data: List[str], embeddings: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    entity_count = {}
    
    for doc_idx, document in tqdm(enumerate(data), total=len(data), desc=f"{BLUE}Processing documents{RESET}", colour='blue'):
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
                if entity_count[entity] >= settings.min_entity_occurrence:
                    if not G.has_node(entity):
                        G.add_node(entity, type='entity', entity_type=entity_type, confidence=confidence)
                    G.add_edge(chunk_node, entity, relation='contains', confidence=confidence)
    
    if settings.enable_semantic_edges:
        doc_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'document']
        total_comparisons = len(doc_nodes) * (len(doc_nodes) - 1) // 2
        with tqdm(total=total_comparisons, desc=f"{GREEN}Creating semantic edges{RESET}", colour='green') as pbar:
            for i, node1 in enumerate(doc_nodes):
                for j in range(i+1, len(doc_nodes)):
                    node2 = doc_nodes[j]
                    similarity = cosine_similarity(embeddings[i], embeddings[j])
                    if similarity > settings.similarity_threshold:
                        G.add_edge(node1, node2, relation='similar', weight=similarity, confidence=similarity)
                    pbar.update(1)
    
    return G

def process_raw_document(document: str, erag_api: EragAPI) -> Tuple[List[str], np.ndarray]:
    chunks = chunk_document(document)
    chunk_embeddings = erag_api.encode(chunks)
    return chunks, chunk_embeddings

def create_graph_from_raw(raw_documents: List[str], erag_api: EragAPI) -> nx.Graph:
    G = nx.Graph()
    entity_count = {}
    all_chunk_embeddings = []
    
    for doc_idx, document in tqdm(enumerate(raw_documents), total=len(raw_documents), desc=f"{BLUE}Processing raw documents{RESET}", colour='blue'):
        doc_node = f"doc_{doc_idx}"
        G.add_node(doc_node, type='document', text=document[:1000])  # Store first 1000 chars as preview
        
        chunks, chunk_embeddings = process_raw_document(document, erag_api)
        all_chunk_embeddings.append(chunk_embeddings)
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_node = f"doc_{doc_idx}_chunk_{chunk_idx}"
            G.add_node(chunk_node, type='chunk', text=chunk)
            G.add_edge(doc_node, chunk_node, relation='contains')
            
            entities = extract_entities_with_confidence(chunk)
            for entity, entity_type, confidence in entities:
                entity_count[entity] = entity_count.get(entity, 0) + 1
                if entity_count[entity] >= settings.min_entity_occurrence:
                    if not G.has_node(entity):
                        G.add_node(entity, type='entity', entity_type=entity_type, confidence=confidence)
                    G.add_edge(chunk_node, entity, relation='contains', confidence=confidence)
    
    if settings.enable_semantic_edges:
        doc_embeddings = np.array([np.mean(emb, axis=0) for emb in all_chunk_embeddings])
        total_comparisons = len(raw_documents) * (len(raw_documents) - 1) // 2
        with tqdm(total=total_comparisons, desc=f"{GREEN}Creating semantic edges{RESET}", colour='green') as pbar:
            for i in range(len(raw_documents)):
                for j in range(i+1, len(raw_documents)):
                    similarity = cosine_similarity(doc_embeddings[i], doc_embeddings[j])
                    if similarity > settings.similarity_threshold:
                        G.add_edge(f"doc_{i}", f"doc_{j}", relation='similar', weight=similarity, confidence=similarity)
                    pbar.update(1)
    
    return G

def save_graph_json(G: nx.Graph, file_path: str):
    graph_data = nx.node_link_data(G)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(graph_data, file, indent=2)

def create_knowledge_graph():
    embeddings_file_path = os.path.join(settings.output_folder, os.path.basename(settings.embeddings_file_path))
    embeddings, _, content = load_embeddings_and_data(embeddings_file_path)
    
    if embeddings is None or content is None:
        logging.error(error(f"Failed to load data from {embeddings_file_path}"))
        return None

    G = create_networkx_graph(content, embeddings)
    knowledge_graph_file_path = os.path.join(settings.output_folder, os.path.basename(settings.knowledge_graph_file_path))
    save_graph_json(G, knowledge_graph_file_path)
    logging.info(success(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."))
    logging.info(success(f"Graph saved as {knowledge_graph_file_path}"))
    return G

def create_knowledge_graph_from_raw(raw_file_path: str):
    erag_api = create_erag_api(settings.api_type, settings.ollama_model)  # Adjust as needed
    
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        raw_documents = f.read().split("---DOCUMENT_SEPARATOR---")
        raw_documents = [doc.strip() for doc in raw_documents if doc.strip()]
    
    G = create_graph_from_raw(raw_documents, erag_api)
    knowledge_graph_file_path = os.path.join(settings.output_folder, os.path.basename(settings.knowledge_graph_file_path))
    save_graph_json(G, knowledge_graph_file_path)
    logging.info(success(f"NetworkX graph created from raw documents with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."))
    logging.info(success(f"Graph saved as {knowledge_graph_file_path}"))
    return G

if __name__ == "__main__":
    create_knowledge_graph()