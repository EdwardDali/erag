import spacy
import networkx as nx
import json
from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging
from typing import List, Tuple, Dict
from nltk.tokenize import sent_tokenize
import nltk
from embeddings_utils import load_embeddings_and_data

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def preprocess_text(text: str) -> str:
    # Remove extra whitespace and newline characters
    return ' '.join(text.split())

def extract_entities_and_relations(text: str) -> List[Tuple[str, str, str]]:
    doc = nlp(text)
    triples = []
    
    for sent in doc.sents:
        entities = [ent for ent in sent.ents]
        for i, entity in enumerate(entities):
            for j in range(i+1, len(entities)):
                # Find the shortest dependency path between the two entities
                path = list(entity.root.subtree) + list(entities[j].root.subtree)
                path = sorted(set(path), key=lambda x: x.i)
                relation = ' '.join([token.lemma_ for token in path if token.pos_ in ['VERB', 'ADP']])
                if relation:
                    triples.append((entity.text, relation, entities[j].text))
    
    return triples

def create_networkx_graph(data: List[str], references: List[Dict[str, str]], embeddings: torch.Tensor) -> nx.Graph:
    G = nx.Graph()
    
    for i, (chunk, ref) in enumerate(zip(data, references)):
        chunk = preprocess_text(chunk)
        sentences = sent_tokenize(chunk)
        
        for sent in sentences:
            triples = extract_entities_and_relations(sent)
            for subj, rel, obj in triples:
                if not G.has_node(subj):
                    G.add_node(subj, type='entity')
                if not G.has_node(obj):
                    G.add_node(obj, type='entity')
                G.add_edge(subj, obj, relation=rel)
        
        # Add document node and connect to entities
        doc_node = f"doc_{i}"
        G.add_node(doc_node, type='document', text=chunk, reference=ref)
        entities = set([ent for triple in triples for ent in [triple[0], triple[2]]])
        for entity in entities:
            G.add_edge(doc_node, entity, relation='contains')
    
    # Add semantic similarity edges between document nodes
    doc_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'document']
    for i, node1 in enumerate(doc_nodes):
        for j in range(i+1, len(doc_nodes)):
            node2 = doc_nodes[j]
            similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            if similarity > 0.5:  # Adjust this threshold as needed
                G.add_edge(node1, node2, relation='similar', weight=similarity)
    
    return G

def prune_graph(G: nx.Graph, min_edge_weight: float = 0.5) -> nx.Graph:
    edges_to_remove = [(u, v) for (u, v, d) in G.edges(data=True) 
                       if 'weight' in d and d['weight'] < min_edge_weight]
    G.remove_edges_from(edges_to_remove)
    return G

def save_graph_json(G: nx.Graph, file_path: str):
    # Convert the graph to a dictionary
    graph_data = nx.node_link_data(G)
    
    # Save the dictionary as a JSON file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(graph_data, file, indent=2)

def create_knowledge_graph():
    # Load data from db_embeddings_r.pt
    embeddings, _, content, references = load_embeddings_and_data("db_embeddings_r.pt")
    
    if embeddings is None or content is None or references is None:
        logging.error("Failed to load data from db_embeddings_r.pt. Make sure the file exists and is properly formatted.")
        return None

    # Create and save NetworkX graph
    G = create_networkx_graph(content, references, embeddings)
    G = prune_graph(G)
    save_graph_json(G, "knowledge_graph_r.json")
    logging.info(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    logging.info("Graph saved as knowledge_graph_r.json")
    return G

if __name__ == "__main__":
    create_knowledge_graph()
