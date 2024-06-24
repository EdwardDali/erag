import spacy
import networkx as nx
import json
from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def create_networkx_graph(data, embeddings):
    G = nx.Graph()
    
    # Create nodes for each chunk of text
    for i, chunk in enumerate(data):
        entities = extract_entities(chunk)
        G.add_node(i, text=chunk, entities=entities)
    
    # Create edges based on similarity
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            if similarity > 0.5:  # Adjust this threshold as needed
                G.add_edge(i, j, weight=similarity)
    
    return G

def save_graph_json(G, file_path):
    # Convert the graph to a dictionary
    graph_data = nx.node_link_data(G)
    
    # Save the dictionary as a JSON file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(graph_data, file, indent=2)

def main():
    # Load data
    data = load_data("db.txt")
    
    # Load or compute embeddings
    if os.path.exists("db_embeddings.pt"):
        embeddings = torch.load("db_embeddings.pt")['embeddings']
    else:
        embeddings = model.encode(data, convert_to_tensor=True)
        torch.save({'embeddings': embeddings}, "db_embeddings.pt")
    
    # Create and save NetworkX graph
    G = create_networkx_graph(data, embeddings)
    save_graph_json(G, "knowledge_graph.json")
    print(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Graph saved as knowledge_graph.json")

if __name__ == "__main__":
    main()
