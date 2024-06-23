import spacy
import networkx as nx
import json
from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging
from neo4j import GraphDatabase

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Neo4j connection details
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")  # Replace with your Neo4j password

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

def create_neo4j_graph(data, embeddings):
    if len(data) != len(embeddings):
        raise ValueError(f"Mismatch between number of data items ({len(data)}) and embeddings ({len(embeddings)})")

    logging.info(f"Creating knowledge graph with {len(data)} items")
    
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with driver.session() as session:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")

                # Create nodes for each chunk of text
                for i, chunk in enumerate(data):
                    entities = extract_entities(chunk)
                    session.run(
                        "CREATE (c:Chunk {id: $id, text: $text, entities: $entities})",
                        id=i, text=chunk, entities=entities
                    )

                # Create edges based on similarity
                for i in range(len(data)):
                    for j in range(i+1, len(data)):
                        similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
                        if similarity > 0.5:  # Adjust this threshold as needed
                            session.run(
                                """
                                MATCH (c1:Chunk {id: $id1})
                                MATCH (c2:Chunk {id: $id2})
                                CREATE (c1)-[r:SIMILAR {weight: $weight}]->(c2)
                                """,
                                id1=i, id2=j, weight=similarity
                            )
        logging.info("Knowledge graph created successfully in Neo4j database.")
    except Exception as e:
        logging.error(f"An error occurred while creating the Neo4j graph: {str(e)}")
        raise

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
    
    # Create Neo4j graph
    create_neo4j_graph(data, embeddings)
    print("Knowledge graph created in Neo4j database.")

if __name__ == "__main__":
    main()
