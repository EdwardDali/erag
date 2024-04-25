import torch
from sentence_transformers import SentenceTransformer
import os

# Function to compute and save both embeddings and their corresponding indexes
def compute_and_save_embeddings(db_content, model, save_path):
    db_embeddings = model.encode(db_content)
    indexes = torch.arange(len(db_content))  # Create a tensor of indices
    data_to_save = {'embeddings': db_embeddings, 'indexes': indexes}  # Combine embeddings and indexes into a dictionary
    torch.save(data_to_save, save_path)

    # Save the indexes to a separate file
    indexes_file = "indexes.pt"
    torch.save(indexes, indexes_file)

# Function to load both embeddings and their corresponding indexes from file
def load_embeddings_and_indexes(embeddings_file):
    if os.path.exists(embeddings_file):
        data = torch.load(embeddings_file)
        return data['embeddings'], data['indexes']  # Return the embeddings and indexes as separate variables
    else:
        return None, None

# Example usage
if __name__ == "__main__":
    # Load the model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load the db content
    db_content = []
    if os.path.exists("db.txt"):
        with open("db.txt", "r", encoding='utf-8') as db_file:
            db_content = db_file.readlines()

    # Define the path to save the embeddings and indexes
    embeddings_file = "db_embeddings.pt"

    # Compute and save embeddings if not already computed and saved
    if not os.path.exists(embeddings_file):
        compute_and_save_embeddings(db_content, model, embeddings_file)
