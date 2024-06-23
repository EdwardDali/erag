import torch
from sentence_transformers import SentenceTransformer
import os
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"
DB_FILE = "db.txt"
EMBEDDINGS_FILE = "db_embeddings.pt"
INDEXES_FILE = "indexes.pt"

def compute_and_save_embeddings(
    db_content: List[str], 
    model: SentenceTransformer, 
    save_path: str,
    batch_size: int = 32
) -> None:
    """
    Compute and save embeddings and their corresponding indexes.

    Args:
        db_content (List[str]): List of text content to encode.
        model (SentenceTransformer): The sentence transformer model.
        save_path (str): Path to save the embeddings and indexes.
        batch_size (int): Batch size for processing large datasets.

    Raises:
        Exception: If there's an error during computation or saving.
    """
    try:
        db_embeddings = []
        for i in range(0, len(db_content), batch_size):
            batch = db_content[i:i+batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            db_embeddings.append(batch_embeddings)

        db_embeddings = torch.cat(db_embeddings, dim=0)
        indexes = torch.arange(len(db_content))
        data_to_save = {'embeddings': db_embeddings, 'indexes': indexes}
        
        torch.save(data_to_save, save_path)
        torch.save(indexes, INDEXES_FILE)
        
        logging.info(f"Embeddings and indexes saved to {save_path} and {INDEXES_FILE}")
    except Exception as e:
        logging.error(f"Error in compute_and_save_embeddings: {str(e)}")
        raise

def load_embeddings_and_indexes(embeddings_file: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load embeddings and their corresponding indexes from file.

    Args:
        embeddings_file (str): Path to the file containing embeddings and indexes.

    Returns:
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: Embeddings and indexes tensors, or (None, None) if file doesn't exist.

    Raises:
        Exception: If there's an error during loading.
    """
    try:
        if os.path.exists(embeddings_file):
            data = torch.load(embeddings_file)
            logging.info(f"Embeddings and indexes loaded from {embeddings_file}")
            return data['embeddings'], data['indexes']
        else:
            logging.warning(f"Embeddings file {embeddings_file} not found")
            return None, None
    except Exception as e:
        logging.error(f"Error in load_embeddings_and_indexes: {str(e)}")
        raise

def load_or_compute_embeddings(model: SentenceTransformer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load existing embeddings or compute new ones if they don't exist.

    Args:
        model (SentenceTransformer): The sentence transformer model.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Embeddings and indexes tensors.

    Raises:
        Exception: If there's an error during loading or computation.
    """
    try:
        embeddings, indexes = load_embeddings_and_indexes(EMBEDDINGS_FILE)
        if embeddings is None or indexes is None:
            db_content = []
            if os.path.exists(DB_FILE):
                with open(DB_FILE, "r", encoding='utf-8') as db_file:
                    db_content = db_file.readlines()
            compute_and_save_embeddings(db_content, model, EMBEDDINGS_FILE)
            embeddings, indexes = load_embeddings_and_indexes(EMBEDDINGS_FILE)
        return embeddings, indexes
    except Exception as e:
        logging.error(f"Error in load_or_compute_embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model = SentenceTransformer(MODEL_NAME)
        embeddings, indexes = load_or_compute_embeddings(model)
        logging.info(f"Embeddings shape: {embeddings.shape}, Indexes shape: {indexes.shape}")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
