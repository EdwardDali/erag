import torch
from sentence_transformers import SentenceTransformer
import os
from typing import List, Tuple, Optional, Union
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"
DB_FILE = "db.txt"
EMBEDDINGS_FILE = "db_embeddings.pt"

def compute_and_save_embeddings(
    db_content: List[str],
    model: SentenceTransformer,
    save_path: str,
    batch_size: int = 32
) -> None:
    try:
        logging.info(f"Computing embeddings for {len(db_content)} items")
        db_embeddings = []
        for i in range(0, len(db_content), batch_size):
            batch = db_content[i:i+batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            db_embeddings.append(batch_embeddings)
            logging.info(f"Processed batch {i//batch_size + 1}/{len(db_content)//batch_size + 1}")

        db_embeddings = torch.cat(db_embeddings, dim=0)
        logging.info(f"Final embeddings shape: {db_embeddings.shape}")
        
        indexes = torch.arange(len(db_content))
        data_to_save = {'embeddings': db_embeddings, 'indexes': indexes}
        
        torch.save(data_to_save, save_path)
        logging.info(f"Embeddings and indexes saved to {save_path}")
        
        # Verify saved data
        loaded_data = torch.load(save_path)
        logging.info(f"Verified saved embeddings shape: {loaded_data['embeddings'].shape}")
        logging.info(f"Verified saved indexes shape: {loaded_data['indexes'].shape}")
    except Exception as e:
        logging.error(f"Error in compute_and_save_embeddings: {str(e)}")
        raise

def load_embeddings_and_indexes(embeddings_file: str) -> Tuple[Optional[Union[torch.Tensor, np.ndarray]], Optional[torch.Tensor]]:
    try:
        if os.path.exists(embeddings_file):
            data = torch.load(embeddings_file)
            embeddings = data['embeddings']
            indexes = data['indexes']
            logging.info(f"Loaded embeddings shape: {embeddings.shape}")
            logging.info(f"Loaded indexes shape: {indexes.shape}")
            return embeddings, indexes
        else:
            logging.warning(f"Embeddings file {embeddings_file} not found")
            return None, None
    except Exception as e:
        logging.error(f"Error in load_embeddings_and_indexes: {str(e)}")
        raise

def load_or_compute_embeddings(model: SentenceTransformer) -> Tuple[torch.Tensor, torch.Tensor]:
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
