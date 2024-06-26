import torch
from sentence_transformers import SentenceTransformer
import os
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variables for settings (previously constants)
MODEL_NAME = "all-MiniLM-L6-v2"
DB_FILE = "db.txt"
EMBEDDINGS_FILE = "db_embeddings.pt"
BATCH_SIZE = 32

def set_model_name(new_model_name: str):
    global MODEL_NAME
    MODEL_NAME = new_model_name
    logging.info(f"Model name set to {MODEL_NAME}")

def set_db_file(new_db_file: str):
    global DB_FILE
    DB_FILE = new_db_file
    logging.info(f"Database file set to {DB_FILE}")

def set_embeddings_file(new_embeddings_file: str):
    global EMBEDDINGS_FILE
    EMBEDDINGS_FILE = new_embeddings_file
    logging.info(f"Embeddings file set to {EMBEDDINGS_FILE}")

def set_batch_size(new_batch_size: int):
    global BATCH_SIZE
    BATCH_SIZE = new_batch_size
    logging.info(f"Batch size set to {BATCH_SIZE}")

def set_batch_size(new_batch_size: int):
    global BATCH_SIZE
    BATCH_SIZE = new_batch_size
    logging.info(f"Batch size set to {BATCH_SIZE}")

def load_db_content(file_path: str) -> List[str]:
    content = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding='utf-8') as db_file:
            content = [line.strip() for line in db_file]
    return content

def compute_and_save_embeddings(
    model: SentenceTransformer,
    save_path: str,
    content: List[str]
) -> None:
    try:
        logging.info(f"Computing embeddings for {len(content)} items")
        db_embeddings = []
        for i in range(0, len(content), BATCH_SIZE):
            batch = content[i:i+BATCH_SIZE]
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            db_embeddings.append(batch_embeddings)
            logging.info(f"Processed batch {i//BATCH_SIZE + 1}/{(len(content)-1)//BATCH_SIZE + 1}")

        db_embeddings = torch.cat(db_embeddings, dim=0)
        logging.info(f"Final embeddings shape: {db_embeddings.shape}")
        
        indexes = torch.arange(len(content))
        data_to_save = {
            'embeddings': db_embeddings, 
            'indexes': indexes,
            'content': content
        }
        
        torch.save(data_to_save, save_path)
        logging.info(f"Embeddings, indexes, and content saved to {save_path}")
        
        # Verify saved data
        loaded_data = torch.load(save_path)
        logging.info(f"Verified saved embeddings shape: {loaded_data['embeddings'].shape}")
        logging.info(f"Verified saved indexes shape: {loaded_data['indexes'].shape}")
        logging.info(f"Verified saved content length: {len(loaded_data['content'])}")
    except Exception as e:
        logging.error(f"Error in compute_and_save_embeddings: {str(e)}")
        raise

def load_embeddings_and_data(embeddings_file: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]]]:
    try:
        if os.path.exists(embeddings_file):
            data = torch.load(embeddings_file)
            embeddings = data['embeddings']
            indexes = data['indexes']
            content = data['content']
            logging.info(f"Loaded embeddings shape: {embeddings.shape}")
            logging.info(f"Loaded indexes shape: {indexes.shape}")
            logging.info(f"Loaded content length: {len(content)}")
            return embeddings, indexes, content
        else:
            logging.warning(f"Embeddings file {embeddings_file} not found")
            return None, None, None
    except Exception as e:
        logging.error(f"Error in load_embeddings_and_data: {str(e)}")
        raise

def load_or_compute_embeddings(model: SentenceTransformer, db_file: str, embeddings_file: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    try:
        embeddings, indexes, content = load_embeddings_and_data(embeddings_file)
        if embeddings is None or indexes is None or content is None:
            content = load_db_content(db_file)
            compute_and_save_embeddings(model, embeddings_file, content)
            embeddings, indexes, content = load_embeddings_and_data(embeddings_file)
        return embeddings, indexes, content
    except Exception as e:
        logging.error(f"Error in load_or_compute_embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model = SentenceTransformer(MODEL_NAME)
        
        # Process db.txt
        embeddings, indexes, content = load_or_compute_embeddings(model, DB_FILE, EMBEDDINGS_FILE)
        logging.info(f"DB Embeddings shape: {embeddings.shape}, Indexes shape: {indexes.shape}")
        logging.info(f"DB Content length: {len(content)}")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
