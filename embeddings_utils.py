import torch
from sentence_transformers import SentenceTransformer
import os
from typing import List, Tuple, Optional, Union, Dict
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"
DB_FILE = "db.txt"
DB_R_FILE = "db_r.txt"
EMBEDDINGS_FILE = "db_embeddings.pt"
EMBEDDINGS_R_FILE = "db_embeddings_r.pt"

def load_db_content(file_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    content = []
    references = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding='utf-8') as db_file:
            for line in db_file:
                parts = line.strip().split(" | ")
                if len(parts) >= 2:
                    content.append(parts[0])
                    ref = json.loads(parts[1])
                    references.append(ref)
                else:
                    content.append(line.strip())
                    references.append({})
    return content, references

def compute_and_save_embeddings(
    model: SentenceTransformer,
    save_path: str,
    content: List[str],
    references: List[Dict[str, str]],
    batch_size: int = 32
) -> None:
    try:
        logging.info(f"Computing embeddings for {len(content)} items")
        db_embeddings = []
        for i in range(0, len(content), batch_size):
            batch = content[i:i+batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            db_embeddings.append(batch_embeddings)
            logging.info(f"Processed batch {i//batch_size + 1}/{(len(content)-1)//batch_size + 1}")

        db_embeddings = torch.cat(db_embeddings, dim=0)
        logging.info(f"Final embeddings shape: {db_embeddings.shape}")
        
        indexes = torch.arange(len(content))
        data_to_save = {
            'embeddings': db_embeddings, 
            'indexes': indexes,
            'content': content,
            'references': references
        }
        
        torch.save(data_to_save, save_path)
        logging.info(f"Embeddings, indexes, content, and references saved to {save_path}")
        
        # Verify saved data
        loaded_data = torch.load(save_path)
        logging.info(f"Verified saved embeddings shape: {loaded_data['embeddings'].shape}")
        logging.info(f"Verified saved indexes shape: {loaded_data['indexes'].shape}")
        logging.info(f"Verified saved content length: {len(loaded_data['content'])}")
        logging.info(f"Verified saved references length: {len(loaded_data['references'])}")
    except Exception as e:
        logging.error(f"Error in compute_and_save_embeddings: {str(e)}")
        raise

def load_embeddings_and_data(embeddings_file: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]], Optional[List[Dict[str, str]]]]:
    try:
        if os.path.exists(embeddings_file):
            data = torch.load(embeddings_file)
            embeddings = data['embeddings']
            indexes = data['indexes']
            content = data['content']
            references = data['references']
            logging.info(f"Loaded embeddings shape: {embeddings.shape}")
            logging.info(f"Loaded indexes shape: {indexes.shape}")
            logging.info(f"Loaded content length: {len(content)}")
            logging.info(f"Loaded references length: {len(references)}")
            return embeddings, indexes, content, references
        else:
            logging.warning(f"Embeddings file {embeddings_file} not found")
            return None, None, None, None
    except Exception as e:
        logging.error(f"Error in load_embeddings_and_data: {str(e)}")
        raise

def load_or_compute_embeddings(model: SentenceTransformer, db_file: str, embeddings_file: str) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[Dict[str, str]]]:
    try:
        embeddings, indexes, content, references = load_embeddings_and_data(embeddings_file)
        if embeddings is None or indexes is None or content is None or references is None:
            content, references = load_db_content(db_file)
            compute_and_save_embeddings(model, embeddings_file, content, references)
            embeddings, indexes, content, references = load_embeddings_and_data(embeddings_file)
        return embeddings, indexes, content, references
    except Exception as e:
        logging.error(f"Error in load_or_compute_embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model = SentenceTransformer(MODEL_NAME)
        
        # Process db.txt
        embeddings, indexes, content, references = load_or_compute_embeddings(model, DB_FILE, EMBEDDINGS_FILE)
        logging.info(f"DB Embeddings shape: {embeddings.shape}, Indexes shape: {indexes.shape}")
        logging.info(f"DB Content length: {len(content)}, References length: {len(references)}")
        
        # Process db_r.txt
        embeddings_r, indexes_r, content_r, references_r = load_or_compute_embeddings(model, DB_R_FILE, EMBEDDINGS_R_FILE)
        logging.info(f"DB_R Embeddings shape: {embeddings_r.shape}, Indexes shape: {indexes_r.shape}")
        logging.info(f"DB_R Content length: {len(content_r)}, References length: {len(references_r)}")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
