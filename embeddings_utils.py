import torch
from sentence_transformers import SentenceTransformer
import os
from typing import List, Tuple, Optional
import logging
from settings import settings
from api_model import configure_api

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        for i in range(0, len(content), settings.batch_size):
            batch = content[i:i+settings.batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            db_embeddings.append(batch_embeddings)
            logging.info(f"Processed batch {i//settings.batch_size + 1}/{(len(content)-1)//settings.batch_size + 1}")

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
        # Use the configure_api function to get the appropriate model
        model = configure_api("sentence_transformer", settings.model_name)
        
        # Process db.txt
        embeddings, indexes, content = load_or_compute_embeddings(model, settings.db_file_path, settings.embeddings_file_path)
        logging.info(f"DB Embeddings shape: {embeddings.shape}, Indexes shape: {indexes.shape}")
        logging.info(f"DB Content length: {len(content)}")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
