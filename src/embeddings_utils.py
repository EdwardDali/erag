# Standard library imports
import logging
import os
from typing import List, Optional, Tuple

# Third-party imports
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Local imports
from src.api_model import EragAPI
from src.look_and_feel import error, info, success, warning
from src.settings import settings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_output_folder():
    os.makedirs(settings.output_folder, exist_ok=True)

def load_db_content(file_path: str) -> List[str]:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding='utf-8') as db_file:
            return [line.strip() for line in db_file]
    return []

def compute_and_save_embeddings(model, save_path: str, content: List[str]) -> None:
    try:
        ensure_output_folder()
        save_path = os.path.join(settings.output_folder, os.path.basename(save_path))
        
        print(info(f"Computing embeddings for {len(content)} items using {type(model).__name__}"))
        db_embeddings = []
        for i in range(0, len(content), settings.batch_size):
            batch = content[i:i+settings.batch_size]
            batch_embeddings = model._encode(batch) if isinstance(model, EragAPI) else model.encode(batch, convert_to_tensor=True)
            db_embeddings.append(batch_embeddings)
            print(info(f"Processed batch {i//settings.batch_size + 1}/{(len(content)-1)//settings.batch_size + 1}"))

        db_embeddings = torch.cat(db_embeddings, dim=0)
        print(info(f"Final embeddings shape: {db_embeddings.shape}"))
        
        data_to_save = {
            'embeddings': db_embeddings,
            'indexes': torch.arange(len(content)),
            'content': content
        }
        
        torch.save(data_to_save, save_path)
        print(success(f"Embeddings, indexes, and content saved to {save_path}"))
        
        # Verify saved data
        loaded_data = torch.load(save_path)
        print(info(f"Verified saved embeddings shape: {loaded_data['embeddings'].shape}"))
        print(info(f"Verified saved indexes shape: {loaded_data['indexes'].shape}"))
        print(info(f"Verified saved content length: {len(loaded_data['content'])}"))
    except Exception as e:
        print(error(f"Error in compute_and_save_embeddings: {str(e)}"))
        raise

def load_embeddings_and_data(embeddings_file: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]]]:
    try:
        embeddings_file = os.path.join(settings.output_folder, os.path.basename(embeddings_file))
        
        if os.path.exists(embeddings_file):
            data = torch.load(embeddings_file)
            embeddings = torch.as_tensor(data['embeddings'])
            indexes = torch.as_tensor(data['indexes'])
            content = data['content']
            
            print(info(f"Loaded embeddings shape: {embeddings.shape}"))
            print(info(f"Loaded indexes shape: {indexes.shape}"))
            print(info(f"Loaded content length: {len(content)}"))
            return embeddings, indexes, content
        else:
            print(warning(f"Embeddings file {embeddings_file} not found"))
            return None, None, None
    except Exception as e:
        print(error(f"Error in load_embeddings_and_data: {str(e)}"))
        raise

def load_or_compute_embeddings(model, db_file: str, embeddings_file: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    try:
        db_file = os.path.join(settings.output_folder, os.path.basename(db_file))
        embeddings_file = os.path.join(settings.output_folder, os.path.basename(embeddings_file))
        
        embeddings, indexes, content = load_embeddings_and_data(embeddings_file)
        if embeddings is None or indexes is None or content is None:
            content = load_db_content(db_file)
            compute_and_save_embeddings(model, embeddings_file, content)
            embeddings, indexes, content = load_embeddings_and_data(embeddings_file)
        return embeddings, indexes, content
    except Exception as e:
        print(error(f"Error in load_or_compute_embeddings: {str(e)}"))
        raise

if __name__ == "__main__":
    try:
        ensure_output_folder()
        model = (EragAPI(settings.api_type, embedding_class=settings.default_embedding_class, embedding_model=settings.default_embedding_model)
                 if settings.default_embedding_class == "ollama" else
                 SentenceTransformer(settings.default_embedding_model))
        
        embeddings, indexes, content = load_or_compute_embeddings(model, settings.db_file_path, settings.embeddings_file_path)
        print(success(f"DB Embeddings shape: {embeddings.shape}, Indexes shape: {indexes.shape}"))
        print(success(f"DB Content length: {len(content)}"))
    except Exception as e:
        print(error(f"Error in main execution: {str(e)}"))