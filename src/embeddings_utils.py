import numpy as np
import os
from typing import List, Tuple, Optional
from src.settings import settings
from src.api_model import EragAPI
from src.look_and_feel import success, info, warning, error

def ensure_output_folder():
    os.makedirs(settings.output_folder, exist_ok=True)

def load_db_content(file_path: str) -> List[str]:
    content = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding='utf-8') as db_file:
            content = [line.strip() for line in db_file]
    return content

def compute_and_save_embeddings(
    erag_api: EragAPI,
    save_path: str,
    content: List[str]
) -> None:
    try:
        ensure_output_folder()
        
        # Ensure save_path is in the output folder and has .npy extension
        save_path = os.path.join(settings.output_folder, os.path.basename(save_path))
        if not save_path.endswith('.npy'):
            save_path += '.npy'
        
        print(info(f"Computing embeddings for {len(content)} items"))
        
        # Process all content in a single batch
        db_embeddings = erag_api.encode(content)
        
        print(info(f"Final embeddings shape: {db_embeddings.shape}"))
        
        indexes = np.arange(len(content))
        data_to_save = {
            'embeddings': db_embeddings, 
            'indexes': indexes,
            'content': content
        }
        
        np.save(save_path, data_to_save)
        print(success(f"Embeddings, indexes, and content saved to {save_path}"))
        
        # Verify saved data
        loaded_data = np.load(save_path, allow_pickle=True).item()
        print(info(f"Verified saved embeddings shape: {loaded_data['embeddings'].shape}"))
        print(info(f"Verified saved indexes shape: {loaded_data['indexes'].shape}"))
        print(info(f"Verified saved content length: {len(loaded_data['content'])}"))
    except Exception as e:
        print(error(f"Error in compute_and_save_embeddings: {str(e)}"))
        raise

def load_embeddings_and_data(embeddings_file: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    try:
        # Ensure embeddings_file is in the output folder and has .npy extension
        embeddings_file = os.path.join(settings.output_folder, os.path.basename(embeddings_file))
        if not embeddings_file.endswith('.npy'):
            embeddings_file += '.npy'
        
        if os.path.exists(embeddings_file):
            data = np.load(embeddings_file, allow_pickle=True).item()
            embeddings = data['embeddings']
            indexes = data['indexes']
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

def load_or_compute_embeddings(erag_api: EragAPI, db_file: str, embeddings_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    try:
        # Ensure db_file and embeddings_file are in the output folder
        db_file = os.path.join(settings.output_folder, os.path.basename(db_file))
        embeddings_file = os.path.join(settings.output_folder, os.path.basename(embeddings_file))
        if not embeddings_file.endswith('.npy'):
            embeddings_file += '.npy'
        
        embeddings, indexes, content = load_embeddings_and_data(embeddings_file)
        if embeddings is None or indexes is None or content is None:
            content = load_db_content(db_file)
            compute_and_save_embeddings(erag_api, embeddings_file, content)
            embeddings, indexes, content = load_embeddings_and_data(embeddings_file)
        return embeddings, indexes, content
    except Exception as e:
        print(error(f"Error in load_or_compute_embeddings: {str(e)}"))
        raise

if __name__ == "__main__":
    try:
        ensure_output_folder()
        # Use the EragAPI
        erag_api = EragAPI(settings.api_type)
        
        # Process db.txt from the output folder
        embeddings, indexes, content = load_or_compute_embeddings(erag_api, settings.db_file_path, settings.embeddings_file_path)
        print(success(f"DB Embeddings shape: {embeddings.shape}, Indexes shape: {indexes.shape}"))
        print(success(f"DB Content length: {len(content)}"))
    except Exception as e:
        print(error(f"Error in main execution: {str(e)}"))
