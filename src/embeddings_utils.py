import torch
from sentence_transformers import SentenceTransformer
import os
from typing import List, Tuple, Optional
import logging
from src.settings import settings
from src.api_model import configure_api, LlamaClient
from src.color_scheme import Colors, colorize
import colorama

# Initialize colorama
colorama.init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_output_folder():
    os.makedirs(settings.output_folder, exist_ok=True)
    print(colorize(f"Output folder ensured: {settings.output_folder}", Colors.INFO))

def load_db_content(file_path: str) -> List[str]:
    content = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding='utf-8') as db_file:
            content = [line.strip() for line in db_file]
        print(colorize(f"Loaded {len(content)} lines from {file_path}", Colors.SUCCESS))
    else:
        print(colorize(f"File not found: {file_path}", Colors.WARNING))
    return content

def compute_and_save_embeddings(
    model: SentenceTransformer,
    save_path: str,
    content: List[str]
) -> None:
    try:
        ensure_output_folder()
        
        # Ensure save_path is in the output folder
        save_path = os.path.join(settings.output_folder, os.path.basename(save_path))
        
        print(colorize(f"Computing embeddings for {len(content)} items", Colors.INFO))
        db_embeddings = []
        for i in range(0, len(content), settings.batch_size):
            batch = content[i:i+settings.batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            db_embeddings.append(batch_embeddings)
            print(colorize(f"Processed batch {i//settings.batch_size + 1}/{(len(content)-1)//settings.batch_size + 1}", Colors.INFO))

        db_embeddings = torch.cat(db_embeddings, dim=0)
        print(colorize(f"Final embeddings shape: {db_embeddings.shape}", Colors.SUCCESS))
        
        indexes = torch.arange(len(content))
        data_to_save = {
            'embeddings': db_embeddings, 
            'indexes': indexes,
            'content': content
        }
        
        torch.save(data_to_save, save_path)
        print(colorize(f"Embeddings, indexes, and content saved to {save_path}", Colors.SUCCESS))
        
        # Verify saved data
        loaded_data = torch.load(save_path)
        print(colorize(f"Verified saved embeddings shape: {loaded_data['embeddings'].shape}", Colors.SUCCESS))
        print(colorize(f"Verified saved indexes shape: {loaded_data['indexes'].shape}", Colors.SUCCESS))
        print(colorize(f"Verified saved content length: {len(loaded_data['content'])}", Colors.SUCCESS))
    except Exception as e:
        print(colorize(f"Error in compute_and_save_embeddings: {str(e)}", Colors.ERROR))
        raise

def load_embeddings_and_data(embeddings_file: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]]]:
    try:
        # Ensure embeddings_file is in the output folder
        embeddings_file = os.path.join(settings.output_folder, os.path.basename(embeddings_file))
        
        if os.path.exists(embeddings_file):
            data = torch.load(embeddings_file)
            embeddings = data['embeddings']
            indexes = data['indexes']
            content = data['content']
            print(colorize(f"Loaded embeddings shape: {embeddings.shape}", Colors.SUCCESS))
            print(colorize(f"Loaded indexes shape: {indexes.shape}", Colors.SUCCESS))
            print(colorize(f"Loaded content length: {len(content)}", Colors.SUCCESS))
            return embeddings, indexes, content
        else:
            print(colorize(f"Embeddings file {embeddings_file} not found", Colors.WARNING))
            return None, None, None
    except Exception as e:
        print(colorize(f"Error in load_embeddings_and_data: {str(e)}", Colors.ERROR))
        raise

def load_or_compute_embeddings(model: SentenceTransformer, db_file: str, embeddings_file: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    try:
        # Ensure db_file and embeddings_file are in the output folder
        db_file = os.path.join(settings.output_folder, os.path.basename(db_file))
        embeddings_file = os.path.join(settings.output_folder, os.path.basename(embeddings_file))
        
        embeddings, indexes, content = load_embeddings_and_data(embeddings_file)
        if embeddings is None or indexes is None or content is None:
            print(colorize("Embeddings not found. Computing new embeddings...", Colors.INFO))
            content = load_db_content(db_file)
            compute_and_save_embeddings(model, embeddings_file, content)
            embeddings, indexes, content = load_embeddings_and_data(embeddings_file)
        return embeddings, indexes, content
    except Exception as e:
        print(colorize(f"Error in load_or_compute_embeddings: {str(e)}", Colors.ERROR))
        raise

if __name__ == "__main__":
    try:
        ensure_output_folder()
        # Use the configure_api function to get the appropriate model
        if settings.api_type == "llama":
            model = LlamaClient()
        else:
            model = configure_api("sentence_transformer", settings.model_name)
        
        print(colorize("Processing db.txt from the output folder", Colors.INFO))
        embeddings, indexes, content = load_or_compute_embeddings(model, settings.db_file_path, settings.embeddings_file_path)
        print(colorize(f"DB Embeddings shape: {embeddings.shape}, Indexes shape: {indexes.shape}", Colors.SUCCESS))
        print(colorize(f"DB Content length: {len(content)}", Colors.SUCCESS))
    except Exception as e:
        print(colorize(f"Error in main execution: {str(e)}", Colors.ERROR))
