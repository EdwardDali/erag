import sys
import os
import torch
from sentence_transformers import SentenceTransformer, util
from api_connectivity import configure_api
from embeddings_utils import load_or_compute_embeddings
from colors import Colors
import logging
from typing import List, Dict
import numpy as np
import torch
from sentence_transformers import util
from typing import List, Union
import re
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Constants
MAX_HISTORY_LENGTH = 5
EMBEDDINGS_FILE = "db_embeddings.pt"
DB_FILE = "db.txt"
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "phi3:instruct"


def lexical_search(query: str, db_content: List[str], top_k: int = 5) -> List[str]:
    """
    Perform a simple lexical search based on word overlap.
    
    Args:
        query (str): The user's query.
        db_content (List[str]): The database content to search through.
        top_k (int): The number of top results to return.
    
    Returns:
        List[str]: The top_k most relevant contexts based on lexical overlap.
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    
    overlap_scores = []
    for context in db_content:
        context_words = set(re.findall(r'\w+', context.lower()))
        overlap = len(query_words.intersection(context_words))
        overlap_scores.append(overlap)
    
    top_indices = sorted(range(len(overlap_scores)), key=lambda i: overlap_scores[i], reverse=True)[:top_k]
    
    return [db_content[i].strip() for i in top_indices]

def semantic_search(query: str, db_embeddings: Union[np.ndarray, torch.Tensor], db_content: List[str], model: SentenceTransformer, top_k: int = 5) -> List[str]:
    """
    Perform a semantic search based on embedding similarity.
    
    Args:
        query (str): The user's query.
        db_embeddings (Union[np.ndarray, torch.Tensor]): The pre-computed embeddings of the database content.
        db_content (List[str]): The database content to search through.
        model (SentenceTransformer): The sentence transformer model.
        top_k (int): The number of top results to return.
    
    Returns:
        List[str]: The top_k most relevant contexts based on semantic similarity.
    """
    if isinstance(db_embeddings, np.ndarray):
        db_embeddings = torch.from_numpy(db_embeddings)
    
    input_embedding = model.encode([query], convert_to_tensor=True)
    cos_scores = util.cos_sim(input_embedding, db_embeddings)[0]
    top_indices = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))[1].tolist()
    
    return [db_content[idx].strip() for idx in top_indices]

def get_relevant_context(user_input: str, db_embeddings: Union[np.ndarray, torch.Tensor], db_content: List[str], model: SentenceTransformer, top_k: int = 5) -> tuple[List[str], List[str]]:
    logging.info(f"DB Embeddings type: {type(db_embeddings)}")
    logging.info(f"DB Embeddings shape: {db_embeddings.shape if hasattr(db_embeddings, 'shape') else 'No shape attribute'}")
    logging.info(f"DB Content length: {len(db_content)}")
    
    if isinstance(db_embeddings, np.ndarray):
        db_embeddings = torch.from_numpy(db_embeddings)
    
    if db_embeddings.numel() == 0 or len(db_content) == 0:
        logging.warning("DB Embeddings or DB Content is empty")
        return [], []
    
    # Perform lexical search
    lexical_results = lexical_search(user_input, db_content, top_k)
    
    # Perform semantic search
    semantic_results = semantic_search(user_input, db_embeddings, db_content, model, top_k)
    
    logging.info(f"Number of lexical results: {len(lexical_results)}")
    logging.info(f"Number of semantic results: {len(semantic_results)}")
    
    return lexical_results, semantic_results

def ollama_chat(user_input: str, system_message: str, db_embeddings: Union[np.ndarray, torch.Tensor], db_content: List[str], model: SentenceTransformer, client, conversation_history: List[Dict[str, str]]) -> str:
    lexical_context, semantic_context = get_relevant_context(user_input, db_embeddings, db_content, model)
    
    lexical_str = "\n".join(lexical_context) if lexical_context else "No relevant lexical context found."
    semantic_str = "\n".join(semantic_context) if semantic_context else "No relevant semantic context found."
    
    context_str = f"Lexical Search Results:\n{lexical_str}\n\nSemantic Search Results:\n{semantic_str}"
    
    logging.info(f"Context pulled: {context_str[:200]}...")

    messages = [
        {"role": "system", "content": system_message},
        *conversation_history,
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_input}\n\nPlease use the most relevant information from either the lexical or semantic search results to answer the question. If neither set of results is relevant, you can say so and answer based on your general knowledge."}
    ]

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in API call: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request."

def append_to_db(new_messages: List[Dict[str, str]], filepath: str):
    with open(filepath, "a", encoding='utf-8') as file:
        for msg in new_messages:
            file.write(f"{msg['role']}: {msg['content']}\n")

def main(api_type: str):
    client = configure_api(api_type)
    model = SentenceTransformer(MODEL_NAME)

    db_content = []
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding='utf-8') as db_file:
            db_content = db_file.readlines()

    db_embeddings, _ = load_or_compute_embeddings(model)
    
    logging.info(f"Loaded DB Embeddings shape: {db_embeddings.shape if hasattr(db_embeddings, 'shape') else 'No shape attribute'}")
    logging.info(f"Loaded DB Content length: {len(db_content)}")

    conversation_history: List[Dict[str, str]] = []

    system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Not all info provided in context is useful. Reply with 'I don't see any relevant info in the context' if the given text does not provide the correct answer. Try to provide a comprehensive answer considering also what you can deduce from information provided as well as what you know already."

    print(f"{Colors.YELLOW.value}Welcome to the RAG system. Type 'exit' to quit or 'clear' to clear conversation history.{Colors.RESET.value}")

    while True:
        user_input = input(f"{Colors.YELLOW.value}Ask a question about your documents: {Colors.RESET.value}").strip()
        
        if user_input.lower() == 'exit':
            print(f"{Colors.NEON_GREEN.value}Thank you for using the RAG system. Goodbye!{Colors.RESET.value}")
            break
        elif user_input.lower() == 'clear':
            conversation_history.clear()
            print(f"{Colors.CYAN.value}Conversation history cleared.{Colors.RESET.value}")
            continue

        if not user_input:
            print(f"{Colors.PINK.value}Please enter a valid question.{Colors.RESET.value}")
            continue

        response = ollama_chat(user_input, system_message, db_embeddings, db_content, model, client, conversation_history)
        print(f"{Colors.NEON_GREEN.value}Response: \n\n{response}{Colors.RESET.value}")

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        # Keep only the last MAX_HISTORY_LENGTH exchanges
        if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
            conversation_history = conversation_history[-MAX_HISTORY_LENGTH * 2:]

        append_to_db(conversation_history[-2:], DB_FILE)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        main(api_type)
    else:
        print("Error: No API type provided.")
        sys.exit(1)
