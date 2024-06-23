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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_HISTORY_LENGTH = 5
EMBEDDINGS_FILE = "db_embeddings.pt"
DB_FILE = "db.txt"
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "phi3:instruct"

def get_relevant_context(user_input: str, db_embeddings: Union[np.ndarray, torch.Tensor], db_content: List[str], model: SentenceTransformer, top_k: int = 5) -> List[str]:
    if isinstance(db_embeddings, np.ndarray):
        db_embeddings = torch.from_numpy(db_embeddings)
    
    if db_embeddings.numel() == 0 or len(db_content) == 0:
        return []
    
    input_embedding = model.encode([user_input], convert_to_tensor=True)
    cos_scores = util.cos_sim(input_embedding, db_embeddings)[0]
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    return [db_content[idx].strip() for idx in top_indices]

def ollama_chat(user_input: str, system_message: str, db_embeddings: torch.Tensor, db_content: List[str], model: SentenceTransformer, client, conversation_history: List[Dict[str, str]]) -> str:
    relevant_context = get_relevant_context(user_input, db_embeddings, db_content, model)
    context_str = "\n".join(relevant_context) if relevant_context else ""
    logging.info(f"Context pulled: {context_str[:100]}..." if context_str else "No relevant context found.")

    messages = [
        {"role": "system", "content": system_message},
        *conversation_history,
        {"role": "user", "content": f"{context_str}\n\n{user_input}".strip()}
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
