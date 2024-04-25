import sys
import os
from sentence_transformers import SentenceTransformer
from api_connectivity import configure_api
from ollama_interaction import ollama_chat, get_relevant_context
from embeddings_utils import compute_and_save_embeddings, load_embeddings_and_indexes
from colors import PINK, CYAN, YELLOW, NEON_GREEN, RESET_COLOR

# How to use:
def main(api_type):
    client = configure_api(api_type)
    
    # Load the model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load or compute and save embeddings
    db_content = []
    embeddings_file = "db_embeddings.pt"
    if os.path.exists("db.txt"):
        with open("db.txt", "r", encoding='utf-8') as db_file:
            db_content = db_file.readlines()

    db_embeddings, db_indexes = load_embeddings_and_indexes("db_embeddings.pt")
    if db_embeddings is None:
        compute_and_save_embeddings(db_content, model, embeddings_file)
        db_embeddings = load_embeddings(embeddings_file)

    # Example usage
    context = None  # Initialize context
    while True:
        user_input = input(YELLOW + "Ask a question about your documents: " + RESET_COLOR)
        options= {"temperature": 0.1}
        system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Reply with 'I don't see any relevant info in the context' if the given text does not provide the correct answer"
        response, context = ollama_chat(user_input, system_message, db_embeddings, db_content, model, client, context=context)
        print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
    else:
        print("Error: No API type provided.")
        sys.exit(1)
    main(api_type)
