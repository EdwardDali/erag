import sys
import os
import torch
from sentence_transformers import SentenceTransformer, util
from api_connectivity import configure_api
from embeddings_utils import compute_and_save_embeddings, load_embeddings_and_indexes
from colors import PINK, CYAN, YELLOW, NEON_GREEN, RESET_COLOR

# Function to get relevant context from the db based on user input
def get_relevant_context(user_input, db_embeddings, db_content, model, top_k=5):
    if db_embeddings is None or len(db_embeddings) == 0:
        return []
    # Encode the user input
    input_embedding = model.encode([user_input])
    # Compute cosine similarity between the input and db embeddings
    cos_scores = util.cos_sim(input_embedding, torch.tensor(db_embeddings))[0]
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the db
    relevant_context = [db_content[idx].strip() for idx in top_indices]
    return relevant_context

# Function to interact with the Ollama model
def ollama_chat(user_input, system_message, db_embeddings, db_content, model, client, conversation_history, context=None):
    # Get relevant context from the db
    relevant_context = get_relevant_context(user_input, db_embeddings, db_content, model)
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    # Prepare the user's input by concatenating it with the relevant context and previous context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    if context:
        user_input_with_context = context + "\n\n" + user_input_with_context

    # Create a message history including the system message and the user's input with context
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input_with_context}
    ]
    # Send the completion request to the Ollama model
    response = client.chat.completions.create(
        model="phi3:instruct",
        messages=messages
    )
    # Return the content of the response from the model
    return response.choices[0].message.content

def create_session_embeddings(conversation_history, model):
    # Extract text content from the conversation history
    text_content = [msg['content'] for msg in conversation_history]

    # Encode the text content using the SentenceTransformer model
    embeddings = model.encode(text_content)

    return embeddings

def save_embeddings(embeddings, filepath):
    # Save the embeddings to the specified file
    torch.save(embeddings, filepath)

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

    # Initialize conversation history
    conversation_history = []

    while True:
        user_input = input(YELLOW + "Ask a question about your documents: " + RESET_COLOR)
        system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Reply with 'I don't see any relevant info in the context' if the given text does not provide the correct answer. The chunks of information provided could be unrelated to one another or with the question"
        response = ollama_chat(user_input, system_message, db_embeddings, db_content, model, client, conversation_history)
        print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)

        # Update conversation history with user's question and LLM's response
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        # Create embeddings from the conversation history
        session_embeddings = create_session_embeddings(conversation_history, model)

        # Save embeddings to a file
        save_embeddings(session_embeddings, "sessions.pt")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
    else:
        print("Error: No API type provided.")
        sys.exit(1)

    # Call main function
    main(api_type)

