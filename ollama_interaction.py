import torch
from sentence_transformers import SentenceTransformer, util
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
def ollama_chat(user_input, system_message, db_embeddings, db_content, model, client, context=None):
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
    # Return the content of the response from the model and the updated context
    return response.choices[0].message.content, user_input_with_context
