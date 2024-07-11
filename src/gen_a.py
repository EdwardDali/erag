import os
from tqdm import tqdm
from src.settings import settings
from src.api_model import configure_api, LlamaClient
from src.talk2doc import RAGSystem
from src.web_rag import WebRAG
from src.color_scheme import Colors, colorize
import colorama

# Initialize colorama
colorama.init(autoreset=True)

def read_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def generate_answer_talk2doc(rag_system, question):
    response = rag_system.get_response(question)
    return response

def generate_answer_web_rag(web_rag, question):
    response = web_rag.search_and_process(question)
    return response

def generate_answer_hybrid(rag_system, web_rag, question, client):
    talk2doc_response = rag_system.get_response(question)
    web_rag_response = web_rag.search_and_process(question)
    
    hybrid_prompt = f"""Combine and rephrase the following two answers into a comprehensive and coherent response:

Answer 1 (Talk2Doc): {talk2doc_response}

Answer 2 (Web RAG): {web_rag_response}

Combined and rephrased answer:"""

    if isinstance(client, LlamaClient):
        combined_response = client.chat([
            {"role": "system", "content": "You are a helpful assistant that combines and rephrases information from multiple sources into a coherent response."},
            {"role": "user", "content": hybrid_prompt}
        ], temperature=settings.temperature)
    else:
        combined_response = client.chat.completions.create(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that combines and rephrases information from multiple sources into a coherent response."},
                {"role": "user", "content": hybrid_prompt}
            ],
            temperature=settings.temperature
        ).choices[0].message.content

    return combined_response

def run_gen_a(questions_file, gen_method, api_type, client):
    questions = read_questions(questions_file)
    
    output_file = os.path.join(settings.output_folder, f"generated_answers_{gen_method}.txt")
    
    rag_system = None
    web_rag = None
    
    if gen_method in ["talk2doc", "hybrid"]:
        rag_system = RAGSystem(api_type)
    
    if gen_method in ["web_rag", "hybrid"]:
        web_rag = WebRAG(api_type)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, question in enumerate(tqdm(questions, desc="Generating Answers", bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.INFO, Colors.RESET))):
            if gen_method == "talk2doc":
                answer = generate_answer_talk2doc(rag_system, question)
            elif gen_method == "web_rag":
                answer = generate_answer_web_rag(web_rag, question)
            else:  # hybrid
                answer = generate_answer_hybrid(rag_system, web_rag, question, client)
            
            # Write to file immediately after generating each answer
            f.write(f"Question: {question}\n\n")
            f.write(f"Answer: {answer}\n\n")
            f.write("-" * 50 + "\n\n")
            f.flush()  # Ensure the content is written to the file immediately
            
            print(colorize(f"\nProcessed question {i+1}/{len(questions)}", Colors.INFO))

    return colorize(f"Generated answers for {len(questions)} questions using {gen_method} method. Saved to {output_file}", Colors.SUCCESS)

if __name__ == "__main__":
    print(colorize("This module is not meant to be run directly. Import and use run_gen_a function in your main script.", Colors.WARNING))
