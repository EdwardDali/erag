import os
from tqdm import tqdm
from src.settings import settings
from src.api_model import EragAPI, create_erag_api
from src.talk2doc import RAGSystem
from src.web_rag import WebRAG
from src.look_and_feel import success, info, warning, error
from tqdm import tqdm

def read_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def generate_answer_talk2doc(rag_system, question):
    response = rag_system.get_response(question)
    return response

def generate_answer_web_rag(web_rag, question):
    response = web_rag.search_and_process(question)
    return response

def generate_answer_hybrid(rag_system, web_rag, question, erag_api):
    talk2doc_response = rag_system.get_response(question)
    web_rag_response = web_rag.search_and_process(question)
    
    hybrid_prompt = f"""Combine and rephrase the following two answers into a comprehensive and coherent response:

Answer 1 (Talk2Doc): {talk2doc_response}

Answer 2 (Web RAG): {web_rag_response}

Combined and rephrased answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that combines and rephrases information from multiple sources into a coherent response."},
        {"role": "user", "content": hybrid_prompt}
    ]
    combined_response = erag_api.chat(messages, temperature=settings.temperature)

    return combined_response

def run_gen_a(questions_file, gen_method, api_type, erag_api):
    questions = read_questions(questions_file)
    
    output_file = os.path.join(settings.output_folder, f"generated_answers_{gen_method}.txt")
    
    rag_system = None
    web_rag = None
    
    if gen_method in ["talk2doc", "hybrid"]:
        rag_system = RAGSystem(erag_api)
    
    if gen_method in ["web_rag", "hybrid"]:
        web_rag = WebRAG(erag_api)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, question in enumerate(tqdm(questions, desc="Generating Answers", bar_format="{l_bar}%s{bar}%s{r_bar}" % (success(""), ""))):
            if gen_method == "talk2doc":
                answer = generate_answer_talk2doc(rag_system, question)
            elif gen_method == "web_rag":
                answer = generate_answer_web_rag(web_rag, question)
            else:  # hybrid
                answer = generate_answer_hybrid(rag_system, web_rag, question, erag_api)
            
            # Write to file immediately after generating each answer
            f.write(f"Question: {question}\n\n")
            f.write(f"Answer: {answer}\n\n")
            f.write("-" * 50 + "\n\n")
            f.flush()  # Ensure the content is written to the file immediately
            
            print(f"\n{info(f'Processed question {i+1}/{len(questions)}')}")

    return success(f"Generated answers for {len(questions)} questions using {gen_method} method. Saved to {output_file}")

if __name__ == "__main__":
    print(warning("This module is not meant to be run directly. Import and use run_gen_a function in your main script."))
