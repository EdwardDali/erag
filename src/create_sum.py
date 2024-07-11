import os
import fitz  # PyMuPDF
from typing import List
from src.settings import settings
from src.api_model import configure_api, LlamaClient
from src.color_scheme import Colors, colorize
import colorama

# Initialize colorama
colorama.init(autoreset=True)

def extract_text(file_path: str) -> str:
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        return process_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

def process_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def split_into_chunks(text: str) -> List[str]:
    return [text[i:i+settings.summarization_chunk_size] for i in range(0, len(text), settings.summarization_chunk_size)]

def summarize_chunk(chunk: str, api_type: str, client) -> str:
    prompt = f"""Write a concise summary (about {settings.summarization_summary_size} characters) of the following text:

{chunk}

SUMMARY:"""

    if api_type == "llama":
        response = client.chat([
            {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
            {"role": "user", "content": prompt}
        ], temperature=settings.temperature)
        return response.strip()
    else:
        response = client.chat.completions.create(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=settings.temperature,
            max_tokens=settings.summarization_summary_size
        )
        return response.choices[0].message.content.strip()

def review_summaries(summaries: List[str], api_type: str, client) -> str:
    prompt = f"""Summarize the following {settings.summarization_combining_number} summaries into one coherent paragraph:

{' '.join(summaries)}

SUMMARIZED PARAGRAPH:"""

    if api_type == "llama":
        response = client.chat([
            {"role": "system", "content": "You are a helpful assistant that combines multiple summaries into a coherent paragraph."},
            {"role": "user", "content": prompt}
        ], temperature=settings.temperature)
        return response.strip()
    else:
        response = client.chat.completions.create(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that combines multiple summaries into a coherent paragraph."},
                {"role": "user", "content": prompt}
            ],
            temperature=settings.temperature,
            max_tokens=settings.summarization_final_chunk_size
        )
        return response.choices[0].message.content.strip()

def create_summary(file_path: str, api_type: str, client) -> str:
    try:
        full_text = extract_text(file_path)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_folder = os.path.join("output", base_name)
        os.makedirs(output_folder, exist_ok=True)

        # Save the full text
        full_text_path = os.path.join(output_folder, f"{base_name}_full.txt")
        with open(full_text_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(colorize(f"Saved full text: {full_text_path}", Colors.SUCCESS))

        chunks = split_into_chunks(full_text)
        
        # Create a single file for all chunk summaries
        all_summaries_path = os.path.join(output_folder, "all_chunk_summaries.txt")
        chunk_summaries = []
        with open(all_summaries_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                summary = summarize_chunk(chunk, api_type, client)
                f.write(f"{summary}\n\n")
                f.flush()  # Ensure the summary is written to the file immediately
                chunk_summaries.append(summary)
                print(colorize(f"Processed chunk {i}/{len(chunks)}", Colors.INFO))

        print(colorize(f"All chunk summaries saved to: {all_summaries_path}", Colors.SUCCESS))

        # Create reviewed summary by processing chunks in groups
        reviewed_summary_path = os.path.join(output_folder, "reviewed_summary.txt")
        with open(reviewed_summary_path, 'w', encoding='utf-8') as f:
            for i in range(0, len(chunk_summaries), settings.summarization_combining_number):
                group = chunk_summaries[i:i+settings.summarization_combining_number]
                reviewed_summary = review_summaries(group, api_type, client)
                f.write(f"{reviewed_summary}\n\n")
                f.flush()  # Ensure the summary is written to the file immediately
                print(colorize(f"Processed reviewed summary group {i//settings.summarization_combining_number + 1}", Colors.INFO))

        print(colorize(f"Reviewed summary saved to: {reviewed_summary_path}", Colors.SUCCESS))

        return colorize(f"Successfully processed {len(chunks)} chunks. Full text, all chunk summaries, and reviewed summary saved in folder: {output_folder}", Colors.SUCCESS)

    except Exception as e:
        return colorize(f"An error occurred while processing the document: {str(e)}", Colors.ERROR)

def run_create_sum(file_path: str, api_type: str, client) -> str:
    if api_type == "llama" and not isinstance(client, LlamaClient):
        client = LlamaClient()  # Create a new LlamaClient instance if it's not already
    return create_summary(file_path, api_type, client)
