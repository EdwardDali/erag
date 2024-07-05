import os
import re
import fitz  # PyMuPDF
from typing import List
from settings import settings

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

def split_into_chunks(text: str, chunk_size: int = 3000) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_chunk(chunk: str, api_type: str, client) -> str:
    prompt = f"""Write a concise summary (about 200 characters) of the following text:

{chunk}

SUMMARY:"""

    response = client.chat.completions.create(
        model=settings.ollama_model if api_type == "ollama" else settings.llama_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
            {"role": "user", "content": prompt}
        ],
        temperature=settings.temperature,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()

def create_final_summary(summaries: List[str], api_type: str, client) -> str:
    combined_summaries = "\n".join(summaries)
    prompt = f"""Create a comprehensive final summary (about 1000 characters) based on the following chunk summaries:

{combined_summaries}

FINAL SUMMARY:"""

    response = client.chat.completions.create(
        model=settings.ollama_model if api_type == "ollama" else settings.llama_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates comprehensive final summaries."},
            {"role": "user", "content": prompt}
        ],
        temperature=settings.temperature,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()

def create_summary(file_path: str, api_type: str, client) -> str:
    try:
        full_text = extract_text(file_path)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_folder = os.path.join("summaries", base_name)
        os.makedirs(output_folder, exist_ok=True)

        # Save the full text
        full_text_path = os.path.join(output_folder, f"{base_name}_full.txt")
        with open(full_text_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Saved full text: {full_text_path}")

        chunks = split_into_chunks(full_text)
        chunk_summaries = []

        for i, chunk in enumerate(chunks, 1):
            summary = summarize_chunk(chunk, api_type, client)
            chunk_summaries.append(summary)

            # Save individual chunk summaries
            summary_path = os.path.join(output_folder, f"chunk_{i:04d}_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Saved chunk summary: {summary_path}")

        # Create and save the final summary
        final_summary = create_final_summary(chunk_summaries, api_type, client)
        final_summary_path = os.path.join(output_folder, "final_summary.txt")
        with open(final_summary_path, 'w', encoding='utf-8') as f:
            f.write(final_summary)
        print(f"Final summary saved to: {final_summary_path}")

        return f"Successfully processed {len(chunks)} chunks. Full text, chunk summaries, and final summary saved in folder: {output_folder}"

    except Exception as e:
        return f"An error occurred while processing the document: {str(e)}"

def run_create_sum(file_path: str, api_type: str, client) -> str:
    return create_summary(file_path, api_type, client)
