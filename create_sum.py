import os
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

def review_summaries(summaries: List[str], api_type: str, client) -> str:
    prompt = f"""Summarize the following three summaries into one coherent paragraph:

{' '.join(summaries)}

SUMMARIZED PARAGRAPH:"""

    response = client.chat.completions.create(
        model=settings.ollama_model if api_type == "ollama" else settings.llama_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that combines multiple summaries into a coherent paragraph."},
            {"role": "user", "content": prompt}
        ],
        temperature=settings.temperature,
        max_tokens=300  # Slightly larger to account for combining 3 summaries
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
        print(f"Saved full text: {full_text_path}")

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
                print(f"Processed chunk {i}/{len(chunks)}")

        print(f"All chunk summaries saved to: {all_summaries_path}")

        # Create reviewed summary by processing chunks in groups of 3
        reviewed_summaries = []
        for i in range(0, len(chunk_summaries), 3):
            group = chunk_summaries[i:i+3]
            reviewed_summary = review_summaries(group, api_type, client)
            reviewed_summaries.append(reviewed_summary)

        # Save the reviewed summary
        reviewed_summary_path = os.path.join(output_folder, "reviewed_summary.txt")
        with open(reviewed_summary_path, 'w', encoding='utf-8') as f:
            for summary in reviewed_summaries:
                f.write(f"{summary}\n\n")
        print(f"Reviewed summary saved to: {reviewed_summary_path}")

        return f"Successfully processed {len(chunks)} chunks. Full text, all chunk summaries, and reviewed summary saved in folder: {output_folder}"

    except Exception as e:
        return f"An error occurred while processing the document: {str(e)}"

def run_create_sum(file_path: str, api_type: str, client) -> str:
    return create_summary(file_path, api_type, client)
