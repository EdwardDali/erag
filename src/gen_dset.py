import json
import csv
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any
from src.settings import settings
from src.look_and_feel import error, success, warning, info
import time

def read_qa_file(file_path: str) -> List[Dict[str, str]]:
    qa_pairs = []
    current_qa = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('Question:'):
                if current_qa:
                    qa_pairs.append(current_qa)
                    current_qa = {}
                current_qa['question'] = line[9:].strip()
            elif line.startswith('Answer:'):
                current_qa['answer'] = line[7:].strip()
    if current_qa:
        qa_pairs.append(current_qa)
    print(info(f"Number of Q&A pairs identified: {len(qa_pairs)}"))
    return qa_pairs

def generate_id(index: int, total: int) -> str:
    # Calculate the number of digits needed to represent the total number of QA pairs
    id_length = len(str(total))
    return f"QA_{index:0{id_length}d}"

def enrich_qa_pair(qa_pair: Dict[str, str], erag_api: Any, qa_id: str) -> Dict[str, Any]:
    prompt = f"""
Given the following question and answer pair, provide additional metadata in the specified format:

Question: {qa_pair['question']}
Answer: {qa_pair['answer']}

Please provide the following metadata:
1. Domain (e.g., Science, History, Literature, General)
2. Difficulty (Easy, Medium, Hard)
3. Keywords (comma-separated list)
4. Language
5. Answer type (choose one: short, medium, long)

Format your response as follows:
Domain: [domain]
Difficulty: [difficulty]
Keywords: [keyword1, keyword2, ...]
Language: [language]
Answer type: [answer_type]
"""

    response = erag_api.chat([{"role": "user", "content": prompt}])
    
    # Parse the LLM response
    metadata = {}
    for line in response.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip().lower().replace(' ', '_')] = value.strip()

    # Add the metadata to the qa_pair
    qa_pair.update(metadata)
    qa_pair['id'] = qa_id
    
    # Ensure answer_type is one of short, medium, or long
    if qa_pair['answer_type'] not in ['short', 'medium', 'long']:
        answer_length = len(qa_pair['answer'].split())
        if answer_length < 30:
            qa_pair['answer_type'] = 'short'
        elif answer_length < 100:
            qa_pair['answer_type'] = 'medium'
        else:
            qa_pair['answer_type'] = 'long'
    
    return qa_pair

def save_jsonl(data: List[Dict[str, Any]], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def save_csv(data: List[Dict[str, Any]], output_file: str):
    fieldnames = settings.dataset_fields
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow({k: item.get(k, '') for k in fieldnames})

def save_parquet(data: List[Dict[str, Any]], output_file: str):
    table = pa.Table.from_pylist(data)
    pq.write_table(table, output_file)

def run_gen_dset(file_path: str, api_type: str, erag_api: Any) -> str:
    qa_pairs = read_qa_file(file_path)
    
    enriched_qa_pairs = []
    total_pairs = len(qa_pairs)
    start_time = time.time()
    
    for i, qa_pair in enumerate(qa_pairs, 1):
        qa_id = generate_id(i, total_pairs)
        enriched_pair = enrich_qa_pair(qa_pair, erag_api, qa_id)
        enriched_qa_pairs.append(enriched_pair)
        
        # Calculate and display progress
        progress = (i / total_pairs) * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / i) * total_pairs
        remaining_time = estimated_total_time - elapsed_time
        
        print(info(f"Processed Q&A pair {i} of {total_pairs} ({progress:.2f}%) - ID: {qa_id}"))
        print(info(f"Estimated time remaining: {remaining_time:.2f} seconds"))

    output_formats = settings.dataset_output_formats
    base_output_file = settings.dataset_output_file

    generated_files = []

    for format in output_formats:
        output_file = f"{base_output_file}.{format}"
        if format == 'jsonl':
            save_jsonl(enriched_qa_pairs, output_file)
        elif format == 'csv':
            save_csv(enriched_qa_pairs, output_file)
        elif format == 'parquet':
            save_parquet(enriched_qa_pairs, output_file)
        generated_files.append(output_file)

    total_time = time.time() - start_time
    print(success(f"Dataset generation completed in {total_time:.2f} seconds"))
    return success(f"Dataset generated successfully. Output files: {', '.join(generated_files)}")
