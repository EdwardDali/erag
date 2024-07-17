import json
import csv
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any
from src.settings import settings
from src.look_and_feel import error, success, warning, info

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
    return qa_pairs

def enrich_qa_pair(qa_pair: Dict[str, str], erag_api: Any) -> Dict[str, Any]:
    prompt = f"""
Given the following question and answer pair, provide additional metadata in the specified format:

Question: {qa_pair['question']}
Answer: {qa_pair['answer']}

Please provide the following metadata:
1. Domain (e.g., Science, History, Literature, General)
2. Difficulty (Easy, Medium, Hard)
3. Keywords (comma-separated list)
4. Language
5. Answer type (short_answer, long_explanation, multiple_choice)

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
    qa_pair['id'] = f"QA_{len(qa_pair):04d}"
    
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
    enriched_qa_pairs = [enrich_qa_pair(qa_pair, erag_api) for qa_pair in qa_pairs]

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

    return success(f"Dataset generated successfully. Output files: {', '.join(generated_files)}")
