import os
import re
from settings import settings
from PyPDF2 import PdfReader

def chunk_text(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def read_file_content(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return read_pdf(file_path)
    else:  # Assume it's a text file
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def generate_questions(client, api_type, chunk, question_number, total_questions, output_file, chunk_size):
    print(f"Generating questions for chunk {question_number}/{total_questions} (size: {chunk_size})")
    
    prompt = f"""Based on the following text, generate 3 insightful questions that would test a reader's understanding of the key concepts, main ideas, or important details. The questions should be specific to the content provided. Provide only the questions, without any additional text or answers.

Text:
{chunk}

Generate 3 questions:"""

    response = client.chat.completions.create(
        model=settings.ollama_model if api_type == 'ollama' else settings.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates insightful questions based on given text."},
            {"role": "user", "content": prompt}
        ],
        temperature=settings.temperature
    )

    questions = response.choices[0].message.content.strip().split('\n')
    
    # Write questions to file immediately
    with open(output_file, 'a', encoding='utf-8') as file:
        file.write(f"Chunk {question_number}. [Chunk size: {chunk_size}]\n")
        for question in questions:
            file.write(f"{question.strip()}\n")
        file.write("\n")  # Add an extra newline for separation between chunks
    
    return questions

def extract_questions(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regular expression to match all questions within each chunk
    chunk_pattern = r'Chunk \d+\. \[Chunk size: \d+\]\n((?:.*?\n)+)'
    question_pattern = r'^(?:\d+\.\s*)?(.*?\?)$'
    
    chunks = re.findall(chunk_pattern, content, re.MULTILINE)
    extracted_questions = []

    for chunk in chunks:
        questions = re.findall(question_pattern, chunk, re.MULTILINE)
        extracted_questions.extend(questions)

    # Write extracted questions to the new file without numbering
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in extracted_questions:
            f.write(f"{question.strip()}\n\n")

    return len(extracted_questions)

def run_create_q(file_path, api_type, client):
    # Get the base name of the input file
    input_file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Read the content of the file
    content = read_file_content(file_path)

    all_chunk_sizes = [settings.initial_question_chunk_size * (2**i) for i in range(settings.question_chunk_levels)]
    
    # Convert excluded_question_levels to a set of integers for efficient lookup
    excluded_levels = set(map(int, settings.excluded_question_levels))
    
    chunk_sizes = [size for i, size in enumerate(all_chunk_sizes) if i not in excluded_levels]
    
    # Calculate total number of chunks
    total_chunks = sum(len(chunk_text(content, size)) for size in chunk_sizes)
    print(f"Identified {total_chunks} chunks using {len(chunk_sizes)} levels")

    # Prepare the output file
    output_file = os.path.join(settings.output_folder, f"{input_file_name}_generated_questions.txt")
    # Clear the file if it exists
    open(output_file, 'w').close()

    chunk_counter = 1

    for chunk_size in chunk_sizes:
        chunks = chunk_text(content, chunk_size)
        for chunk in chunks:
            questions = generate_questions(client, api_type, chunk, chunk_counter, total_chunks, output_file, chunk_size)
            chunk_counter += 1

    print(f"Generated questions for {chunk_counter - 1} chunks and saved to {output_file}")

    # Extract questions
    extracted_file = os.path.join(settings.output_folder, f"{input_file_name}_extracted_questions.txt")
    num_extracted = extract_questions(output_file, extracted_file)
    
    extraction_message = f"{num_extracted} questions were extracted and saved to {extracted_file}"
    print(extraction_message)
    
    return f"Generated questions for {chunk_counter - 1} chunks. {extraction_message}"
