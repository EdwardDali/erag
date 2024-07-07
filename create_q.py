import os
from settings import settings

def chunk_text(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def generate_questions(client, api_type, chunk, question_number, total_questions, output_file):
    print(f"Generating question(s) {question_number}/{total_questions}")
    
    prompt = f"""Based on the following text, generate {settings.questions_per_chunk} insightful question(s) that would test a reader's understanding of the key concepts, main ideas, or important details. The question(s) should be specific to the content provided.

Text:
{chunk}

Generate {settings.questions_per_chunk} question(s):"""

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
        for i, question in enumerate(questions, question_number):
            file.write(f"{i}. [Chunk size: {len(chunk)}] {question}\n\n")
    
    return questions

def run_create_q(file_path, api_type, client):
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    chunk_sizes = [settings.initial_question_chunk_size * (2**i) for i in range(settings.question_chunk_levels)]
    
    # Calculate total number of questions
    total_questions = sum(len(chunk_text(content, size)) for size in chunk_sizes) * settings.questions_per_chunk
    print(f"Identified {total_questions} possible questions using {settings.question_chunk_levels} levels")

    # Prepare the output file
    output_file = os.path.join(settings.output_folder, "generated_questions.txt")
    # Clear the file if it exists
    open(output_file, 'w').close()

    question_counter = 1

    for chunk_size in chunk_sizes:
        chunks = chunk_text(content, chunk_size)
        for chunk in chunks:
            new_questions = generate_questions(client, api_type, chunk, question_counter, total_questions, output_file)
            question_counter += len(new_questions)

    return f"Generated {question_counter - 1} questions and saved to {output_file}"
