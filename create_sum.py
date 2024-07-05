import os
import re
import fitz  # PyMuPDF
from typing import List, Tuple
from settings import settings

def extract_chapters(file_path: str) -> List[Tuple[str, str]]:
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        return process_pdf(file_path)
    else:
        return extract_chapters_text(file_path)

def process_pdf(pdf_path: str) -> List[Tuple[str, str]]:
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    summaries_folder = os.path.join("summaries", base_name)
    os.makedirs(summaries_folder, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    # Save the entire book in text format
    full_text_path = os.path.join(summaries_folder, f"{base_name}_full.txt")
    with open(full_text_path, 'w', encoding='utf-8') as text_file:
        text_file.write(full_text)
    print(f"Saved entire book as text file: {full_text_path}")
    
    # Extract chapters
    chapters = split_into_chapters(full_text)
    
    doc.close()
    return chapters

def split_into_chapters(text: str) -> List[Tuple[str, str]]:
    chapter_pattern = r'(?:^|\n)(?:CHAPTER|Chapter)\s+(?:\d+|[IVX]+)[\.:]\s*(.*?)(?=\n)'
    chapters = list(re.finditer(chapter_pattern, text, re.MULTILINE | re.IGNORECASE))
    
    if not chapters:
        return [("Entire Document", text)]
    
    result = []
    for i, match in enumerate(chapters):
        chapter_title = match.group(0).strip()
        start = match.start()
        end = chapters[i+1].start() if i+1 < len(chapters) else len(text)
        chapter_content = text[start:end].strip()
        
        # Skip if the chapter content is too short or seems to be a TOC entry
        if len(chapter_content.split('\n')) > 25 and len(chapter_content) > 1000:
            # Remove page numbers and repeated chapter titles at the beginning of the content
            lines = chapter_content.split('\n')
            clean_lines = []
            for line in lines:
                if not re.match(r'^Chapter\s+\d+\s*$', line.strip()) and not re.match(r'^\d+\s*$', line.strip()):
                    clean_lines.append(line)
            clean_content = '\n'.join(clean_lines).strip()
            
            result.append((chapter_title, clean_content))
    
    return result

def extract_chapters_text(file_path: str) -> List[Tuple[str, str]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return split_into_chapters(content)

def summarize_chapter(chapter_title: str, chapter_content: str, api_type: str, client) -> str:
    prompt = f"""Write an extensive summary of the following chapter:

Chapter title: {chapter_title}

Chapter content:
{chapter_content[:3000]}  # Limit content to 3000 characters to fit within context window

SUMMARY:"""

    response = client.chat.completions.create(
        model=settings.ollama_model if api_type == "ollama" else settings.llama_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates comprehensive chapter summaries."},
            {"role": "user", "content": prompt}
        ],
        temperature=settings.temperature
    )

    return response.choices[0].message.content.strip()

def create_summary(file_path: str, api_type: str, client) -> str:
    try:
        chapters = extract_chapters(file_path)
        if not chapters:
            return "No chapters found in the document."

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_folder = os.path.join("summaries", base_name)
        os.makedirs(output_folder, exist_ok=True)

        all_summaries = []

        for i, (chapter_title, chapter_content) in enumerate(chapters, 1):
            safe_title = re.sub(r'[^\w\-_\. ]', '_', chapter_title)
            safe_title = safe_title[:50]  # Limit filename length
            
            # Save the full chapter content
            full_chapter_path = os.path.join(output_folder, f"{i:02d}_{safe_title}_full.txt")
            with open(full_chapter_path, 'w', encoding='utf-8') as f:
                f.write(f"{chapter_title}\n\n")
                f.write(chapter_content)
            print(f"Saved full chapter: {full_chapter_path}")

            # Create and save the summary
            summary = summarize_chapter(chapter_title, chapter_content, api_type, client)
            summary_path = os.path.join(output_folder, f"{i:02d}_{safe_title}_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Summary of {chapter_title}\n\n")
                f.write(summary)
            print(f"Saved summary: {summary_path}")

            all_summaries.append(f"Chapter {i}: {chapter_title}\n{summary}\n\n")

        # Create an overall summary file
        overall_summary_path = os.path.join(output_folder, "overall_summary.txt")
        with open(overall_summary_path, 'w', encoding='utf-8') as f:
            f.write("Overall Summary of the Book\n\n")
            f.write("".join(all_summaries))

        print(f"Overall summary saved to: {overall_summary_path}")
        return f"Successfully processed {len(chapters)} chapters. Full chapters and summaries saved in folder: {output_folder}"

    except Exception as e:
        return f"An error occurred while processing the book: {str(e)}"

def run_create_sum(file_path: str, api_type: str, client) -> str:
    return create_summary(file_path, api_type, client)
