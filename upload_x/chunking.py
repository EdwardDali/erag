import re

def handle_text_chunking(text):
    # Normalize whitespace and clean up text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Define chunk size and overlap size
    chunk_size = 1000
    overlap_size = 100  # Adjust overlap size as needed
    
    # Split text into chunks with overlap
    chunks = []
    start = 0
    end = chunk_size
    while start < len(text):
        chunks.append(text[start:end])
        start += chunk_size - overlap_size
        end = start + chunk_size
    
    return chunks
