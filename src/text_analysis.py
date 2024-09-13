# Standard library imports
import os
import re
from collections import defaultdict

# Third-party imports
import spacy
import matplotlib.pyplot as plt
import networkx as nx
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from wordcloud import WordCloud
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Local imports
from src.settings import settings
from src.api_model import EragAPI
from src.look_and_feel import success, info, warning, error

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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

def chunk_text(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def generate_summary(erag_api, chunk):
    prompt = f"""Summarize the following text, focusing on key concepts and main ideas:

Text:
{chunk}

Summary:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates concise summaries."},
        {"role": "user", "content": prompt}
    ]
    response = erag_api.chat(messages, temperature=settings.temperature)
    return response.strip()

def generate_key_concepts(doc):
    return [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop]

def create_concept_graph(text):
    doc = nlp(text)
    concepts = generate_key_concepts(doc)
    
    G = nx.Graph()
    for i, concept in enumerate(concepts[:-1]):
        G.add_edge(concept, concepts[i+1])
    
    return G

def visualize_concept_graph(G, output_path):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_weight='bold')
    plt.title("Concept Relationship Graph")
    plt.savefig(output_path)
    plt.close()

def generate_word_cloud(text, output_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    plt.savefig(output_path)
    plt.close()

def generate_taxonomy(text_chunks):
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(text_chunks)
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
    clustering.fit(tfidf_matrix.toarray())
    
    taxonomy = {}
    for i, label in enumerate(clustering.labels_):
        if label not in taxonomy:
            taxonomy[label] = []
        taxonomy[label].append(text_chunks[i])
    
    return taxonomy

def visualize_taxonomy(taxonomy, output_path):
    G = nx.Graph()
    for label, chunks in taxonomy.items():
        G.add_node(f"Topic {label}")
        for i, chunk in enumerate(chunks):
            subtopic = f"Subtopic {label}.{i}"
            G.add_node(subtopic)
            G.add_edge(f"Topic {label}", subtopic)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=3000, font_size=8, font_weight='bold')
    plt.title("Content Taxonomy")
    plt.savefig(output_path)
    plt.close()

def extract_definitions(text):
    patterns = [
        r'(?P<term>\w+)\s+is defined as\s+(?P<definition>[^.]+)',
        r'(?P<term>\w+)\s+refers to\s+(?P<definition>[^.]+)',
        r'(?P<term>\w+):\s+(?P<definition>[^.]+)',
    ]
    
    glossary = defaultdict(list)
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            term = match.group('term').lower()
            definition = match.group('definition').strip()
            glossary[term].append(definition)
    
    return glossary

def generate_pdf_report(output_folder, file_name, summary, glossary, taxonomy):
    pdf_path = os.path.join(output_folder, f"{file_name}_analysis_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Text Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Summary
    story.append(Paragraph("Summary:", styles['Heading2']))
    story.append(Paragraph(summary, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Glossary
    story.append(Paragraph("Glossary:", styles['Heading2']))
    for term, definitions in glossary.items():
        story.append(Paragraph(f"<b>{term}</b>: {'; '.join(definitions)}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Taxonomy
    story.append(Paragraph("Content Taxonomy:", styles['Heading2']))
    for label, chunks in taxonomy.items():
        story.append(Paragraph(f"Topic {label}:", styles['Heading3']))
        for i, chunk in enumerate(chunks):
            story.append(Paragraph(f"Subtopic {label}.{i}: {chunk[:100]}...", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Concept Graph
    story.append(Paragraph("Concept Relationship Graph:", styles['Heading2']))
    story.append(Image(os.path.join(output_folder, "concept_graph.png"), width=400, height=300))
    story.append(Spacer(1, 12))

    # Word Cloud
    story.append(Paragraph("Word Cloud:", styles['Heading2']))
    story.append(Image(os.path.join(output_folder, "word_cloud.png"), width=400, height=200))

    doc.build(story)
    return pdf_path

def run_text_analysis(file_path, erag_api):
    content = read_file_content(file_path)
    chunks = chunk_text(content, settings.file_chunk_size)
    
    output_folder = os.path.join(settings.output_folder, "text_analysis_output")
    os.makedirs(output_folder, exist_ok=True)
    
    print(info("Generating summary and analyzing text..."))
    summaries = []
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks)):
        summary = generate_summary(erag_api, chunk)
        summaries.append(summary)
    
    combined_summary = "\n\n".join(summaries)
    
    print(info("Creating concept graph..."))
    G = create_concept_graph(content)
    graph_output = os.path.join(output_folder, "concept_graph.png")
    visualize_concept_graph(G, graph_output)
    
    print(info("Generating word cloud..."))
    word_cloud_output = os.path.join(output_folder, "word_cloud.png")
    generate_word_cloud(content, word_cloud_output)
    
    print(info("Generating taxonomy..."))
    taxonomy = generate_taxonomy(chunks)
    taxonomy_output = os.path.join(output_folder, "taxonomy.png")
    visualize_taxonomy(taxonomy, taxonomy_output)
    
    print(info("Extracting glossary..."))
    glossary = extract_definitions(content)
    
    print(info("Generating PDF report..."))
    pdf_path = generate_pdf_report(output_folder, os.path.basename(file_path), combined_summary, glossary, taxonomy)
    
    return success(f"Text analysis completed. PDF report saved at: {pdf_path}")

if __name__ == "__main__":
    print(warning("This module is not meant to be run directly. Import and use run_text_analysis function in your main script."))