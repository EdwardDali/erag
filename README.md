Installation one-liner: 

git clone https://github.com/yourusername/rag-tool.git && cd rag-tool && pip install -r requirements.txt && python -m spacy download en_core_web_sm && python -m nltk.downloader punkt


Overview
This RAG (Retrieval-Augmented Generation) tool is a sophisticated system that combines lexical, semantic, text, and knowledge graph searches with conversation context to provide accurate and contextually relevant responses. The tool processes various document types, creates embeddings, builds a knowledge graph, and uses this information to answer user queries intelligently.
Features

Multi-modal Search: Incorporates lexical, semantic, text, and knowledge graph searches.
Conversation Context: Maintains context across multiple interactions for more coherent conversations.
Document Processing: Handles various document types including DOCX, JSON, PDF, and plain text.
Embedding Generation: Creates and manages embeddings for efficient semantic search.
Knowledge Graph: Builds and utilizes a knowledge graph for enhanced information retrieval.
Customizable Settings: Offers a wide range of configurable parameters through a GUI.

Components
1. Document Processing (file_processing.py)

Handles the chunking of documents into manageable pieces.
Supports multiple file formats (DOCX, PDF, Text).
Configurable chunk size and overlap for text processing.
Provides functions for uploading different file types.

2. Embedding Utils (embeddings_utils.py)

Manages the creation and storage of document embeddings.
Utilizes sentence transformers for embedding generation.
Supports batch processing for efficient embedding computation.
Provides functions for loading and saving embeddings.

3. Knowledge Graph Creation (create_graph.py)

Builds a knowledge graph from processed documents.
Extracts entities and relationships to form the graph structure.
Utilizes spaCy for named entity recognition and relation extraction.
Supports family relation extraction.
Creates semantic similarity edges between document nodes.

4. Search Utils (search_utils.py)

Implements various search strategies:

Lexical search
Semantic search
Text-based search
Knowledge graph search


Combines search results based on configurable weights.

5. RAG System (run_model.py)

Core component that integrates all other modules.
Manages the conversation flow and context.
Interfaces with the language model (e.g., Ollama) for generating responses.

6. Settings Manager (settings.py)

Provides a GUI for configuring various parameters of the system.
Allows saving and loading of configuration templates.

7. Main GUI (main.py)

Offers a graphical interface for interacting with the RAG system.
Provides options for document upload, embedding generation, and model execution.

Installation

Clone the repository and navigate to the project directory:
Copygit clone https://github.com/yourusername/rag-tool.git && cd rag-tool

Install the required dependencies:
Copypip install -r requirements.txt

Download the required spaCy and NLTK models:
Copypython -m spacy download en_core_web_sm
python -m nltk.downloader punkt


Usage

Document Upload:

Use the GUI to upload and process documents of various formats (DOCX, PDF, Text).


Embedding Generation:

After document upload, generate embeddings for efficient semantic search.


Knowledge Graph Creation:

Create a knowledge graph based on the uploaded documents.


Configure Settings:

Use the Settings tab to customize various parameters of the system.


Run Model:

Select the API type (Ollama or Llama) and start the conversation.


Interact:

Ask questions or provide prompts in the console to interact with the RAG system.



Key Settings

Chunk Size: Size of document chunks for processing (default: 500).
Overlap Size: Overlap between document chunks (default: 200).
Batch Size: Batch size for embedding generation (default: 32).
Max History Length: Maximum number of conversation turns to consider.
Conversation Context Size: Number of recent messages to include in context.
Update Threshold: Number of new entries before updating embeddings.
Temperature: Controls randomness in model outputs.
Top K: Number of top results to consider from each search method.
Search Weights: Relative importance of each search method (lexical, semantic, graph, text).
Similarity Threshold: Threshold for semantic similarity edges in the knowledge graph (default: 0.7).
Enable Family Extraction: Toggle for extracting family relationships in the knowledge graph.
Min Entity Occurrence: Minimum number of occurrences for an entity to be included in the graph.

Customization
The system is highly customizable. You can modify various aspects including:

Embedding model (MODEL_NAME in embeddings_utils.py)
NLP model for entity extraction (NLP_MODEL in create_graph.py)
Search method weights and thresholds
Knowledge graph parameters

Refer to the Settings tab in the GUI for all available customization options.
File Descriptions

file_processing.py: Handles document uploading and chunking.
embeddings_utils.py: Manages embedding generation and storage.
create_graph.py: Creates and manages the knowledge graph.
search_utils.py: Implements various search strategies.
run_model.py: Core RAG system implementation.
settings.py: Settings management and GUI.
main.py: Main application GUI.

Troubleshooting

If you encounter issues with specific file formats, ensure you have the necessary libraries installed (e.g., python-docx for DOCX files, PyPDF2 for PDF files).
For performance issues, try adjusting the chunk size, batch size, or reducing the number of documents processed.
If the knowledge graph is not providing useful results, you may need to adjust the entity extraction settings or increase the minimum entity occurrence.
