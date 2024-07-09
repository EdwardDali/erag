## Overview

This RAG (Retrieval-Augmented Generation) tool is a sophisticated system that combines lexical, semantic, text, and knowledge graph searches with conversation context to provide accurate and contextually relevant responses. The tool processes various document types, creates embeddings, builds a knowledge graph, and uses this information to answer user queries intelligently.


![Alt text](https://github.com/EdwardDali/e-rag/blob/main/docs/gui1.PNG)
![Alt text](https://github.com/EdwardDali/e-rag/blob/main/docs/gui2.PNG)
![Alt text](https://github.com/EdwardDali/e-rag/blob/main/docs/gui3.PNG)


## Installation

1. Clone the repository and navigate to the project directory:
 ```
git clone https://github.com/EdwardDali/erag.git && cd erag   
```

2. Install the required Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the required spaCy and NLTK models:
   ```
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt
   ```

4. Install Ollama:
   - For Linux/macOS:
     ```
     curl https://ollama.ai/install.sh | sh
     ```
   - For Windows:
     Visit https://ollama.ai/download and follow the installation instructions.

5. Run the phi3 model:
   ```
   ollama run phi3:instruct
   ```

Note: Ensure you have Python 3.7 or later installed on your system before proceeding with the installation.

## System Requirements

- Python 3.7+
- 8GB+ RAM recommended
- GPU (optional, but recommended for faster processing)
- Internet connection (for downloading models and dependencies)

## Troubleshooting Installation

- If you encounter permission errors during Ollama installation, you may need to use `sudo` (on Linux/macOS) or run as administrator (on Windows).
- Ensure your Python environment is properly set up and activated if you're using virtual environments.
- If you face issues with torch installation, consider installing it separately following the official PyTorch installation guide for your specific system configuration.



# RAG Tool Documentation

## Overview

This RAG (Retrieval-Augmented Generation) tool is a sophisticated system that combines lexical, semantic, text, and knowledge graph searches with conversation context to provide accurate and contextually relevant responses. The tool processes various document types, creates embeddings, builds a knowledge graph, and uses this information to answer user queries intelligently.
ERAG (Embeddings and Retrieval-Augmented Generation) is a system designed to process documents, generate embeddings, create knowledge graphs, and perform retrieval-augmented generation tasks. This document outlines the main components and their functionalities within the ERAG system.
The ERAG system provides a comprehensive solution for document processing, embedding generation, knowledge graph creation, and retrieval-augmented generation. By leveraging these components, users can enhance their document understanding and question-answering capabilities.

# ERAG System Features

The ERAG (Embeddings and Retrieval-Augmented Generation) system offers a comprehensive set of features designed to enhance document processing, information retrieval, and knowledge generation. Here are the key features of the ERAG system:

1. **Multi-modal Search**
   - Incorporates lexical, semantic, text, and knowledge graph searches
   - Combines multiple search methods for more accurate and relevant results
   - Allows customizable weighting of different search types

2. **Conversation Context**
   - Maintains context across multiple interactions for more coherent conversations
   - Implements a sliding window approach to manage conversation history
   - Utilizes conversation context to improve answer relevance

3. **Document Processing**
   - Handles various document types including DOCX, JSON, PDF, and plain text
   - Implements configurable text chunking with overlap for optimal processing
   - Supports batch processing of multiple documents

4. **Embedding Generation**
   - Creates and manages embeddings for efficient semantic search
   - Utilizes state-of-the-art sentence transformer models for embedding generation
   - Supports incremental updating of embeddings as new content is added

5. **Knowledge Graph**
   - Builds and utilizes a knowledge graph for enhanced information retrieval
   - Extracts entities and relationships from processed documents
   - Supports graph-based context retrieval for more comprehensive answers

6. **Customizable Settings**
   - Offers a wide range of configurable parameters through a graphical user interface
   - Allows fine-tuning of chunk sizes, similarity thresholds, and search weights
   - Provides options to enable/disable specific search types and features

7. **Web Content Processing**
   - Implements real-time web crawling and content extraction
   - Supports dynamic expansion of the knowledge base with web content
   - Offers web content summarization capabilities

8. **Knol Creation**
   - Generates comprehensive knowledge entries (knols) on specific subjects
   - Implements a multi-stage process for knol creation, improvement, and refinement
   - Generates and answers relevant questions to enhance knol content

9. **Retrieval-Augmented Generation (RAG)**
   - Combines retrieved context with language model capabilities for enhanced responses
   - Supports multiple API types (e.g., Ollama, LLaMA) for flexible deployment
   - Implements a RAG pipeline for both document-based and web-based content

10. **Adaptive Context Retrieval**
    - Dynamically adjusts the amount of context retrieved based on query complexity
    - Implements iterative searching and processing for web content
    - Supports on-the-fly expansion of the knowledge base during conversations

11. **Multi-stage Summarization**
    - Provides capabilities for summarizing individual web pages
    - Generates comprehensive final summaries from multiple sources
    - Offers configurable summary sizes for different use cases

12. **Entity Extraction and Linking**
    - Extracts named entities from processed text
    - Links entities across documents and web content
    - Utilizes entity information to enhance the knowledge graph and improve search results

13. **Modular Architecture**
    - Designed with separate modules for different functionalities (e.g., document processing, web RAG, knol creation)
    - Allows for easy extension and customization of system capabilities
    - Supports integration of new search methods or processing techniques

14. **Debug and Logging Capabilities**
    - Implements comprehensive logging for system operations and errors
    - Provides detailed debug information for search results and RAG processes
    - Supports saving of intermediate results for analysis and improvement

15. **User-friendly Interfaces**
    - Offers a command-line interface for direct interaction with the RAG system
    - Implements a graphical user interface for easy configuration and management
    - Provides color-coded console output for improved readability and user experience

These features make the ERAG system a powerful and flexible tool for document processing, information retrieval, and knowledge generation. The system's modular design and extensive customization options allow it to be adapted for a wide range of applications, from document-based question answering to web content analysis and comprehensive knowledge base creation.


## Components

### 1. Document Processing (`file_processing.py`)

This module handles the ingestion and preprocessing of various document types.

#### Key Features:
- Supports multiple file formats: DOCX, PDF, Text, and JSON
- Implements configurable text chunking with overlap
- Provides functions for uploading and processing different file types

#### Main Functions:
- `upload_docx()`, `upload_pdf()`, `upload_txt()`, `upload_json()`: Handle file uploads for respective formats
- `handle_text_chunking(text)`: Splits text into chunks with configurable size and overlap
- `process_file(file_type)`: Processes files based on their type
- `append_to_db(chunks, db_file)`: Appends processed chunks to the database file

### 2. Embedding Utils (`embeddings_utils.py`)

This module manages the creation, storage, and retrieval of document embeddings.

#### Key Features:
- Utilizes sentence transformers for embedding generation
- Supports batch processing for efficient embedding computation
- Provides functions for loading and saving embeddings

#### Main Functions:
- `compute_and_save_embeddings(model, save_path, content)`: Computes and saves embeddings for given content
- `load_embeddings_and_data(embeddings_file)`: Loads previously saved embeddings and associated data
- `load_or_compute_embeddings(model, db_file, embeddings_file)`: Loads existing embeddings or computes new ones if necessary

### 3. Knowledge Graph Creation (`create_graph.py`)

This module is responsible for creating a knowledge graph from processed documents.

#### Key Features:
- Uses spaCy for named entity recognition and natural language processing
- Creates a NetworkX graph representing document structure and entity relationships
- Supports semantic edge creation based on document similarity

#### Main Functions:
- `extract_entities_with_confidence(text)`: Extracts named entities from text with confidence scores
- `create_networkx_graph(data, embeddings)`: Creates a knowledge graph from document data and embeddings
- `create_knowledge_graph()`: Main function to create and save the knowledge graph
- `create_knowledge_graph_from_raw(raw_file_path)`: Creates a knowledge graph from a raw text file

### 4. Settings Management (`settings.py`)

This module manages the configuration settings for the entire ERAG system.

#### Key Features:
- Implements a singleton pattern for global access to settings
- Provides methods for loading, saving, and resetting settings
- Stores various configuration parameters for different components of the system

#### Main Functions:
- `load_settings()`: Loads settings from a JSON file
- `save_settings()`: Saves current settings to a JSON file
- `update_setting(key, value)`: Updates a specific setting
- `reset_to_defaults()`: Resets all settings to their default values

### 5. Main Application (`main.py`)

This is the entry point of the ERAG system, implementing the graphical user interface and orchestrating the various components.

#### Key Features:
- Implements a tkinter-based GUI with multiple tabs for different functionalities
- Manages the interaction between user inputs and the underlying ERAG components
- Provides buttons and interfaces for document upload, embedding creation, knowledge graph generation, and RAG operations

#### Main Classes:
- `ERAGGUI`: The main GUI class that sets up the interface and handles user interactions

#### Key Functions:
- `create_widgets()`: Sets up the main GUI components
- `upload_and_chunk(file_type)`: Handles file upload and processing
- `execute_embeddings()`: Triggers the embedding computation process
- `create_knowledge_graph()`: Initiates the knowledge graph creation process
- `run_model()`: Starts the RAG system for interaction

## 6. Talk2Doc Module (`talk2doc.py`)

This module implements the core Retrieval-Augmented Generation (RAG) system for document interaction.

### Key Components:
- `RAGSystem` class: Manages the RAG process, including API configuration, embedding loading, and conversation handling.
- Supports multiple API types (ollama, llama).
- Implements a colored console interface for user interaction.

### Main Functions:
- `configure_api(api_type)`: Sets up the API client based on the specified type.
- `load_embeddings()`: Loads pre-computed embeddings for the document database.
- `load_knowledge_graph()`: Loads the knowledge graph for enhanced context retrieval.
- `ollama_chat(user_input, system_message)`: Generates responses using the configured API.
- `run()`: Main loop for user interaction with the RAG system.

## 7. Web RAG Module (`web_rag.py`)

This module extends the RAG system to work with web content, allowing for real-time information retrieval and processing.

### Key Components:
- `WebRAG` class: Manages web content retrieval, processing, and RAG-based question answering.
- Implements web crawling, content chunking, and embedding generation for web pages.
- Supports iterative searching and processing of web content.

### Main Functions:
- `search_and_process(query)`: Performs web search and processes relevant URLs.
- `generate_qa(query)`: Generates answers based on processed web content.
- `process_next_urls()`: Processes additional URLs to expand the knowledge base.
- `run()`: Main loop for user interaction with the Web RAG system.

## 8. Web Sum Module (`web_sum.py`)

This module focuses on creating summaries of web content based on user queries.

### Key Components:
- `WebSum` class: Manages web content retrieval, summarization, and final summary generation.
- Implements web search, content relevance filtering, and multi-stage summarization.

### Main Functions:
- `search_and_process(query)`: Performs web search, filters relevant content, and generates summaries.
- `create_summary(content, query, index)`: Creates a summary for a single web page.
- `create_final_summary(summaries, query)`: Generates a comprehensive final summary from individual page summaries.
- `run()`: Main loop for user interaction with the Web Sum system.

## 9. Knol Creator Module (`create_knol.py`)

This module is responsible for creating comprehensive knowledge entries (knols) on specific subjects.

### Key Components:
- `KnolCreator` class: Manages the process of creating, improving, and finalizing knols.
- Implements a multi-stage process including initial creation, improvement, question generation, and answering.

### Main Functions:
- `create_knol(subject)`: Creates an initial structured knowledge entry.
- `improve_knol(knol, subject)`: Enhances and expands the initial knol.
- `generate_questions(knol, subject)`: Generates relevant questions based on the knol content.
- `answer_questions(questions, subject, knol)`: Answers generated questions using the RAG system.
- `create_final_knol(subject)`: Combines improved knol and Q&A to create the final knowledge entry.
- `run_knol_creator()`: Main loop for user interaction with the Knol Creation system.

## 10. Search Utils Module (`search_utils.py`)

This module provides various search utilities to enhance the retrieval capabilities of the ERAG system.

### Key Components:
- `SearchUtils` class: Implements different search methods including lexical, semantic, graph-based, and text search.

### Main Functions:
- `lexical_search(query)`: Performs lexical (keyword-based) search on the document content.
- `semantic_search(query)`: Conducts semantic search using document embeddings.
- `get_graph_context(query)`: Retrieves context from the knowledge graph based on the query.
- `text_search(query)`: Performs basic text search on the document content.
- `get_relevant_context(user_input, conversation_context)`: Combines different search methods to retrieve the most relevant context.


## System Workflow

1. Users upload documents through the GUI, which are processed and chunked by `file_processing.py`.
2. Embeddings are generated for the processed documents using `embeddings_utils.py`.
3. A knowledge graph is created from the processed documents and embeddings using `create_graph.py`.
4. The RAG system can then be initiated, utilizing the processed documents, embeddings, and knowledge graph for enhanced question-answering capabilities.
5. The core RAG system (`talk2doc.py`) provides the foundation for document interaction and question answering.
6. Web RAG (`web_rag.py`) and Web Sum (`web_sum.py`) modules extend the system's capabilities to process real-time web content.
7. The Knol Creator (`create_knol.py`) utilizes the RAG system to generate comprehensive knowledge entries on specific subjects.
8. Search Utils (`search_utils.py`) enhance the retrieval process by providing various search methods, improving the overall performance of the RAG system.

## Configuration

The system's behavior can be customized through various settings in the `settings.py` file. These include:

- File processing parameters (chunk size, overlap)
- Embedding generation settings (batch size, model name)
- Knowledge graph creation parameters (similarity threshold, entity occurrence threshold)
- RAG system settings (max history length, temperature)

Refer to the `settings.py` file for a complete list of configurable options.




## Installation

1. Clone the repository and navigate to the project directory:
   ```
   git clone https://github.com/EdwardDali/erag.git && cd erag
   ```

2. Install the required Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the required spaCy and NLTK models:
   ```
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt
   ```

4. Install Ollama:
   - For Linux/macOS:
     ```
     curl https://ollama.ai/install.sh | sh
     ```
   - For Windows:
     Visit https://ollama.ai/download and follow the installation instructions.

5. Pull the phi3:instruct model:
   ```
   ollama pull phi3:instruct
   ```

Note: Ensure you have Python 3.7 or later installed on your system before proceeding with the installation.

## System Requirements

- Python 3.7+
- 8GB+ RAM recommended
- GPU (optional, but recommended for faster processing)
- Internet connection (for downloading models and dependencies)

## Troubleshooting Installation

- If you encounter permission errors during Ollama installation, you may need to use `sudo` (on Linux/macOS) or run as administrator (on Windows).
- Ensure your Python environment is properly set up and activated if you're using virtual environments.
- If you face issues with torch installation, consider installing it separately following the official PyTorch installation guide for your specific system configuration.


## Usage

1. **Document Upload**:
   - Use the GUI to upload and process documents of various formats (DOCX, PDF, Text).

2. **Embedding Generation**:
   - After document upload, generate embeddings for efficient semantic search.

3. **Knowledge Graph Creation**:
   - Create a knowledge graph based on the uploaded documents.

4. **Configure Settings**:
   - Use the Settings tab to customize various parameters of the system.

5. **Run Model**:
   - Select the API type (Ollama or Llama) and start the conversation.

6. **Interact**:
   - Ask questions or provide prompts in the console to interact with the RAG system.

## Key Settings

- **Chunk Size**: Size of document chunks for processing (default: 500).
- **Overlap Size**: Overlap between document chunks (default: 200).
- **Batch Size**: Batch size for embedding generation (default: 32).
- **Max History Length**: Maximum number of conversation turns to consider.
- **Conversation Context Size**: Number of recent messages to include in context.
- **Update Threshold**: Number of new entries before updating embeddings.
- **Temperature**: Controls randomness in model outputs.
- **Top K**: Number of top results to consider from each search method.
- **Search Weights**: Relative importance of each search method (lexical, semantic, graph, text).
- **Similarity Threshold**: Threshold for semantic similarity edges in the knowledge graph (default: 0.7).
- **Enable Family Extraction**: Toggle for extracting family relationships in the knowledge graph.
- **Min Entity Occurrence**: Minimum number of occurrences for an entity to be included in the graph.

## Customization

The system is highly customizable. You can modify various aspects including:
- Embedding model (`MODEL_NAME` in `embeddings_utils.py`)
- NLP model for entity extraction (`NLP_MODEL` in `create_graph.py`)
- Search method weights and thresholds
- Knowledge graph parameters

Refer to the Settings tab in the GUI for all available customization options.

## Troubleshooting

- If you encounter issues with specific file formats, ensure you have the necessary libraries installed (e.g., `python-docx` for DOCX files, `PyPDF2` for PDF files).
- For performance issues, try adjusting the chunk size, batch size, or reducing the number of documents processed.
- If the knowledge graph is not providing useful results, you may need to adjust the entity extraction settings or increase the minimum entity occurrence.


For any issues or feature requests, please open an issue on the GitHub repository.


