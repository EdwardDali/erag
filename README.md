## Overview

This RAG (Retrieval-Augmented Generation) tool is a sophisticated system that combines lexical, semantic, text, and knowledge graph searches with conversation context to provide accurate and contextually relevant responses. The tool processes various document types, creates embeddings, builds a knowledge graph, and uses this information to answer user queries intelligently.

## Features

1. **Multi-modal Search**: Incorporates lexical, semantic, text, and knowledge graph searches.
2. **Conversation Context**: Maintains context across multiple interactions for more coherent conversations.
3. **Document Processing**: Handles various document types including DOCX, JSON, PDF, and plain text.
4. **Embedding Generation**: Creates and manages embeddings for efficient semantic search.
5. **Knowledge Graph**: Builds and utilizes a knowledge graph for enhanced information retrieval.
6. **Customizable Settings**: Offers a wide range of configurable parameters through a GUI.

![Alt text](https://github.com/EdwardDali/e-rag/blob/main/docs/gui.PNG)
![Alt text](https://github.com/EdwardDali/e-rag/blob/main/docs/logical.PNG)
![Alt text](https://github.com/EdwardDali/e-rag/blob/main/docs/user_flow.PNG)

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
