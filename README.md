# ERAG 

## Overview


ERAG is an advanced system that combines lexical, semantic, text, and knowledge graph searches with conversation context to provide accurate and contextually relevant responses. This tool processes various document types, creates embeddings, builds knowledge graphs, and uses this information to answer user queries intelligently. It also includes modules for interacting with web content, GitHub repositories, performing exploratoru data analysis using various language models.

working on CPU only

tested on Windows 10

![ERAG GUI 1](https://github.com/EdwardDali/e-rag/blob/main/docs/gui1.PNG)
![ERAG GUI 2](https://github.com/EdwardDali/e-rag/blob/main/docs/gui2.PNG)
![ERAG GUI 3](https://github.com/EdwardDali/e-rag/blob/main/docs/gui3.PNG)

## Key Features

1. **Multi-modal Document Processing**: Handles DOCX, PDF, TXT, and JSON files with intelligent chunking and table of contents extraction.
2. **Advanced Embedding Generation**: Creates and manages embeddings for efficient semantic search using sentence transformers, with support for batch processing and caching.
3. **Knowledge Graph Creation**: Builds and utilizes a knowledge graph for enhanced information retrieval using spaCy and NetworkX.
4. **Multi-API Support**: Integrates with Ollama, LLaMA, and Groq APIs for flexible language model deployment.
5. **Retrieval-Augmented Generation (RAG)**: Combines retrieved context with language model capabilities for improved responses.
6. **Web Content Processing**: Implements real-time web crawling, content extraction, and summarization.
7. **Query Routing**: Intelligently routes queries to the most appropriate subsystem based on content relevance and query complexity.
8. **Server Management**: Provides a GUI for managing local LLaMA.cpp servers, including model selection and server configuration.
9. **Customizable Settings**: Offers a wide range of configurable parameters through a graphical user interface and a centralized settings management system.
10. **Advanced Search Utilities**: Implements lexical, semantic, graph-based, and text search methods with configurable weights and thresholds.
11. **Conversation Context Management**: Maintains and utilizes conversation history for more coherent and contextually relevant responses.
12. **GitHub Repository Analysis**: Provides tools for analyzing and summarizing GitHub repositories, including code analysis, dependency checking, and code smell detection.
13. **Web Summarization**: Offers capabilities to summarize web content based on user queries.
14. **Interactive Model Chat**: Allows direct interaction with various language models for general conversation and task completion.
15. **Debug and Logging Capabilities**: Provides comprehensive logging and debug information for system operations and search results.
16. **Color-coded Console Output**: Enhances user experience with color-coded console messages for different types of information.
17. **Structured Data Analysis**: Implements tools for analyzing structured data stored in SQLite databases, including value counts, grouped summary statistics, and advanced visualizations.
18. **Exploratory Data Analysis (EDA)**: Offers comprehensive EDA capabilities, including distribution analysis, correlation studies, and outlier detection.
19. **Advanced Data Visualization**: Generates various types of plots and charts, such as histograms, box plots, scatter plots, and pair plots for in-depth data exploration.
20. **Statistical Analysis**: Provides tools for conducting statistical tests and generating statistical summaries of the data.
21. **Multi-Model Collaboration**: Utilizes worker, supervisor, and manager AI models to create, improve, and evaluate knowledge entries.
22. **Iterative Knowledge Refinement**: Implements an iterative process of knowledge creation, improvement, and evaluation to achieve high-quality, comprehensive knowledge entries.
23. **Automated Quality Assessment**: Includes an automated grading system for evaluating the quality of generated knowledge entries.
24. **Structured Knowledge Format**: Enforces a consistent, hierarchical structure for knowledge entries to ensure comprehensive coverage and easy navigation.
25. **PDF Report Generation**: Automatically generates comprehensive PDF reports summarizing the results of various analyses, including visualizations and AI-generated interpretations.

## System Architecture

ERAG is composed of several interconnected components:

1. **File Processing**: Handles document upload and processing, including table of contents extraction.
2. **Embedding Utilities**: Manages the creation and retrieval of document embeddings.
3. **Knowledge Graph**: Creates and maintains a graph representation of document content and entity relationships.
4. **RAG System**: Implements the core retrieval-augmented generation functionality.
5. **Query Router**: Analyzes queries and routes them to the appropriate subsystem.
6. **Server Manager**: Handles the configuration and management of local LLaMA.cpp servers.
7. **Settings Manager**: Centralizes system configuration and provides easy customization options.
8. **Search Utilities**: Implements various search methods to retrieve relevant context for queries.
9. **API Integration**: Provides a unified interface for interacting with different language model APIs.
10. **Talk2Model**: Enables direct interaction with language models for general queries and tasks.
11. **Talk2URL**: Allows interaction with web content, including crawling and question-answering based on web pages.
12. **WebRAG**: Implements a web-based retrieval-augmented generation system for answering queries using internet content.
13. **WebSum**: Provides tools for summarizing web content based on user queries.
14. **Talk2Git**: Offers capabilities for analyzing and summarizing GitHub repositories.
15. **Talk2SD**: Implements tools for interacting with and analyzing structured data stored in SQLite databases.
16. **Exploratory Data Analysis (EDA)**: Provides comprehensive EDA capabilities, including various statistical analyses and visualizations.
17. **Advanced Exploratory Data Analysis**: Offers more sophisticated data analysis techniques, including machine learning-based approaches and complex visualizations.
18. **Self Knol Creator**: Manages the process of creating, improving, and evaluating comprehensive knowledge entries on specific subjects.
19. **Innovative Exploratory Data Analysis**: while the individual analytical techniques are not particularly innovative on their own, the overall system's attempt to automate the entire process from data analysis to interpretation and reporting, using multiple AI models, represents a more innovative approach to data analysis automation. However, the true innovation and effectiveness of this system would depend heavily on the quality of the AI models used.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/EdwardDali/erag.git && cd erag
   ```
2. Install torch
CPU only
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

3. Install required Python dependencies:
   ```
   pip install -r requirements.txt
   ```
   

4. Download required spaCy and NLTK models:
   ```
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt
   ```

5. Install Ollama (for using Ollama API and **for embeddings**) and install ollama models:
   - Linux/macOS: `curl https://ollama.ai/install.sh | sh`
   - Windows: Visit https://ollama.ai/download and follow installation instructions
  
   - ollama run gemma2:2b
   - ollama run chroma/all-minilm-l6-v2-f32:latest   - for embedddings

6. Set up environment variables:
   - Create a `.env` file in the project root
   - Add the following variables (if applicable):
     ```
     GROQ_API_KEY=your_groq_api_key
     GITHUB_TOKEN=your_github_token
     GEMINI_API_KEY=your_gemini_api_key
     ```

## Usage

1. Start the ERAG GUI:
   ```
   python main.py
   ```

2. Use the GUI to:
   - Upload and process documents
   - Generate embeddings
   - Create knowledge graphs
   - Configure system settings
   - Manage local LLaMA.cpp servers
   - Run various RAG operations (Talk2Doc, WebRAG, etc.)
   - Analyze structured data and perform exploratory data analysis
   - Create and refine comprehensive knowledge entries (Self Knols)

## Configuration

Customize ERAG's behavior through the Settings tab in the GUI or by modifying `settings.py`. Key configurable options include:

- Chunk sizes and overlap for document processing
- Embedding model selection and batch size
- Knowledge graph parameters (similarity threshold, minimum entity occurrence)
- API selection (Ollama, LLaMA, Groq) and model choices
- Search method weights and thresholds
- RAG system parameters (conversation context size, update threshold)
- Server configuration for local LLaMA.cpp instances
- Web crawling and summarization settings
- GitHub analysis parameters
- Data analysis and visualization parameters
- Self Knol creation parameters (iteration thresholds, quality assessment criteria)

## Advanced Features

- **Query Routing**: Automatically determines the best subsystem to handle a query based on its content and complexity.
- **Hybrid Search**: Combines lexical, semantic, graph-based, and text search methods for comprehensive context retrieval.
- **Dynamic Embedding Updates**: Automatically updates embeddings as new content is added to the system.
- **Conversation Context Management**: Maintains a sliding window of recent conversation history for improved contextual understanding.
- **Web Content Analysis**: Crawls and analyzes web pages to answer queries and generate summaries.
- **GitHub Repository Analysis**: Provides static code analysis, dependency checking, project summarization, and code smell detection for GitHub repositories.
- **Multi-model Support**: Allows interaction with various language models through a unified interface.
- **Structured Data Analysis**: Offers tools for analyzing and visualizing structured data stored in SQLite databases.
- **Advanced Exploratory Data Analysis**: Provides comprehensive EDA capabilities, including statistical analyses, machine learning techniques, and various types of data visualizations.
- **Automated Report Generation**: Generates detailed PDF reports summarizing the results of data analyses, complete with visualizations and AI-generated interpretations.
- **Self Knol Creation**: Utilizes a multi-model approach to create, refine, and evaluate comprehensive knowledge entries on specific subjects.
- **Iterative Knowledge Improvement**: Implements an iterative process with AI-driven feedback and improvement cycles to enhance the quality and depth of knowledge entries.

# Comprehensive List of Data Analytics Techniques

1. Basic Statistics
2. Data Types and Missing Values Analysis
3. Numerical Features Distribution
4. Correlation Analysis
5. Categorical Features Analysis
6. Outlier Detection
7. Data Quality Report
8. Hypothesis Testing Suggestions
9. Value Counts Analysis
10. Grouped Summary Statistics
11. Frequency Distribution Analysis
12. KDE Plot Analysis
13. Violin Plot Analysis
14. Pair Plot Analysis
15. Box Plot Analysis
16. Scatter Plot Analysis
17. Time Series Analysis
18. Feature Importance Analysis
19. Principal Component Analysis (PCA)
20. Cluster Analysis
21. Correlation Network Analysis
22. Q-Q Plot Analysis
23. Parallel Coordinates Plot
24. Andrews Curves
25. Radar Charts
26. Sankey Diagrams
27. Bubble Charts
28. Geographical Plots
29. Word Clouds
30. Hierarchical Clustering Dendrogram
31. Empirical Cumulative Distribution Function (ECDF) Plots
32. Ridgeline Plots
33. Hexbin Plots
34. Mosaic Plots
35. Lag Plots
36. Shapley Value Analysis
37. Partial Dependence Plots
38. Factor Analysis
39. Multidimensional Scaling (MDS)
40. t-Distributed Stochastic Neighbor Embedding (t-SNE)
41. Conditional Plots
42. Individual Conditional Expectation (ICE) Plots
43. Time Series Decomposition
44. Autocorrelation Plots
45. Bayesian Networks
46. Isolation Forest
47. One-Class SVM
48. Local Outlier Factor (LOF)
49. Robust Principal Component Analysis (RPCA)
50. Bayesian Change Point Detection
51. Hidden Markov Models (HMMs)
52. Dynamic Time Warping (DTW)
53. Matrix Profile
54. Ensemble Anomaly Detection
55. Gaussian Mixture Models (GMM)
56. Expectation-Maximization Algorithm
57. Statistical Process Control (SPC) Charts
58. Z-Score and Modified Z-Score Analysis
59. Mahalanobis Distance
60. Box-Cox Transformation
61. Grubbs' Test
62. Chauvenet's Criterion
63. Benford's Law Analysis
64. Forensic Accounting Techniques
65. Network Analysis for Fraud Detection
66. Sequence Alignment and Matching
67. Conformal Anomaly Detection
68. Cook's Distance Analysis
69. STL Decomposition Analysis
70. Hampel Filter Analysis
71. GESD Test Analysis
72. Dixon's Q Test Analysis
73. Peirce's Criterion Analysis
74. Thompson Tau Test Analysis
75. Control Charts Analysis (CUSUM, EWMA)
76. KDE Anomaly Detection Analysis
77. Hotelling's T-squared Analysis
78. Breakdown Point Analysis
79. Chi-Square Test Analysis
80. Simple Thresholding Analysis
81. Lilliefors Test Analysis
82. Jarque-Bera Test Analysis
83. Adaptive Multi-dimensional Pattern Recognition (AMPR)
84. Enhanced Time Series Forecasting (ETSF)
85. Trend Analysis
86. Variance Analysis
87. Regression Analysis (simple and multiple)
88. Stratification Analysis
89. Gap Analysis
90. Duplicate Detection
91. Process Mining
92. Data Validation Techniques
93. Risk Scoring Models
94. Fuzzy Matching
95. Continuous Auditing Techniques
96. Sensitivity Analysis
97. Scenario Analysis
98. Monte Carlo Simulation
99. Key Performance Indicator (KPI) Analysis

## Troubleshooting

- Ensure all dependencies are correctly installed.
- Check console output for detailed error messages (color-coded for easier identification).
- Verify API keys and tokens are correctly set in the `.env` file.
- For performance issues, adjust chunk sizes, batch processing parameters, or consider using a GPU.
- If using local LLaMA.cpp servers, ensure the correct model files are available and properly configured.

## Contact

For support or queries, please open an issue on the GitHub repository or contact the project maintainers.
