# ERAG 

## Overview

You can use this application to:

1. Talk **privately** with your documents using Ollama or talk **fast** using Groq and others.
2. Perform Retrieval-Augmented Generation activities (**RAG**) using various APIs (Ollama, LLaMA, Groq, Gemini, Cohere). 
3. Perform AI powered **web search**.
4. Talk with a **specific url**.
5. Analyze and summarize **GitHub** repositories.
6. Do AI powered Exploratory Data Analysis (**EDA**) with AI generated Business Intelligence and insights on excels and csv (see some examples in images below). 
7. Utilize multiple AI models in **collaboration** (worker, supervisor, manager) for pre-defined complex tasks.
8. Generate specific knowledge entries (knol), or generate full size textbooks or use AI generated questions and answers to create datasets.


Thus, ERAG is an advanced system that combines lexical, semantic, text, and knowledge graph searches with conversation context to provide accurate and contextually relevant responses. This tool processes various document types, creates embeddings, builds knowledge graphs, and uses this information to answer user queries intelligently. It also includes modules for interacting with web content, GitHub repositories, performing exploratoru data analysis using various language models.

working on CPU only

tested on Windows 10

![ERAG GUI 1](https://github.com/EdwardDali/e-rag/blob/main/docs/gui1.PNG)
![ERAG GUI 2](https://github.com/EdwardDali/e-rag/blob/main/docs/gui2.PNG)
![ERAG GUI 3](https://github.com/EdwardDali/e-rag/blob/main/docs/gui3.PNG)
![ERAG GUI 3](https://github.com/EdwardDali/e-rag/blob/main/docs/da1.PNG)
![ERAG GUI 3](https://github.com/EdwardDali/e-rag/blob/main/docs/da2.PNG)
![ERAG GUI 3](https://github.com/EdwardDali/e-rag/blob/main/docs/da3.PNG)
![ERAG GUI 3](https://github.com/EdwardDali/e-rag/blob/main/docs/da4.PNG)

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
  
2. Install torch
CPU only
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```
4. Install required Python dependencies:
   ```
   pip install -r requirements.txt
   ```
   

5. Download required spaCy and NLTK models:
   ```
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt
   ```

6. Install Ollama (for using Ollama API and **for embeddings**) and install ollama models:
   - Linux/macOS: `curl https://ollama.ai/install.sh | sh`
   - Windows: Visit https://ollama.ai/download and follow installation instructions
  
   - ollama run gemma2:2b
   - ollama run chroma/all-minilm-l6-v2-f32:latest   - for embedddings

7. Set up environment variables:
   - Create a `.env` file in the project root
   - Add the following variables (if applicable):
     ```
      GROQ_API_KEY='your_groq_api_key_here'
      GEMINI_API_KEY='your_gemini_api_key_here'
      CO_API_KEY='your_cohere_api_key_here'
      GITHUB_TOKEN='your_github_token_here'
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

# Data Analytics with LLM used in this aplication

**Data Preprocessing Techniques**
- Data type conversion (e.g., string to datetime)
- Sorting data (for time series analysis)
- Standardization of numerical features

**Visualization Techniques**
- Various plot types: bar plots, pie charts, line plots, scatter plots, heatmaps
- Use of libraries like Matplotlib and Seaborn for visualization

**Performance Optimization**
- Use of timeouts to handle long-running operations

**Automated Reporting**
- Generation of text reports
- Creation of PDF reports with embedded visualizations

**Natural Language Processing (NLP)**
- Use of language models for interpreting analysis results
- Generating human-readable insights from data analysis

**Business Intelligence**
- Extraction of key findings and insights
- Formulation of business implications and recommendations based on data analysis

# Comprehensive List of Data Analytics Techniques

1. **Overall Table Analysis**
   - Row and column count
   - Data type distribution
   - Memory usage calculation
   - Missing value analysis
   - Unique value counts

2. **Statistical Analysis**
   - Descriptive statistics (mean, median, standard deviation, min, max)
   - Skewness and kurtosis calculation
   - Visualization of statistical measures

3. **Correlation Analysis**
   - Correlation matrix computation
   - Heatmap visualization of correlations
   - Identification of high correlations
   - Analysis of top positive and negative correlations

4. **Categorical Features Analysis**
   - Value counts for categorical variables
   - Bar plots and pie charts for category distribution
   - Analysis of top categories

5. **Distribution Analysis**
   - Histogram with Kernel Density Estimation (KDE)
   - Q-Q (Quantile-Quantile) plots
   - Normality assessment

6. **Outlier Detection**
   - Interquartile Range (IQR) method
   - Box plots for visualizing outliers
   - Calculation of outlier percentages

7. **Time Series Analysis**
   - Identification of date columns
   - Time series plotting
   - Trend visualization over time

8. **Feature Importance Analysis**
   - Random Forest Regressor for feature importance
   - Visualization of feature importance

9. **Dimensionality Reduction Analysis**
   - Principal Component Analysis (PCA)
   - Scree plot for explained variance
   - Cumulative explained variance plot

10. **Cluster Analysis**
    - K-means clustering
    - Elbow method for optimal cluster number
    - 2D projection of clusters using PCA

11. **Adaptive Multi-dimensional Pattern Recognition (AMPR)**
    - Standardization of numeric data
    - Principal Component Analysis (PCA) for dimensionality reduction
    - DBSCAN clustering with adaptive epsilon selection
    - Silhouette score optimization for clustering
    - Isolation Forest for anomaly detection
    - Visualization of clusters and anomalies in reduced dimensional space
    - Feature correlation analysis in the transformed space

12. **Enhanced Time Series Forecasting (ETSF)**
    - Augmented Dickey-Fuller test for stationarity
    - Seasonal decomposition of time series
    - ARIMA modeling with exogenous variables
    - Incorporation of lag features and Fourier terms for seasonality
    - Time series cross-validation
    - Forecast evaluation using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    - Visualization of observed data, trend, seasonality, residuals, and forecasts

13. **Value Counts Analysis**
    - Pie chart visualization of categorical variable distributions

14. **Grouped Summary Statistics**
    - Calculation of summary statistics grouped by categorical variables

15. **Frequency Distribution Analysis**
    - Histogram plots with Kernel Density Estimation (KDE)

16. **KDE Plot Analysis**
    - Kernel Density Estimation plots for continuous variables

17. **Violin Plot Analysis**
    - Visualization of data distribution across categories

18. **Pair Plot Analysis**
    - Scatter plots for all pairs of numerical variables

19. **Box Plot Analysis**
    - Visualization of data distribution and outliers

20. **Scatter Plot Analysis**
    - Visualization of relationships between pairs of variables

21. **Correlation Network Analysis**
    - Network graph of correlations between variables

22. **Q-Q Plot Analysis**
    - Quantile-Quantile plots for assessing normality

23. **Factor Analysis**
    - Identification of underlying factors in the data

24. **Multidimensional Scaling (MDS)**
    - Visualization of high-dimensional data in lower dimensions

25. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
    - Non-linear dimensionality reduction for data visualization

26. **Conditional Plots**
    - Visualization of relationships between variables conditioned on categories

27. **Individual Conditional Expectation (ICE) Plots**
    - Visualization of model predictions for individual instances

28. **STL Decomposition Analysis**
    - Seasonal and Trend decomposition using Loess

29. **Autocorrelation Plots**
    - Visualization of serial correlation in time series data

30. **Bayesian Networks**
    - Probabilistic graphical models for representing dependencies

31. **Isolation Forest**
    - Anomaly detection using isolation trees

32. **One-Class SVM**
    - Anomaly detection using support vector machines

33. **Local Outlier Factor (LOF)**
    - Anomaly detection based on local density deviation

34. **Robust PCA**
    - Principal Component Analysis robust to outliers

35. **Bayesian Change Point Detection**
    - Detection of changes in time series data

36. **Hidden Markov Models (HMMs)**
    - Modeling of sequential data with hidden states

37. **Dynamic Time Warping (DTW)**
    - Measurement of similarity between temporal sequences

38. **Matrix Profile**
    - Time series motif discovery and anomaly detection

39. **Ensemble Anomaly Detection**
    - Combination of multiple anomaly detection methods

40. **Gaussian Mixture Models (GMM)**
    - Probabilistic model for representing normally distributed subpopulations

41. **Expectation-Maximization Algorithm**
    - Iterative method for finding maximum likelihood estimates

42. **Statistical Process Control (SPC) Charts**
    - CUSUM (Cumulative Sum) and EWMA (Exponentially Weighted Moving Average) charts

43. **KDE Anomaly Detection**
    - Anomaly detection using Kernel Density Estimation

44. **Hotelling's T-squared Analysis**
    - Multivariate statistical process control

45. **Breakdown Point Analysis**
    - Assessment of the robustness of statistical estimators

46. **Chi-Square Test Analysis**
    - Test for independence between categorical variables

47. **Simple Thresholding Analysis**
    - Basic method for anomaly detection

48. **Lilliefors Test Analysis**
    - Test for normality of the data

49. **Jarque-Bera Test Analysis**
    - Test for normality based on skewness and kurtosis

50. **Cook's Distance Analysis**
    - Identification of influential data points in regression analysis

51. **Hampel Filter Analysis**
    - Robust outlier detection in time series data

52. **GESD Test Analysis**
    - Generalized Extreme Studentized Deviate Test for outliers

53. **Dixon's Q Test Analysis**
    - Identification of outliers in small sample sizes

54. **Peirce's Criterion Analysis**
    - Method for eliminating outliers from data samples

55. **Thompson Tau Test Analysis**
    - Statistical technique for detecting a single outlier in a dataset

56. **Sequence Alignment and Matching**
    - Techniques for comparing and aligning sequences (e.g., in text data)

57. **Conformal Anomaly Detection**
    - Anomaly detection with statistical guarantees

58. **Trend Analysis**
    - Time series plotting
    - Trend visualization over time
    - Calculation of trend statistics (start, end, change, percent change)

59. **Variance Analysis**
    - Calculation of variance for numeric columns
    - Visualization of variance across different variables

60. **Regression Analysis**
    - Simple linear regression for pairs of numeric columns
    - Calculation of R-squared, coefficients, and intercepts

61. **Stratification Analysis**
    - Data grouping and aggregation
    - Box plot visualization of stratified data

62. **Gap Analysis**
    - Comparison of current values to target values
    - Calculation of gaps and gap percentages

63. **Duplicate Detection**
    - Identification of duplicate rows
    - Calculation of duplicate percentages

64. **Process Mining**
    - Analysis of process sequences
    - Visualization of top process flows

65. **Data Validation Techniques**
    - Checking for missing values, negative values, and out-of-range values

66. **Risk Scoring Models**
    - Development of simple risk scoring models
    - Visualization of risk distributions

67. **Fuzzy Matching**
    - Identification of similar text entries
    - Calculation of string similarity ratios

68. **Continuous Auditing Techniques**
    - Ongoing analysis of data for anomalies and outliers

69. **Sensitivity Analysis**
    - Assessment of impact of variable changes on outcomes

70. **Scenario Analysis**
    - Creation and comparison of different business scenarios

71. **Monte Carlo Simulation**
    - Generation of multiple simulated scenarios
    - Analysis of probabilistic outcomes

72. **KPI Analysis**
    - Definition and calculation of Key Performance Indicators

73. **ARIMA (AutoRegressive Integrated Moving Average) Analysis**
    - Time series forecasting
    - Model parameter optimization

74. **Auto ARIMAX Analysis**
    - ARIMA with exogenous variables
    - Automatic model selection

75. **Exponential Smoothing**
    - Time series smoothing and forecasting
    - Handling of trends and seasonality

76. **Holt-Winters Method**
    - Triple exponential smoothing
    - Handling of level, trend, and seasonal components

77. **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) Analysis**
    - Seasonal time series forecasting
    - Incorporation of external factors

78. **Gradient Boosting for Time Series**
    - Machine learning approach to time series forecasting
    - Feature importance analysis in time series context

79. **Fourier Analysis**
    - Frequency domain analysis of time series
    - Identification of dominant frequencies

80. **Trend Extraction**
    - Separation of trend and cyclical components
    - Use of Hodrick-Prescott filter

81. **Cross-Sectional Regression**
    - Analysis of relationships between variables at a single point in time

82. **Ensemble Time Series**
    - Combination of multiple time series models
    - Improved forecast accuracy through model averaging

83. **Bootstrapping Time Series**
    - Resampling techniques for time series data
    - Estimation of forecast uncertainty

84. **Theta Method**
    - Decomposition-based forecasting method
    - Combination of linear regression and Simple Exponential Smoothing

This comprehensive list covers a wide range of data analytics techniques, from basic statistical analysis to advanced machine learning and time series forecasting methods, demonstrating a thorough approach to exploratory and innovative data analysis.

## Data Quality Checks in structured data

1. **Missing Values**
   - Checks for null or empty values in columns.

2. **Data Type Mismatches**
   - Verifies if the data type of values matches the expected column type.

3. **Duplicate Records**
   - Identifies duplicate entries across all columns in a table.

4. **Inconsistent Formatting**
   - Detects inconsistent date formats within a column.

5. **Outliers**
   - Identifies statistical outliers in numeric columns using the Interquartile Range (IQR) method.

6. **Whitespace Issues**
   - Checks for leading, trailing, or excessive whitespace in text columns.

7. **Special Characters**
   - Detects the presence of special characters in text columns.

8. **Inconsistent Capitalization**
   - Identifies inconsistent use of uppercase, lowercase, or title case in text columns.

9. **Possible Data Truncation**
   - Checks for values that are close to the maximum observed length in a column, which might indicate truncation.

10. **High Frequency Values**
    - Identifies values that appear with unusually high frequency (>90%) in a column.

11. **Suspicious Date Range**
    - Checks for dates outside a reasonable range (e.g., before 1900 or far in the future).

12. **Large Numeric Range**
    - Detects numeric columns with an unusually large range of values.

13. **Very Short Strings**
    - Identifies strings that are unusually short (less than 2 characters).

14. **Very Long Strings**
    - Identifies strings that are unusually long (more than 255 characters).

15. **Invalid Email Format**
    - Checks if email addresses conform to a standard format.

16. **Non-unique Values**
    - Identifies columns with non-unique values where uniqueness might be expected.

17. **Invalid Foreign Keys**
    - Checks for foreign key violations in the database.

18. **Date Inconsistency**
    - Verifies logical relationships between date columns (e.g., start date before end date).

19. **Logical Relationship Violations**
    - Checks for violations of expected relationships between columns (e.g., a total column should equal the sum of its parts).

20. **Pattern Mismatch**
    - Verifies if values in certain columns match expected patterns (e.g., phone numbers, zip codes, URLs).

## Troubleshooting

- Ensure all dependencies are correctly installed.
- Check console output for detailed error messages (color-coded for easier identification).
- Verify API keys and tokens are correctly set in the `.env` file.
- For performance issues, adjust chunk sizes, batch processing parameters, or consider using a GPU.
- If using local LLaMA.cpp servers, ensure the correct model files are available and properly configured.

## Contact

For support or queries, please open an issue on the GitHub repository or contact the project maintainers.
