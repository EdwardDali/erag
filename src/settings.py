import json
import os
from typing import Dict, Any

class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Get the project root directory (one level up from src)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Output folder setting
        self.output_folder = os.path.join(self.project_root, "output")

        # Helper function to ensure correct path
        def ensure_output_path(path):
            if os.path.dirname(path) == self.output_folder:
                return path
            return os.path.join(self.output_folder, os.path.basename(path))

        # Upload Settings
        self.file_chunk_size: int = 500
        self.file_overlap_size: int = 200

        # Embeddings Settings
        self.batch_size = 32
        self.embeddings_file_path = ensure_output_path("db_embeddings.pt")
        self.db_file_path = ensure_output_path("db.txt")

        # Graph Settings
        self.graph_chunk_size: int = 5000
        self.graph_overlap_size: int = 200
        self.nlp_model = "en_core_web_sm"
        self.similarity_threshold = 0.7
        self.min_entity_occurrence = 1
        self.enable_semantic_edges = True
        self.knowledge_graph_file_path = ensure_output_path("knowledge_graph.json")

        # Model Settings
        self.max_history_length = 5
        self.conversation_context_size = 3
        self.update_threshold = 10
        self.ollama_model = "qwen2:1.5b-instruct-q8_0"
        self.llama_model = "llama-default"
        self.groq_model = "llama3-groq-8b-8192-tool-use-preview"
        self.gemini_model = "gemini-pro"  # Add default Gemini model
        self.temperature = 0.1
        self.model_name = "all-MiniLM-L6-v2"
        self.default_manager_model = None

        # Embedding model settings
        self.default_embedding_class = "ollama"  # Changed to ollama
        self.default_embedding_model = "chroma/all-minilm-l6-v2-f32:latest"  # Changed to Ollama model
        self.sentence_transformer_model = "all-MiniLM-L6-v2"  # Keep this for when sentence_transformers is selected
        self.ollama_embedding_model = "chroma/all-minilm-l6-v2-f32:latest"
        

        # Knol Creation Settings
        self.num_questions = 8

        # Web Sum Settings
        self.web_sum_urls_to_crawl = 5
        self.summary_size = 5000
        self.final_summary_size = 10000

        # Web RAG Settings
        self.web_rag_urls_to_crawl = 5
        self.initial_context_size = 5
        self.web_rag_file = ensure_output_path("web_rag_qa.txt")
        self.web_rag_chunk_size = 500
        self.web_rag_overlap_size = 100

        # Search Settings
        self.top_k = 5
        self.entity_relevance_threshold = 0.5
        self.lexical_weight = 1.0
        self.semantic_weight = 1.0
        self.graph_weight = 1.0
        self.text_weight = 1.0
        self.enable_lexical_search = True
        self.enable_semantic_search = True
        self.enable_graph_search = True
        self.enable_text_search = True

        # Summarization settings
        self.summarization_chunk_size: int = 3000
        self.summarization_summary_size: int = 200
        self.summarization_combining_number: int = 3
        self.summarization_final_chunk_size: int = 300

        # File Settings
        self.results_file_path = ensure_output_path("results.txt")

        # Question Generation Settings
        self.initial_question_chunk_size: int = 1000
        self.question_chunk_levels: int = 3
        self.excluded_question_levels: list = []
        self.questions_per_chunk: int = 3

        # Talk2URL Settings
        self.talk2url_limit_content_size = True
        self.talk2url_content_size_per_url = 500

        # GitHub Settings
        self.file_analysis_limit = 2000

        #self.groq_api_key = ""
        #self.github_token = ""


        # API Settings
        self.api_type = "ollama"

        # Config file path (in project root)
        self.config_file = ensure_output_path("config.json")

        # Dataset Generation Settings
        self.dataset_fields = ["id", "question", "answer", "domain", "difficulty", "keywords", "language", "answer_type"]
        self.dataset_output_formats = ["jsonl", "csv", "parquet"]
        self.dataset_output_file = os.path.join(self.output_folder, "qa_dataset")

        self.structured_data_db = os.path.join(self.output_folder, 'structured_data.db')

         # Add this new setting
        self.save_results_to_txt = False  # Default to False


       
    def load_settings(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                saved_settings = json.load(f)
            for key, value in saved_settings.items():
                if hasattr(self, key):
                    if key in ['dataset_fields', 'dataset_output_formats']:
                        setattr(self, key, value if isinstance(value, list) else value.split(','))
                    elif key == 'default_manager_model':
                        setattr(self, key, None if value == 'None' else value)
                    else:
                        setattr(self, key, value)
        else:
            self.save_settings()

    def save_settings(self):
        settings_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        # Convert list settings to comma-separated strings for JSON serialization
        settings_dict['dataset_fields'] = ','.join(self.dataset_fields)
        settings_dict['dataset_output_formats'] = ','.join(self.dataset_output_formats)
        settings_dict['default_manager_model'] = 'None' if self.default_manager_model is None else self.default_manager_model
        with open(self.config_file, 'w') as f:
            json.dump(settings_dict, f, indent=4)

    def apply_settings(self):
        # This method remains unchanged as it applies settings to various components
        # You may need to update this method if you've changed how settings are applied in your application
        pass


    def get_all_settings(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}


    def get_default_embedding_model(self, embedding_class):
        if embedding_class == "sentence_transformers":
            return self.default_embedding_model  # Change this line
        elif embedding_class == "ollama":
            return self.ollama_embedding_model
        else:
            raise ValueError(f"Invalid embedding class: {embedding_class}")
    
    def get_default_model(self, api_type: str) -> str:
        if api_type == "ollama":
            return self.ollama_model
        elif api_type == "llama":
            return self.llama_model
        elif api_type == "groq":
            return self.groq_model
        elif api_type == "gemini":  # Add this condition
            return self.gemini_model
        else:
            raise ValueError(f"Unknown API type: {api_type}")

    def update_setting(self, key: str, value: Any):
        if hasattr(self, key):
            if key == "dataset_output_file":
                # Ensure the dataset output file is always in the output folder
                value = os.path.join(self.output_folder, os.path.basename(value))
            setattr(self, key, value)
            self.save_settings()
        else:
            raise AttributeError(f"Setting '{key}' does not exist")

    def reset_to_defaults(self):
        self._initialize()
        self.save_settings()

# Singleton instance
settings = Settings()
