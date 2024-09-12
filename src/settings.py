# Standard library imports
import json
import os
from typing import Any, Dict, List, Optional

class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_folder = os.path.join(self.project_root, "output")

        # Helper function to ensure correct path
        def ensure_output_path(path: str) -> str:
            return path if os.path.dirname(path) == self.output_folder else os.path.join(self.output_folder, os.path.basename(path))

        # Upload Settings
        self.file_chunk_size: int = 500
        self.file_overlap_size: int = 200

        # Embeddings Settings
        self.batch_size: int = 32
        self.embeddings_file_path: str = ensure_output_path("db_embeddings.pt")
        self.db_file_path: str = ensure_output_path("db.txt")

        # Graph Settings
        self.graph_chunk_size: int = 5000
        self.graph_overlap_size: int = 200
        self.nlp_model: str = "en_core_web_sm"
        self.similarity_threshold: float = 0.7
        self.min_entity_occurrence: int = 1
        self.enable_semantic_edges: bool = True
        self.knowledge_graph_file_path: str = ensure_output_path("knowledge_graph.json")

        # Model Settings
        self.max_history_length: int = 5
        self.conversation_context_size: int = 3
        self.update_threshold: int = 10
        self.ollama_model: str = "qwen2:1.5b-instruct-q8_0"
        self.llama_model: str = "llama-default"
        self.groq_model: str = "llama3-groq-8b-8192-tool-use-preview"
        self.gemini_model: str = "gemini-pro"
        self.cohere_model: str = "command-r-plus"
        self.temperature: float = 0.1
        self.model_name: str = "all-MiniLM-L6-v2"
        self.default_manager_model: Optional[str] = None
         # Re-ranker model setting
        self.reranker_model: str = "qwen2:1.5b-instruct-q8_0"  # Default value, can be changed

        # Embedding model settings
        self.default_embedding_class: str = "ollama"
        self.default_embedding_model: str = "chroma/all-minilm-l6-v2-f32:latest"
        self.sentence_transformer_model: str = "all-MiniLM-L6-v2"
        self.ollama_embedding_model: str = "chroma/all-minilm-l6-v2-f32:latest"

        # Knol Creation Settings
        self.num_questions: int = 8

        # Web Sum Settings
        self.web_sum_urls_to_crawl: int = 5
        self.summary_size: int = 5000
        self.final_summary_size: int = 10000

        # Web RAG Settings
        self.web_rag_urls_to_crawl: int = 5
        self.initial_context_size: int = 5
        self.web_rag_file: str = ensure_output_path("web_rag_qa.txt")
        self.web_rag_chunk_size: int = 500
        self.web_rag_overlap_size: int = 100

        # Search Settings
        self.top_k: int = 5
        self.rerank_top_k: int = 2  # Add this line
        self.entity_relevance_threshold: float = 0.5
        self.lexical_weight: float = 1.0
        self.semantic_weight: float = 1.0
        self.graph_weight: float = 1.0
        self.text_weight: float = 1.0
        self.enable_lexical_search: bool = True
        self.enable_semantic_search: bool = True
        self.enable_graph_search: bool = True
        self.enable_text_search: bool = True

        # Summarization settings
        self.summarization_chunk_size: int = 3000
        self.summarization_summary_size: int = 200
        self.summarization_combining_number: int = 3
        self.summarization_final_chunk_size: int = 300

        # File Settings
        self.results_file_path: str = ensure_output_path("results.txt")

        # Question Generation Settings
        self.initial_question_chunk_size: int = 1000
        self.question_chunk_levels: int = 3
        self.excluded_question_levels: List[int] = []
        self.questions_per_chunk: int = 3

        # Talk2URL Settings
        self.talk2url_limit_content_size: bool = True
        self.talk2url_content_size_per_url: int = 500

        # GitHub Settings
        self.file_analysis_limit: int = 2000

        # API Settings
        self.api_type: str = "ollama"

        # Config file path (in project root)
        self.config_file: str = ensure_output_path("config.json")

        # Dataset Generation Settings
        self.dataset_fields: List[str] = ["id", "question", "answer", "domain", "difficulty", "keywords", "language", "answer_type"]
        self.dataset_output_formats: List[str] = ["jsonl", "csv", "parquet"]
        self.dataset_output_file: str = os.path.join(self.output_folder, "qa_dataset")

        self.structured_data_db: str = os.path.join(self.output_folder, 'structured_data.db')

        # Additional Settings
        self.save_results_to_txt: bool = False


    def load_settings(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                saved_settings = json.load(f)
            for key, value in saved_settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            self.save_settings()

    def save_settings(self):
        settings_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        with open(self.config_file, 'w') as f:
            json.dump(settings_dict, f, indent=4)

    def apply_settings(self):
        # This method remains unchanged as it applies settings to various components
        pass

    def get_all_settings(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}

    def get_default_embedding_class(self) -> str:
        return self.default_embedding_class

    def get_default_embedding_model(self, embedding_class: str) -> str:
        if embedding_class == "sentence_transformers":
            return self.sentence_transformer_model
        elif embedding_class == "ollama":
            return self.ollama_embedding_model
        else:
            raise ValueError(f"Invalid embedding class: {embedding_class}")
    
    def get_default_model(self, api_type: str) -> str:
        api_models = {
            "ollama": self.ollama_model,
            "llama": self.llama_model,
            "groq": self.groq_model,
            "gemini": self.gemini_model,
            "cohere": self.cohere_model 
        }
        if api_type not in api_models:
            raise ValueError(f"Unknown API type: {api_type}")
        return api_models[api_type]

    def update_setting(self, key: str, value: Any):
        if hasattr(self, key):
            if key == "dataset_output_file":
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