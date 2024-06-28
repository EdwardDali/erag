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
        # General Settings
        self.chunk_size = 500
        self.overlap_size = 200
        self.batch_size = 32
        self.results_file_path = "results.txt"
        self.max_history_length = 5
        self.conversation_context_size = 3
        self.update_threshold = 10
        self.ollama_model = "phi3:instruct"
        self.temperature = 0.1
        self.num_questions = 8

        # File Paths
        self.db_file_path = "db.txt"
        self.embeddings_file_path = "db_embeddings.pt"
        self.knowledge_graph_file_path = "knowledge_graph.json"

        # Model Settings
        self.model_name = "all-MiniLM-L6-v2"
        self.nlp_model = "en_core_web_sm"

        # Search Settings
        self.top_k = 5
        self.entity_relevance_threshold = 0.5
        self.similarity_threshold = 0.7
        self.enable_family_extraction = True
        self.min_entity_occurrence = 1
        self.enable_semantic_edges = True

        # Search Weights
        self.lexical_weight = 1.0
        self.semantic_weight = 1.0
        self.graph_weight = 1.0
        self.text_weight = 1.0

        # Search Toggles
        self.enable_lexical_search = True
        self.enable_semantic_search = True
        self.enable_graph_search = True
        self.enable_text_search = True

        self.config_file = "config.json"

    def load_settings(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                saved_settings = json.load(f)
            for key, value in saved_settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            self.save_settings()  # Create default config file if it doesn't exist

    def save_settings(self):
        settings_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        with open(self.config_file, 'w') as f:
            json.dump(settings_dict, f, indent=4)

    def apply_settings(self):
        from file_processing import set_chunk_sizes
        from embeddings_utils import set_batch_size, set_model_name, set_db_file, set_embeddings_file
        from create_graph import set_graph_settings, set_nlp_model, set_knowledge_graph_file
        from run_model import RAGSystem
        from search_utils import SearchUtils

        # Apply settings to various components
        set_chunk_sizes(self.chunk_size, self.overlap_size)
        set_batch_size(self.batch_size)
        set_model_name(self.model_name)
        set_db_file(self.db_file_path)
        set_embeddings_file(self.embeddings_file_path)
        set_graph_settings(self.similarity_threshold, self.enable_family_extraction, self.min_entity_occurrence)
        set_nlp_model(self.nlp_model)
        set_knowledge_graph_file(self.knowledge_graph_file_path)

        # Update RAGSystem settings
        RAGSystem.MAX_HISTORY_LENGTH = self.max_history_length
        RAGSystem.CONVERSATION_CONTEXT_SIZE = self.conversation_context_size
        RAGSystem.UPDATE_THRESHOLD = self.update_threshold
        RAGSystem.OLLAMA_MODEL = self.ollama_model
        RAGSystem.TEMPERATURE = self.temperature
        RAGSystem.EMBEDDINGS_FILE = self.embeddings_file_path
        RAGSystem.DB_FILE = self.db_file_path
        RAGSystem.MODEL_NAME = self.model_name
        RAGSystem.KNOWLEDGE_GRAPH_FILE = self.knowledge_graph_file_path
        RAGSystem.RESULTS_FILE = self.results_file_path

        # Update SearchUtils settings
        SearchUtils.top_k = self.top_k
        SearchUtils.entity_relevance_threshold = self.entity_relevance_threshold
        SearchUtils.lexical_weight = self.lexical_weight
        SearchUtils.semantic_weight = self.semantic_weight
        SearchUtils.graph_weight = self.graph_weight
        SearchUtils.text_weight = self.text_weight
        SearchUtils.enable_lexical_search = self.enable_lexical_search
        SearchUtils.enable_semantic_search = self.enable_semantic_search
        SearchUtils.enable_graph_search = self.enable_graph_search
        SearchUtils.enable_text_search = self.enable_text_search

    def get_all_settings(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}

    def update_setting(self, key: str, value: Any):
        if hasattr(self, key):
            setattr(self, key, value)
            self.save_settings()
        else:
            raise AttributeError(f"Setting '{key}' does not exist")

    def reset_to_defaults(self):
        self._initialize()
        self.save_settings()

# Singleton instance
settings = Settings()
