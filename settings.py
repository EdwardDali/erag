import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import json
import os

# Import necessary functions from other modules
# Note: You may need to adjust these imports based on your project structure
from file_processing import set_chunk_sizes
from embeddings_utils import set_batch_size, set_model_name, set_db_file, set_embeddings_file
from create_graph import set_graph_settings, set_nlp_model, set_knowledge_graph_file
from run_model import set_max_history_length, set_conversation_context_size, set_update_threshold, set_ollama_model, set_temperature
from search_utils import set_top_k, set_entity_relevance_threshold, set_search_weights, set_search_toggles

class SettingsManager:
    def __init__(self, notebook):
        self.notebook = notebook
        self.settings_tab = ttk.Frame(self.notebook)

        # General Settings
        self.chunk_size_var = tk.IntVar(value=500)
        self.overlap_size_var = tk.IntVar(value=200)
        self.batch_size_var = tk.IntVar(value=32)
        self.results_file_path_var = tk.StringVar(value="results.txt")
        self.max_history_length_var = tk.IntVar(value=5)
        self.conversation_context_size_var = tk.IntVar(value=3)
        self.update_threshold_var = tk.IntVar(value=10)
        self.ollama_model_var = tk.StringVar(value="phi3:instruct")
        self.temperature_var = tk.DoubleVar(value=0.1)
        self.num_questions_var = tk.IntVar(value=8)

        # Advanced Settings
        self.similarity_threshold_var = tk.DoubleVar(value=0.7)
        self.enable_family_extraction_var = tk.BooleanVar(value=True)
        self.min_entity_occurrence_var = tk.IntVar(value=1)
        self.model_name_var = tk.StringVar(value="all-MiniLM-L6-v2")
        self.nlp_model_var = tk.StringVar(value="en_core_web_sm")
        self.db_file_path_var = tk.StringVar(value="db.txt")
        self.embeddings_file_path_var = tk.StringVar(value="db_embeddings.pt")
        self.knowledge_graph_file_path_var = tk.StringVar(value="knowledge_graph.json")
        self.enable_semantic_edges_var = tk.BooleanVar(value=True)
        self.top_k_var = tk.IntVar(value=5)
        self.entity_relevance_threshold_var = tk.DoubleVar(value=0.5)

        # Search Weights
        self.lexical_weight_var = tk.DoubleVar(value=1.0)
        self.semantic_weight_var = tk.DoubleVar(value=1.0)
        self.graph_weight_var = tk.DoubleVar(value=1.0)
        self.text_weight_var = tk.DoubleVar(value=1.0)

        # Search Toggles
        self.enable_lexical_search_var = tk.BooleanVar(value=True)
        self.enable_semantic_search_var = tk.BooleanVar(value=True)
        self.enable_graph_search_var = tk.BooleanVar(value=True)
        self.enable_text_search_var = tk.BooleanVar(value=True)

        self.config_file = "current_config.json"
        self.load_current_config()

        self.create_settings_tab()

    def create_settings_tab(self):
        self.create_settings_management_frame()
        
        # Create two columns
        left_column = ttk.Frame(self.settings_tab)
        left_column.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        right_column = ttk.Frame(self.settings_tab)
        right_column.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # General Settings (Left Column)
        self.create_general_settings_frame(left_column)
        self.create_search_settings_frame(left_column)

        # Advanced Settings (Right Column)
        self.create_advanced_settings_frame(right_column)
        self.create_file_path_settings_frame(right_column)

        apply_button = tk.Button(self.settings_tab, text="Apply Settings", command=self.apply_settings)
        apply_button.pack(pady=10)

    def create_settings_management_frame(self):
        management_frame = tk.LabelFrame(self.settings_tab, text="Settings Management")
        management_frame.pack(fill="x", padx=10, pady=5)

        use_default_button = tk.Button(management_frame, text="Use Default Settings", command=self.use_default_settings)
        use_default_button.pack(side="left", padx=5, pady=5)

        save_template_button = tk.Button(management_frame, text="Save Settings as Template", command=self.save_settings_template)
        save_template_button.pack(side="left", padx=5, pady=5)

        load_template_button = tk.Button(management_frame, text="Load Settings Template", command=self.load_settings_template)
        load_template_button.pack(side="left", padx=5, pady=5)

    def create_general_settings_frame(self, parent):
        frame = tk.LabelFrame(parent, text="General Settings")
        frame.pack(fill="x", padx=5, pady=5)

        settings = [
            ("Chunk Size:", self.chunk_size_var),
            ("Overlap Size:", self.overlap_size_var),
            ("Batch Size:", self.batch_size_var),
            ("Max History Length:", self.max_history_length_var),
            ("Conversation Context Size:", self.conversation_context_size_var),
            ("Update Threshold:", self.update_threshold_var),
            ("Ollama Model:", self.ollama_model_var),
            ("Temperature:", self.temperature_var),
            ("Number of Questions:", self.num_questions_var), 
        ]

        for i, (label, var) in enumerate(settings):
            tk.Label(frame, text=label).grid(row=i, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(frame, textvariable=var, width=20).grid(row=i, column=1, sticky="w", padx=5, pady=2)

        tk.Label(frame, text="Results File:").grid(row=len(settings), column=0, sticky="e", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.results_file_path_var, width=20).grid(row=len(settings), column=1, sticky="w", padx=5, pady=2)
        tk.Button(frame, text="Browse", command=lambda: self.browse_file_path(self.results_file_path_var)).grid(row=len(settings), column=2, padx=5, pady=2)

    def create_search_settings_frame(self, parent):
        frame = tk.LabelFrame(parent, text="Search Settings")
        frame.pack(fill="x", padx=5, pady=5)

        toggles = [
            ("Enable Lexical Search", self.enable_lexical_search_var),
            ("Enable Semantic Search", self.enable_semantic_search_var),
            ("Enable Graph Search", self.enable_graph_search_var),
            ("Enable Text Search", self.enable_text_search_var),
        ]

        for i, (label, var) in enumerate(toggles):
            tk.Checkbutton(frame, text=label, variable=var).grid(row=i, column=0, sticky="w", padx=5, pady=2)

        weights = [
            ("Lexical Weight:", self.lexical_weight_var),
            ("Semantic Weight:", self.semantic_weight_var),
            ("Graph Weight:", self.graph_weight_var),
            ("Text Weight:", self.text_weight_var),
        ]

        for i, (label, var) in enumerate(weights):
            tk.Label(frame, text=label).grid(row=i, column=1, sticky="e", padx=5, pady=2)
            tk.Entry(frame, textvariable=var, width=10).grid(row=i, column=2, sticky="w", padx=5, pady=2)

    def create_advanced_settings_frame(self, parent):
        frame = tk.LabelFrame(parent, text="Advanced Settings")
        frame.pack(fill="x", padx=5, pady=5)

        settings = [
            ("Similarity Threshold:", self.similarity_threshold_var),
            ("Min Entity Occurrence:", self.min_entity_occurrence_var),
            ("Top K:", self.top_k_var),
            ("Entity Relevance Threshold:", self.entity_relevance_threshold_var),
        ]

        for i, (label, var) in enumerate(settings):
            tk.Label(frame, text=label).grid(row=i, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(frame, textvariable=var, width=20).grid(row=i, column=1, sticky="w", padx=5, pady=2)

        tk.Checkbutton(frame, text="Enable Family Extraction", variable=self.enable_family_extraction_var).grid(row=len(settings), column=0, columnspan=2, sticky="w", padx=5, pady=2)
        tk.Checkbutton(frame, text="Enable Semantic Edges", variable=self.enable_semantic_edges_var).grid(row=len(settings)+1, column=0, columnspan=2, sticky="w", padx=5, pady=2)

    def create_file_path_settings_frame(self, parent):
        frame = tk.LabelFrame(parent, text="File Path Settings")
        frame.pack(fill="x", padx=5, pady=5)

        paths = [
            ("Database File:", self.db_file_path_var),
            ("Embeddings File:", self.embeddings_file_path_var),
            ("Knowledge Graph File:", self.knowledge_graph_file_path_var)
        ]

        for i, (label, var) in enumerate(paths):
            tk.Label(frame, text=label).grid(row=i, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(frame, textvariable=var, width=20).grid(row=i, column=1, sticky="w", padx=5, pady=2)
            tk.Button(frame, text="Browse", command=lambda v=var: self.browse_file_path(v)).grid(row=i, column=2, padx=5, pady=2)

        tk.Label(frame, text="Embedding Model:").grid(row=len(paths), column=0, sticky="e", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.model_name_var, width=20).grid(row=len(paths), column=1, sticky="w", padx=5, pady=2)

        tk.Label(frame, text="NLP Model:").grid(row=len(paths)+1, column=0, sticky="e", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.nlp_model_var, width=20).grid(row=len(paths)+1, column=1, sticky="w", padx=5, pady=2)

    def browse_file_path(self, var):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("All Files", "*.*")])
        if file_path:
            var.set(file_path)

    def use_default_settings(self):
        # Reset all variables to their default values
        self.chunk_size_var.set(500)
        self.overlap_size_var.set(200)
        self.batch_size_var.set(32)
        self.results_file_path_var.set("results.txt")
        self.max_history_length_var.set(5)
        self.conversation_context_size_var.set(3)
        self.update_threshold_var.set(10)
        self.ollama_model_var.set("phi3:instruct")
        self.temperature_var.set(0.1)
        self.similarity_threshold_var.set(0.7)
        self.enable_family_extraction_var.set(True)
        self.min_entity_occurrence_var.set(1)
        self.model_name_var.set("all-MiniLM-L6-v2")
        self.nlp_model_var.set("en_core_web_sm")
        self.db_file_path_var.set("db.txt")
        self.embeddings_file_path_var.set("db_embeddings.pt")
        self.knowledge_graph_file_path_var.set("knowledge_graph.json")
        self.enable_semantic_edges_var.set(True)
        self.top_k_var.set(5)
        self.entity_relevance_threshold_var.set(0.5)
        self.lexical_weight_var.set(1.0)
        self.semantic_weight_var.set(1.0)
        self.graph_weight_var.set(1.0)
        self.text_weight_var.set(1.0)
        self.enable_lexical_search_var.set(True)
        self.enable_semantic_search_var.set(True)
        self.enable_graph_search_var.set(True)
        self.enable_text_search_var.set(True)
        self.num_questions_var.set(8)
        
        messagebox.showinfo("Default Settings", "Default settings have been restored.")

    def save_settings_template(self):
        settings = self.get_current_settings()
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Save Successful", f"Settings template saved to {file_path}")

    def load_settings_template(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                
                self.apply_loaded_settings(settings)
                messagebox.showinfo("Load Successful", f"Settings loaded from {file_path}")
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Invalid JSON file. Could not load settings.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while loading settings: {str(e)}")

    def apply_settings(self):
        try:
            # Validate settings
            if self.chunk_size_var.get() <= 0 or self.overlap_size_var.get() < 0 or self.overlap_size_var.get() >= self.chunk_size_var.get():
                raise ValueError("Invalid chunk size or overlap size")
            if self.batch_size_var.get() <= 0:
                raise ValueError("Batch size must be positive")
            if not 0 <= self.similarity_threshold_var.get() <= 1:
                raise ValueError("Similarity threshold must be between 0 and 1")
            if self.min_entity_occurrence_var.get() <= 0:
                raise ValueError("Minimum entity occurrence must be positive")
            if self.max_history_length_var.get() <= 0:
                raise ValueError("Max history length must be positive")
            if self.conversation_context_size_var.get() <= 0:
                raise ValueError("Conversation context size must be positive")
            if self.update_threshold_var.get() <= 0:
                raise ValueError("Update threshold must be positive")
            if self.temperature_var.get() < 0 or self.temperature_var.get() > 1:
                raise ValueError("Temperature must be between 0 and 1")
            if self.top_k_var.get() <= 0:
                raise ValueError("Top K must be positive")
            if not 0 <= self.entity_relevance_threshold_var.get() <= 1:
                raise ValueError("Entity relevance threshold must be between 0 and 1")
            if self.num_questions_var.get() <= 0:
                raise ValueError("Number of questions must be positive")

            # Apply settings
            set_chunk_sizes(self.chunk_size_var.get(), self.overlap_size_var.get())
            set_batch_size(self.batch_size_var.get())
            set_graph_settings(self.similarity_threshold_var.get(), self.enable_family_extraction_var.get(), self.min_entity_occurrence_var.get())
            set_model_name(self.model_name_var.get())
            set_nlp_model(self.nlp_model_var.get())
            set_db_file(self.db_file_path_var.get())
            set_embeddings_file(self.embeddings_file_path_var.get())
            set_knowledge_graph_file(self.knowledge_graph_file_path_var.get())
            set_max_history_length(self.max_history_length_var.get())
            set_conversation_context_size(self.conversation_context_size_var.get())
            set_update_threshold(self.update_threshold_var.get())
            set_ollama_model(self.ollama_model_var.get())
            set_temperature(self.temperature_var.get())
            set_top_k(self.top_k_var.get())
            set_entity_relevance_threshold(self.entity_relevance_threshold_var.get())
            set_search_weights(self.lexical_weight_var.get(), self.semantic_weight_var.get(), 
                               self.graph_weight_var.get(), self.text_weight_var.get())
            set_search_toggles(self.enable_lexical_search_var.get(), self.enable_semantic_search_var.get(),
                               self.enable_graph_search_var.get(), self.enable_text_search_var.get())

            # Save results file path
            with open("config.json", "w") as f:
                json.dump({"results_file_path": self.results_file_path_var.get()}, f)

            self.save_current_config()  # Save the current configuration
            messagebox.showinfo("Settings Applied", "All settings have been successfully applied and saved.")
        except ValueError as e:
            messagebox.showerror("Invalid Settings", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while applying settings: {str(e)}")

    def get_current_settings(self):
        return {
            "chunk_size": self.chunk_size_var.get(),
            "overlap_size": self.overlap_size_var.get(),
            "batch_size": self.batch_size_var.get(),
            "results_file_path": self.results_file_path_var.get(),
            "max_history_length": self.max_history_length_var.get(),
            "conversation_context_size": self.conversation_context_size_var.get(),
            "update_threshold": self.update_threshold_var.get(),
            "ollama_model": self.ollama_model_var.get(),
            "temperature": self.temperature_var.get(),
            "similarity_threshold": self.similarity_threshold_var.get(),
            "enable_family_extraction": self.enable_family_extraction_var.get(),
            "min_entity_occurrence": self.min_entity_occurrence_var.get(),
            "model_name": self.model_name_var.get(),
            "nlp_model": self.nlp_model_var.get(),
            "db_file_path": self.db_file_path_var.get(),
            "embeddings_file_path": self.embeddings_file_path_var.get(),
            "knowledge_graph_file_path": self.knowledge_graph_file_path_var.get(),
            "enable_semantic_edges": self.enable_semantic_edges_var.get(),
            "top_k": self.top_k_var.get(),
            "entity_relevance_threshold": self.entity_relevance_threshold_var.get(),
            "lexical_weight": self.lexical_weight_var.get(),
            "semantic_weight": self.semantic_weight_var.get(),
            "graph_weight": self.graph_weight_var.get(),
            "text_weight": self.text_weight_var.get(),
            "enable_lexical_search": self.enable_lexical_search_var.get(),
            "enable_semantic_search": self.enable_semantic_search_var.get(),
            "enable_graph_search": self.enable_graph_search_var.get(),
            "enable_text_search": self.enable_text_search_var.get(),
            "num_questions": self.num_questions_var.get(),
        }

    def save_current_config(self):
        settings = self.get_current_settings()
        with open(self.config_file, 'w') as f:
            json.dump(settings, f, indent=4)

    def load_current_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    settings = json.load(f)
                
                self.apply_loaded_settings(settings)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while loading settings: {str(e)}")

    def apply_loaded_settings(self, settings):
        self.chunk_size_var.set(settings.get("chunk_size", 500))
        self.overlap_size_var.set(settings.get("overlap_size", 200))
        self.batch_size_var.set(settings.get("batch_size", 32))
        self.results_file_path_var.set(settings.get("results_file_path", "results.txt"))
        self.max_history_length_var.set(settings.get("max_history_length", 5))
        self.conversation_context_size_var.set(settings.get("conversation_context_size", 3))
        self.update_threshold_var.set(settings.get("update_threshold", 10))
        self.ollama_model_var.set(settings.get("ollama_model", "phi3:instruct"))
        self.temperature_var.set(settings.get("temperature", 0.1))
        self.similarity_threshold_var.set(settings.get("similarity_threshold", 0.7))
        self.enable_family_extraction_var.set(settings.get("enable_family_extraction", True))
        self.min_entity_occurrence_var.set(settings.get("min_entity_occurrence", 1))
        self.model_name_var.set(settings.get("model_name", "all-MiniLM-L6-v2"))
        self.nlp_model_var.set(settings.get("nlp_model", "en_core_web_sm"))
        self.db_file_path_var.set(settings.get("db_file_path", "db.txt"))
        self.embeddings_file_path_var.set(settings.get("embeddings_file_path", "db_embeddings.pt"))
        self.knowledge_graph_file_path_var.set(settings.get("knowledge_graph_file_path", "knowledge_graph.json"))
        self.enable_semantic_edges_var.set(settings.get("enable_semantic_edges", True))
        self.top_k_var.set(settings.get("top_k", 5))
        self.entity_relevance_threshold_var.set(settings.get("entity_relevance_threshold", 0.5))
        self.lexical_weight_var.set(settings.get("lexical_weight", 1.0))
        self.semantic_weight_var.set(settings.get("semantic_weight", 1.0))
        self.graph_weight_var.set(settings.get("graph_weight", 1.0))
        self.text_weight_var.set(settings.get("text_weight", 1.0))
        self.enable_lexical_search_var.set(settings.get("enable_lexical_search", True))
        self.enable_semantic_search_var.set(settings.get("enable_semantic_search", True))
        self.enable_graph_search_var.set(settings.get("enable_graph_search", True))
        self.enable_text_search_var.set(settings.get("enable_text_search", True))
        self.num_questions_var.set(settings.get("num_questions", 8))

    def get_settings_tab(self):
        return self.settings_tab

# You might want to add a main function if you want to test the SettingsManager independently
def main():
    root = tk.Tk()
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")
    
    settings_manager = SettingsManager(notebook)
    notebook.add(settings_manager.get_settings_tab(), text="Settings")
    
    root.mainloop()

if __name__ == "__main__":
    main()
