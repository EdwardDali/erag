import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import json
from file_processing import set_chunk_sizes
from embeddings_utils import set_batch_size, set_model_name, set_db_file, set_embeddings_file
from create_graph import set_graph_settings, set_nlp_model, set_knowledge_graph_file, set_embeddings_file

class SettingsManager:
    def __init__(self, notebook):
        self.notebook = notebook
        self.settings_tab = ttk.Frame(self.notebook)

        # Existing settings variables
        self.chunk_size_var = tk.IntVar(value=500)
        self.overlap_size_var = tk.IntVar(value=200)
        self.batch_size_var = tk.IntVar(value=32)
        self.similarity_threshold_var = tk.DoubleVar(value=0.7)
        self.enable_family_extraction_var = tk.BooleanVar(value=True)
        self.min_entity_occurrence_var = tk.IntVar(value=1)

        # New settings variables
        self.model_name_var = tk.StringVar(value="all-MiniLM-L6-v2")
        self.nlp_model_var = tk.StringVar(value="en_core_web_sm")
        self.db_file_path_var = tk.StringVar(value="db.txt")
        self.embeddings_file_path_var = tk.StringVar(value="db_embeddings.pt")
        self.knowledge_graph_file_path_var = tk.StringVar(value="knowledge_graph.json")

        self.create_settings_tab()

    def create_settings_tab(self):
        self.create_settings_management_frame()
        self.create_chunk_settings_frame()
        self.create_embedding_settings_frame()
        self.create_graph_settings_frame()
        self.create_model_settings_frame()
        self.create_file_path_settings_frame()

        apply_button = tk.Button(self.settings_tab, text="Apply Settings", command=self.apply_settings)
        apply_button.pack(pady=10)


    def create_model_settings_frame(self):
        model_frame = tk.LabelFrame(self.settings_tab, text="Model Settings")
        model_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(model_frame, text="Embedding Model:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        model_name_entry = ttk.Entry(model_frame, textvariable=self.model_name_var, width=30)
        model_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        tk.Label(model_frame, text="NLP Model:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        nlp_model_entry = ttk.Entry(model_frame, textvariable=self.nlp_model_var, width=30)
        nlp_model_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    def create_file_path_settings_frame(self):
        file_path_frame = tk.LabelFrame(self.settings_tab, text="File Path Settings")
        file_path_frame.pack(fill="x", padx=10, pady=5)

        paths = [
            ("Database File:", self.db_file_path_var),
            ("Embeddings File:", self.embeddings_file_path_var),
            ("Knowledge Graph File:", self.knowledge_graph_file_path_var)
        ]

        for i, (label, var) in enumerate(paths):
            tk.Label(file_path_frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry = ttk.Entry(file_path_frame, textvariable=var, width=30)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            browse_button = tk.Button(file_path_frame, text="Browse", command=lambda v=var: self.browse_file_path(v))
            browse_button.grid(row=i, column=2, padx=5, pady=5)

    def browse_file_path(self, var):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("All Files", "*.*")])
        if file_path:
            var.set(file_path)


    def create_settings_management_frame(self):
        management_frame = tk.LabelFrame(self.settings_tab, text="Settings Management")
        management_frame.pack(fill="x", padx=10, pady=5)

        use_default_button = tk.Button(management_frame, text="Use Default Settings", command=self.use_default_settings)
        use_default_button.pack(side="left", padx=5, pady=5)

        save_template_button = tk.Button(management_frame, text="Save Settings as Template", command=self.save_settings_template)
        save_template_button.pack(side="left", padx=5, pady=5)

        load_template_button = tk.Button(management_frame, text="Load Settings Template", command=self.load_settings_template)
        load_template_button.pack(side="left", padx=5, pady=5)

    def create_chunk_settings_frame(self):
        chunk_frame = tk.LabelFrame(self.settings_tab, text="Chunk Settings")
        chunk_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(chunk_frame, text="Chunk Size:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        chunk_size_entry = ttk.Entry(chunk_frame, textvariable=self.chunk_size_var, width=10)
        chunk_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        tk.Label(chunk_frame, text="Overlap Size:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        overlap_size_entry = ttk.Entry(chunk_frame, textvariable=self.overlap_size_var, width=10)
        overlap_size_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    def create_embedding_settings_frame(self):
        embedding_frame = tk.LabelFrame(self.settings_tab, text="Embedding Settings")
        embedding_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(embedding_frame, text="Batch Size:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        batch_size_entry = ttk.Entry(embedding_frame, textvariable=self.batch_size_var, width=10)
        batch_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    def create_graph_settings_frame(self):
        graph_frame = tk.LabelFrame(self.settings_tab, text="Graph Settings")
        graph_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(graph_frame, text="Similarity Threshold:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        similarity_threshold_entry = ttk.Entry(graph_frame, textvariable=self.similarity_threshold_var, width=10)
        similarity_threshold_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        enable_family_extraction_check = ttk.Checkbutton(graph_frame, text="Enable Family Relation Extraction", 
                                                         variable=self.enable_family_extraction_var)
        enable_family_extraction_check.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        tk.Label(graph_frame, text="Min Entity Occurrence:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        min_entity_occurrence_entry = ttk.Entry(graph_frame, textvariable=self.min_entity_occurrence_var, width=10)
        min_entity_occurrence_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    def use_default_settings(self):
        # Existing default settings
        self.chunk_size_var.set(500)
        self.overlap_size_var.set(200)
        self.batch_size_var.set(32)
        self.similarity_threshold_var.set(0.7)
        self.enable_family_extraction_var.set(True)
        self.min_entity_occurrence_var.set(1)
        
        # New default settings
        self.model_name_var.set("all-MiniLM-L6-v2")
        self.nlp_model_var.set("en_core_web_sm")
        self.db_file_path_var.set("db.txt")
        self.embeddings_file_path_var.set("db_embeddings.pt")
        self.knowledge_graph_file_path_var.set("knowledge_graph.json")
        
        messagebox.showinfo("Default Settings", "Default settings have been restored.")

    def save_settings_template(self):
        settings = {
            # Existing settings
            "chunk_size": self.chunk_size_var.get(),
            "overlap_size": self.overlap_size_var.get(),
            "batch_size": self.batch_size_var.get(),
            "similarity_threshold": self.similarity_threshold_var.get(),
            "enable_family_extraction": self.enable_family_extraction_var.get(),
            "min_entity_occurrence": self.min_entity_occurrence_var.get(),
            # New settings
            "model_name": self.model_name_var.get(),
            "nlp_model": self.nlp_model_var.get(),
            "db_file_path": self.db_file_path_var.get(),
            "embeddings_file_path": self.embeddings_file_path_var.get(),
            "knowledge_graph_file_path": self.knowledge_graph_file_path_var.get()
        }
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
                # Load existing settings
                self.chunk_size_var.set(settings.get("chunk_size", 500))
                self.overlap_size_var.set(settings.get("overlap_size", 200))
                self.batch_size_var.set(settings.get("batch_size", 32))
                self.similarity_threshold_var.set(settings.get("similarity_threshold", 0.7))
                self.enable_family_extraction_var.set(settings.get("enable_family_extraction", True))
                self.min_entity_occurrence_var.set(settings.get("min_entity_occurrence", 1))
                # Load new settings
                self.model_name_var.set(settings.get("model_name", "all-MiniLM-L6-v2"))
                self.nlp_model_var.set(settings.get("nlp_model", "en_core_web_sm"))
                self.db_file_path_var.set(settings.get("db_file_path", "db.txt"))
                self.embeddings_file_path_var.set(settings.get("embeddings_file_path", "db_embeddings.pt"))
                self.knowledge_graph_file_path_var.set(settings.get("knowledge_graph_file_path", "knowledge_graph.json"))
                messagebox.showinfo("Load Successful", f"Settings loaded from {file_path}")
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Invalid JSON file. Could not load settings.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while loading settings: {str(e)}")

    def apply_settings(self):
      try:
        # Existing settings
        chunk_size = self.chunk_size_var.get()
        overlap_size = self.overlap_size_var.get()
        batch_size = self.batch_size_var.get()
        similarity_threshold = self.similarity_threshold_var.get()
        enable_family_extraction = self.enable_family_extraction_var.get()
        min_entity_occurrence = self.min_entity_occurrence_var.get()

        # New settings
        model_name = self.model_name_var.get()
        nlp_model = self.nlp_model_var.get()
        db_file_path = self.db_file_path_var.get()
        embeddings_file_path = self.embeddings_file_path_var.get()
        knowledge_graph_file_path = self.knowledge_graph_file_path_var.get()
        

        # Validate settings
        if chunk_size <= 0 or overlap_size <= 0 or overlap_size >= chunk_size:
            raise ValueError("Invalid chunk size or overlap size")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        if min_entity_occurrence <= 0:
            raise ValueError("Minimum entity occurrence must be positive")
        if not model_name or not nlp_model:
            raise ValueError("Model name and NLP model cannot be empty")
        if not db_file_path or not embeddings_file_path or not knowledge_graph_file_path:
            raise ValueError("File paths cannot be empty")

        # Apply settings
        set_chunk_sizes(chunk_size, overlap_size)
        set_batch_size(batch_size)
        set_graph_settings(similarity_threshold, enable_family_extraction, min_entity_occurrence)
        set_model_name(model_name)
        set_nlp_model(nlp_model)
        set_embeddings_file(embeddings_file_path)

        # New calls to update file paths
        set_db_file(db_file_path)
        set_embeddings_file(embeddings_file_path)
        set_knowledge_graph_file(knowledge_graph_file_path)

        messagebox.showinfo("Settings Applied", "All settings have been successfully applied.")
      except ValueError as e:
        messagebox.showerror("Invalid Settings", str(e))
      except Exception as e:
        messagebox.showerror("Error", f"An error occurred while applying settings: {str(e)}")

    def get_settings_tab(self):
     return self.settings_tab

