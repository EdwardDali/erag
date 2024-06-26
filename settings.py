import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import json
from file_processing import set_chunk_sizes
from embeddings_utils import set_batch_size
from create_graph import set_graph_settings

class SettingsManager:
    def __init__(self, notebook):
        self.notebook = notebook
        self.settings_tab = ttk.Frame(self.notebook)

        # Settings variables
        self.chunk_size_var = tk.IntVar(value=500)
        self.overlap_size_var = tk.IntVar(value=200)
        self.batch_size_var = tk.IntVar(value=32)
        self.similarity_threshold_var = tk.DoubleVar(value=0.7)
        self.enable_family_extraction_var = tk.BooleanVar(value=True)
        self.min_entity_occurrence_var = tk.IntVar(value=1)

        self.create_settings_tab()

    def create_settings_tab(self):
        self.create_settings_management_frame()
        self.create_chunk_settings_frame()
        self.create_embedding_settings_frame()
        self.create_graph_settings_frame()

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
        self.chunk_size_var.set(500)
        self.overlap_size_var.set(200)
        self.batch_size_var.set(32)
        self.similarity_threshold_var.set(0.7)
        self.enable_family_extraction_var.set(True)
        self.min_entity_occurrence_var.set(1)
        messagebox.showinfo("Default Settings", "Default settings have been restored.")

    def save_settings_template(self):
        settings = {
            "chunk_size": self.chunk_size_var.get(),
            "overlap_size": self.overlap_size_var.get(),
            "batch_size": self.batch_size_var.get(),
            "similarity_threshold": self.similarity_threshold_var.get(),
            "enable_family_extraction": self.enable_family_extraction_var.get(),
            "min_entity_occurrence": self.min_entity_occurrence_var.get()
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
                self.chunk_size_var.set(settings.get("chunk_size", 500))
                self.overlap_size_var.set(settings.get("overlap_size", 200))
                self.batch_size_var.set(settings.get("batch_size", 32))
                self.similarity_threshold_var.set(settings.get("similarity_threshold", 0.7))
                self.enable_family_extraction_var.set(settings.get("enable_family_extraction", True))
                self.min_entity_occurrence_var.set(settings.get("min_entity_occurrence", 1))
                messagebox.showinfo("Load Successful", f"Settings loaded from {file_path}")
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Invalid JSON file. Could not load settings.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while loading settings: {str(e)}")

    def apply_settings(self):
        try:
            chunk_size = self.chunk_size_var.get()
            overlap_size = self.overlap_size_var.get()
            batch_size = self.batch_size_var.get()
            similarity_threshold = self.similarity_threshold_var.get()
            enable_family_extraction = self.enable_family_extraction_var.get()
            min_entity_occurrence = self.min_entity_occurrence_var.get()

            if chunk_size <= 0 or overlap_size <= 0 or overlap_size >= chunk_size:
                raise ValueError("Invalid chunk size or overlap size")
            if batch_size <= 0:
                raise ValueError("Batch size must be positive")
            if not 0 <= similarity_threshold <= 1:
                raise ValueError("Similarity threshold must be between 0 and 1")
            if min_entity_occurrence <= 0:
                raise ValueError("Minimum entity occurrence must be positive")

            set_chunk_sizes(chunk_size, overlap_size)
            set_batch_size(batch_size)
            set_graph_settings(similarity_threshold, enable_family_extraction, min_entity_occurrence)

            messagebox.showinfo("Settings Applied", "All settings have been successfully applied.")
        except ValueError as e:
            messagebox.showerror("Invalid Settings", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while applying settings: {str(e)}")

    def get_settings_tab(self):
        return self.settings_tab
