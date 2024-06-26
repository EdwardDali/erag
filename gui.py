import tkinter as tk
from tkinter import messagebox, ttk
import threading
import os
from file_processing import process_file, append_to_db, set_chunk_sizes
from run_model import RAGSystem
from embeddings_utils import compute_and_save_embeddings, load_or_compute_embeddings, set_batch_size
from sentence_transformers import SentenceTransformer
from create_graph import create_knowledge_graph, set_graph_settings

class ERAGGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("E-RAG")
        self.api_type_var = tk.StringVar(master)
        self.api_type_var.set("ollama")  # Default value
        self.rag_system = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Settings variables
        self.chunk_size_var = tk.IntVar(value=500)
        self.overlap_size_var = tk.IntVar(value=200)
        self.batch_size_var = tk.IntVar(value=32)
        self.similarity_threshold_var = tk.DoubleVar(value=0.7)
        self.enable_family_extraction_var = tk.BooleanVar(value=True)
        self.min_entity_occurrence_var = tk.IntVar(value=1)

        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.main_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.main_tab, text="Main")
        self.notebook.add(self.settings_tab, text="Settings")

        self.create_main_tab()
        self.create_settings_tab()

    def create_main_tab(self):
        self.create_upload_frame()
        self.create_embeddings_frame()
        self.create_model_frame()

    def create_settings_tab(self):
        self.create_chunk_settings_frame()
        self.create_embedding_settings_frame()
        self.create_graph_settings_frame()

    def create_upload_frame(self):
        upload_frame = tk.LabelFrame(self.main_tab, text="Upload")
        upload_frame.pack(fill="x", padx=10, pady=5)

        file_types = ["DOCX", "JSON", "PDF", "Text"]
        for file_type in file_types:
            button = tk.Button(upload_frame, text=f"Upload {file_type}", 
                               command=lambda ft=file_type: self.upload_and_chunk(ft))
            button.pack(side="left", padx=5, pady=5)

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

        apply_button = tk.Button(self.settings_tab, text="Apply Settings", command=self.apply_settings)
        apply_button.pack(pady=10)

    def create_embeddings_frame(self):
        embeddings_frame = tk.LabelFrame(self.main_tab, text="Embeddings")
        embeddings_frame.pack(fill="x", padx=10, pady=5)

        execute_embeddings_button = tk.Button(embeddings_frame, text="Execute Embeddings", 
                                              command=self.execute_embeddings)
        execute_embeddings_button.pack(side="left", padx=5, pady=5)

        create_knowledge_graph_button = tk.Button(embeddings_frame, text="Create Knowledge Graph", 
                                                  command=self.create_knowledge_graph)
        create_knowledge_graph_button.pack(side="left", padx=5, pady=5)

    def create_model_frame(self):
        model_frame = tk.LabelFrame(self.main_tab, text="Model")
        model_frame.pack(fill="x", padx=10, pady=5)

        api_options = ["ollama", "llama"]
        api_menu = tk.OptionMenu(model_frame, self.api_type_var, *api_options)
        api_menu.pack(side="left", padx=5, pady=5)

        run_model_button = tk.Button(model_frame, text="Run Model", command=self.run_model)
        run_model_button.pack(side="left", padx=5, pady=5)

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

    def upload_and_chunk(self, file_type: str):
        try:
            chunks = process_file(file_type)
            if chunks:
                append_to_db(chunks)
                messagebox.showinfo("Success", f"{file_type} file content processed and appended to db.txt with overlapping chunks.")
            else:
                messagebox.showwarning("Warning", "No file selected or file was empty.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the file: {str(e)}")

    def execute_embeddings(self):
        try:
            if not os.path.exists("db.txt"):
                messagebox.showerror("Error", "db.txt not found. Please upload some documents first.")
                return

            # Process db.txt
            embeddings, _, _ = load_or_compute_embeddings(self.model, "db.txt", "db_embeddings.pt")
            messagebox.showinfo("Success", f"Embeddings for db.txt computed and saved successfully. Shape: {embeddings.shape}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while computing embeddings: {str(e)}")

    def create_knowledge_graph(self):
        try:
            if not os.path.exists("db.txt") or not os.path.exists("db_embeddings.pt"):
                messagebox.showerror("Error", "db.txt or db_embeddings.pt not found. Please upload documents and execute embeddings first.")
                return

            G = create_knowledge_graph()  # Call the imported function
            messagebox.showinfo("Success", f"Knowledge graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges, and saved as knowledge_graph.json.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while creating the knowledge graph: {str(e)}")

    def run_model(self):
        try:
            api_type = self.api_type_var.get()
            self.rag_system = RAGSystem(api_type)
            
            # Run the CLI in a separate thread to keep the GUI responsive
            threading.Thread(target=self.rag_system.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"RAG system started with {api_type} API. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the RAG system: {str(e)}")

def main():
    root = tk.Tk()
    ERAGGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
