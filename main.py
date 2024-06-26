import tkinter as tk
from tkinter import messagebox, ttk
import threading
import os
from file_processing import process_file, append_to_db
from run_model import RAGSystem
from embeddings_utils import compute_and_save_embeddings, load_or_compute_embeddings
from sentence_transformers import SentenceTransformer
from create_graph import create_knowledge_graph
from settings import SettingsManager

class ERAGGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("E-RAG")
        self.api_type_var = tk.StringVar(master)
        self.api_type_var.set("ollama")  # Default value
        self.rag_system = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main")

        self.settings_manager = SettingsManager(self.notebook)
        self.settings_tab = self.settings_manager.get_settings_tab()
        self.notebook.add(self.settings_tab, text="Settings")

        self.create_main_tab()

    def create_main_tab(self):
        self.create_upload_frame()
        self.create_embeddings_frame()
        self.create_model_frame()

    def create_upload_frame(self):
        upload_frame = tk.LabelFrame(self.main_tab, text="Upload")
        upload_frame.pack(fill="x", padx=10, pady=5)

        file_types = ["DOCX", "JSON", "PDF", "Text"]
        for file_type in file_types:
            button = tk.Button(upload_frame, text=f"Upload {file_type}", 
                               command=lambda ft=file_type: self.upload_and_chunk(ft))
            button.pack(side="left", padx=5, pady=5)

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
        if not os.path.exists(self.settings_manager.db_file_path_var.get()):
            messagebox.showerror("Error", f"{self.settings_manager.db_file_path_var.get()} not found. Please upload some documents first.")
            return

        # Process db.txt
        embeddings, _, _ = load_or_compute_embeddings(
            self.model, 
            self.settings_manager.db_file_path_var.get(), 
            self.settings_manager.embeddings_file_path_var.get()
        )
        messagebox.showinfo("Success", f"Embeddings computed and saved successfully. Shape: {embeddings.shape}")

      except Exception as e:
        messagebox.showerror("Error", f"An error occurred while computing embeddings: {str(e)}")

    def create_knowledge_graph(self):
      try:
        if not os.path.exists(self.settings_manager.db_file_path_var.get()) or not os.path.exists(self.settings_manager.embeddings_file_path_var.get()):
            messagebox.showerror("Error", f"{self.settings_manager.db_file_path_var.get()} or {self.settings_manager.embeddings_file_path_var.get()} not found. Please upload documents and execute embeddings first.")
            return

        G = create_knowledge_graph()  # Call the imported function
        messagebox.showinfo("Success", f"Knowledge graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges, and saved as {self.settings_manager.knowledge_graph_file_path_var.get()}.")
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
