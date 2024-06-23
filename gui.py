import tkinter as tk
from tkinter import messagebox
import threading
import os
from file_processing import process_file
from localrag14 import RAGSystem
from embeddings_utils import compute_and_save_embeddings, load_or_compute_embeddings
from sentence_transformers import SentenceTransformer

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
        self.create_upload_frame()
        self.create_embeddings_frame()
        self.create_model_frame()

    def create_upload_frame(self):
        upload_frame = tk.LabelFrame(self.master, text="Upload")
        upload_frame.pack(fill="x", padx=10, pady=5)

        file_types = ["DOCX", "JSON", "PDF", "Text"]
        for file_type in file_types:
            button = tk.Button(upload_frame, text=f"Upload {file_type}", 
                               command=lambda ft=file_type: self.upload_and_chunk(ft))
            button.pack(side="left", padx=5, pady=5)

    def create_embeddings_frame(self):
        embeddings_frame = tk.LabelFrame(self.master, text="Embeddings")
        embeddings_frame.pack(fill="x", padx=10, pady=5)

        execute_embeddings_button = tk.Button(embeddings_frame, text="Execute Embeddings", 
                                              command=self.execute_embeddings)
        execute_embeddings_button.pack(side="left", padx=5, pady=5)

    def create_model_frame(self):
        model_frame = tk.LabelFrame(self.master, text="Model")
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
                with open("db.txt", "a", encoding="utf-8") as db_file:
                    for chunk in chunks:
                        db_file.write(chunk.strip() + "\n\n")  # Two newlines to separate chunks
                messagebox.showinfo("Success", f"{file_type} file content appended to db.txt with overlapping chunks.")
            else:
                messagebox.showwarning("Warning", "No file selected or file was empty.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the file: {str(e)}")

    def execute_embeddings(self):
        try:
            if not os.path.exists("db.txt"):
                messagebox.showerror("Error", "db.txt not found. Please upload some documents first.")
                return

            with open("db.txt", "r", encoding="utf-8") as db_file:
                db_content = db_file.readlines()

            compute_and_save_embeddings(db_content, self.model, "db_embeddings.pt")
            messagebox.showinfo("Success", "Embeddings computed and saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while computing embeddings: {str(e)}")

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
