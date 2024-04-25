import tkinter as tk
from tkinter import filedialog
from upload_x.upload_pdf import upload_pdf
from upload_x.upload_txt import upload_txt
from upload_x.upload_json import upload_json
from upload_x.upload_csv import upload_csv
from upload_x.upload_docx import upload_docx
from upload_x.chunking import handle_text_chunking
from api_connectivity import configure_api
import os

def upload_and_chunk(file_type):
    if file_type == "PDF":
        text = upload_pdf()
    elif file_type == "Text":
        text = upload_txt()
    elif file_type == "JSON":
        text = upload_json()
    elif file_type == "CSV":
        text = upload_csv()
    elif file_type == "DOCX":
        text = upload_docx()
    if text:
        chunks = handle_text_chunking(text)
        with open("db.txt", "a", encoding="utf-8") as db_file:
            for chunk in chunks:
                db_file.write(chunk.strip() + "\n\n")  # Two newlines to separate chunks
        print(f"{file_type} file content appended to db.txt with overlapping chunks.")
    else:
        print("No file selected.")

def run_localragX(api_type):
    client = configure_api(api_type)
    os.system(f"python localrag13.py {api_type}")

def set_api(api_type):
    run_localragX(api_type)

def execute_embeddings():
    os.system("python embeddings_utils.py")

def main():
    root = tk.Tk()
    root.title("E-RAG")

    # Frame for upload buttons with a different color
    upload_frame = tk.Frame(root)
    upload_frame.pack(fill="x")

    # Title for the row of upload buttons
    upload_title = tk.Label(upload_frame, text="Upload")
    upload_title.pack(side="left", padx=5, pady=10)

    # Frame for upload buttons
    upload_buttons_frame = tk.Frame(upload_frame)
    upload_buttons_frame.pack(side="left", padx=5, pady=10)

    def create_upload_and_chunk_func(file_type):
        return lambda: upload_and_chunk(file_type)

    file_types = ["PDF", "Text", "JSON", "CSV", "DOCX"]
    for file_type in file_types:
        button = tk.Button(upload_buttons_frame, text=f"Upload {file_type}", command=create_upload_and_chunk_func(file_type))
        button.pack(side="left", padx=5)

    # Frame for embeddings button
    embeddings_frame = tk.Frame(root)
    embeddings_frame.pack(fill="x")

    # Title for the embeddings row
    embeddings_title = tk.Label(embeddings_frame, text="Embeddings")
    embeddings_title.pack(side="left", padx=5, pady=10)

    # Button to execute embeddings
    execute_embeddings_button = tk.Button(embeddings_frame, text="Execute Embeddings", command=execute_embeddings)
    execute_embeddings_button.pack(side="left", padx=5)

    # Frame for model options
    model_frame = tk.Frame(root)
    model_frame.pack(fill="x")

    # Title for the model row
    model_title = tk.Label(model_frame, text="Model")
    model_title.pack(side="left", padx=5, pady=10)

    # Option menu to choose between ollama and llama
    api_type_var = tk.StringVar(root)
    api_type_var.set("ollama")  # Default value
    api_options = ["ollama", "llama"]
    api_menu = tk.OptionMenu(model_frame, api_type_var, *api_options, command=set_api)
    api_menu.pack(side="left", padx=5)

    # Button to run localragX.py
    run_model_button = tk.Button(model_frame, text="Run Model", command=lambda: run_localragX(api_type_var.get()))
    run_model_button.pack(side="left", padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()
