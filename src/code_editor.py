import os
import zipfile
import tempfile
import webbrowser
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import messagebox
import datetime
from src.api_model import EragAPI
from src.look_and_feel import error, success, warning, info
from src.settings import settings

class CodeEditor:
    def __init__(self, erag_api: EragAPI):
        self.api = erag_api
        self.files = {}
        self.output_folder = None
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Toplevel()
        self.root.title("AI-powered Application Generator")
        self.root.geometry("1200x800")

        # User request input
        request_frame = ttk.Frame(self.root)
        request_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(request_frame, text="Enter your application request:").pack(side=tk.LEFT)
        self.request_entry = ttk.Entry(request_frame, width=50)
        self.request_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(request_frame, text="Generate", command=self.generate_application).pack(side=tk.LEFT)

        # Main content area
        content_frame = ttk.Frame(self.root)
        content_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        # File listing
        self.file_list = ttk.Treeview(content_frame, columns=("name",), show="tree")
        self.file_list.heading("#0", text="Files")
        self.file_list.column("#0", width=200)
        self.file_list.grid(row=0, column=0, sticky="nsew")
        self.file_list.bind("<<TreeviewSelect>>", self.on_file_select)

        # Code display
        self.code_display = scrolledtext.ScrolledText(content_frame, wrap=tk.NONE)
        self.code_display.grid(row=0, column=1, sticky="nsew")

        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(button_frame, text="Download ZIP", command=self.download_zip).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Preview App", command=self.preview_app).pack(side=tk.LEFT)

        # Loading indicator
        self.loading_label = ttk.Label(self.root, text="Generating application...", foreground="blue")
        self.loading_label.pack(side=tk.BOTTOM)
        self.loading_label.pack_forget()

    def save_file(self, filename: str, content: str, stage: str):
        if self.output_folder is None:
            # Create a new subfolder for this application generation process
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = os.path.join(settings.output_folder, f"generated_app_{timestamp}")
            os.makedirs(self.output_folder, exist_ok=True)
            print(info(f"Created output folder: {self.output_folder}"))

        # Sanitize filename to remove any invalid characters
        safe_filename = "".join([c for c in filename if c.isalnum() or c in ("_", "-", ".")]).rstrip()
        file_path = os.path.join(self.output_folder, f"{stage}_{safe_filename}")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(success(f"Saved {stage} file: {file_path}"))
        except IOError as e:
            print(error(f"Error saving {stage} file: {str(e)}"))

    def generate_application(self):
        request = self.request_entry.get()
        if not request:
            messagebox.showwarning("Warning", "Please enter an application request.")
            return

        self.show_loading()
        try:
            # Step 1: Generate initial file structure
            print(info("Generating initial file structure..."))
            prompt = f"Create a basic application structure based on this request: {request}\n"
            prompt += "Provide a list of files needed for this application, each on a new line. Include only the file names, not their content."
            response = self.api.complete(prompt)
            if not response:
                raise ValueError("No response received from the API.")
            file_list = [file.strip() for file in response.split('\n') if file.strip()]
            print(success(f"Generated file list: {file_list}"))

            # Step 2: Generate initial content for each file
            self.files = {}
            for file in file_list:
                if file.endswith(':'):  # Skip lines that are just categories
                    continue
                print(info(f"Generating initial content for {file}..."))
                prompt = f"Generate the initial content for the file '{file}' for the application: {request}\n"
                prompt += "Provide only the code, no explanations."
                content = self.api.complete(prompt)
                if not content:
                    print(warning(f"No content generated for {file}"))
                    continue
                self.files[file] = content
                self.save_file(file, content, "initial")
                print(success(f"Initial content generated and saved for {file}"))

            # Step 3: Iterate through files and improve them
            for file in self.files:
                print(info(f"Improving content for {file}..."))
                prompt = f"Improve the following code for {file}:\n\n{self.files[file]}\n\n"
                prompt += "Provide the improved code, no explanations."
                improved_content = self.api.complete(prompt)
                if improved_content:
                    self.files[file] = improved_content
                    self.save_file(file, improved_content, "improved")
                    print(success(f"Content improved and saved for {file}"))
                else:
                    print(warning(f"No improvements made for {file}"))

            # Update file listing
            self.file_list.delete(*self.file_list.get_children())
            for file in self.files:
                self.file_list.insert('', 'end', text=file, values=(file,))

            messagebox.showinfo("Success", f"Application generated and improved successfully!\nFiles saved in: {self.output_folder}")
        except Exception as e:
            error_message = f"An error occurred while generating the application: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)
        finally:
            self.hide_loading()

    def on_file_select(self, event):
        selected_items = self.file_list.selection()
        if not selected_items:
            return  # No file selected, do nothing
        selected_item = selected_items[0]
        file_name = self.file_list.item(selected_item)['text']
        self.code_display.delete('1.0', tk.END)
        self.code_display.insert(tk.END, self.files[file_name])

    def download_zip(self):
        if not self.files:
            messagebox.showwarning("Warning", "No files to download. Generate an application first.")
            return

        if self.output_folder:
            zip_path = os.path.join(self.output_folder, "generated_app.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for filename, content in self.files.items():
                    zipf.writestr(filename, content)
            webbrowser.open('file://' + os.path.dirname(zip_path))
            messagebox.showinfo("Success", f"ZIP file created: {zip_path}")
        else:
            messagebox.showwarning("Warning", "No output folder found. Please generate the application first.")

    def preview_app(self):
        if 'index.html' in self.files:
            if self.output_folder:
                file_path = os.path.join(self.output_folder, "index.html")
                webbrowser.open('file://' + file_path)
            else:
                messagebox.showwarning("Warning", "No output folder found. Please generate the application first.")
        else:
            messagebox.showinfo("Preview", "No index.html file found to preview.")

    def show_loading(self):
        self.loading_label.pack()
        self.root.update()

    def hide_loading(self):
        self.loading_label.pack_forget()
        self.root.update()

    def run(self):
        self.root.mainloop()
