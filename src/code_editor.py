import os
import zipfile
import tempfile
import webbrowser
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import messagebox
import datetime
import logging
import re
from src.api_model import EragAPI
from src.look_and_feel import error, success, warning, info
from src.settings import settings

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CodeEditor:
    def __init__(self, erag_api: EragAPI):
        self.api = erag_api
        self.files = {}
        self.output_folder = None
        self.allowed_file_types = {
            'py': 'Python',
            'js': 'JavaScript',
            'html': 'HTML',
            'css': 'CSS',
            'json': 'JSON',
            'md': 'Markdown'
        }
        print(info(f"CodeEditor initialized with {self.api.api_type} backend."))
        print(info(f"Using model: {self.api.model}"))
        self.setup_gui()

    def sanitize_filename(self, filename: str) -> str:
        # Remove any path components (e.g., 'foo/bar')
        filename = os.path.basename(filename)
        # Remove any non-alphanumeric characters except for periods, hyphens, and underscores
        filename = re.sub(r'[^\w\-_\.]', '', filename)
        # Ensure the filename is not too long (max 255 characters)
        filename = filename[:255]
        return filename

    def setup_gui(self):
        self.root = tk.Toplevel()
        self.root.title("AI-powered Application Generator")
        self.root.geometry("1000x600")

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
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = os.path.join(settings.output_folder, f"generated_app_{timestamp}")
            os.makedirs(self.output_folder, exist_ok=True)
            logging.info(f"Created output folder: {self.output_folder}")

        safe_filename = self.sanitize_filename(filename)
        file_path = os.path.join(self.output_folder, f"{stage}_{safe_filename}")
        logging.debug(f"Attempting to save file: {file_path}")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Saved {stage} file: {file_path}")
        except IOError as e:
            logging.error(f"Error saving {stage} file: {str(e)}")

    def generate_application(self):
        request = self.request_entry.get()
        if not request:
            messagebox.showwarning("Warning", "Please enter an application request.")
            return

        self.show_loading()
        try:
            logging.info(f"Starting application generation with API type: {self.api.api_type}, Model: {self.api.model}")
            # Step 1: Generate initial file structure
            logging.info("Generating initial file structure...")
            prompt = f"""Create a basic application structure based on this request: {request}
            Provide a list of files needed for this application, each on a new line. 
            Include only the file names with appropriate extensions. 
            Use standard naming conventions (e.g., snake_case for Python, camelCase for JavaScript).
            Allowed file types are: {', '.join(self.allowed_file_types.keys())}"""
            messages = [{"role": "user", "content": prompt}]
            response = self.api.chat(messages)
            if not response:
                raise ValueError("No response received from the API.")
            file_list = [file.strip() for file in response.split('\n') if file.strip() and '.' in file]
            file_list = [file for file in file_list if file.split('.')[-1] in self.allowed_file_types]
            logging.info(f"Generated file list: {file_list}")

            # Step 2: Generate initial content for each file
            self.files = {}
            for file in file_list:
                logging.info(f"Generating initial content for {file}...")
                file_type = self.allowed_file_types.get(file.split('.')[-1], "Unknown")
                prompt = f"""Generate the initial content for the file '{file}' ({file_type}) for the application: {request}
                Ensure the code is complete, syntactically correct, and follows best practices for {file_type}.
                Provide only the code, no explanations."""
                messages = [{"role": "user", "content": prompt}]
                content = self.api.chat(messages)
                if not content:
                    logging.warning(f"No content generated for {file}")
                    continue
                self.files[file] = content
                self.save_file(file, content, "initial")
                logging.info(f"Initial content generated and saved for {file}")

            # Step 3: Improve and integrate the files
            logging.info("Improving and integrating files...")
            file_contents = "\n\n".join([f"File: {file}\nContent:\n{content}" for file, content in self.files.items()])
            prompt = f"""Improve and integrate the following files for the application: {request}

    {file_contents}

    Ensure all files work together cohesively. Improve error handling, consistency, and best practices across all files.
    Provide the improved content for each file, starting with the file name on a new line, followed by the improved code."""

            messages = [{"role": "user", "content": prompt}]
            improved_content = self.api.chat(messages)
            if improved_content:
                # Parse the improved content and update files
                current_file = None
                current_content = []
                for line in improved_content.split('\n'):
                    if line.startswith("File: "):
                        if current_file and current_content:
                            self.files[current_file] = '\n'.join(current_content)
                            self.save_file(current_file, self.files[current_file], "improved")
                            logging.info(f"Content improved and saved for {current_file}")
                        current_file = line[6:].strip()
                        current_content = []
                    else:
                        current_content.append(line)
                if current_file and current_content:
                    self.files[current_file] = '\n'.join(current_content)
                    self.save_file(current_file, self.files[current_file], "improved")
                    logging.info(f"Content improved and saved for {current_file}")

            # Update file listing
            self.file_list.delete(*self.file_list.get_children())
            for file in self.files:
                self.file_list.insert('', 'end', text=file, values=(file,))

            messagebox.showinfo("Success", f"Application generated and improved successfully!\nFiles saved in: {self.output_folder}")
        except Exception as e:
            error_message = f"An error occurred while generating the application: {str(e)}"
            logging.error(error_message)
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
