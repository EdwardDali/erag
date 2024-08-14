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
from src.api_model import EragAPI, create_erag_api
from src.look_and_feel import error, success, warning, info
from src.settings import settings

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CodeEditor:
    def __init__(self, worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, manager_erag_api: EragAPI = None):
        self.worker_api = worker_erag_api
        self.supervisor_api = supervisor_erag_api
        self.manager_api = manager_erag_api
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
        print(info(f"CodeEditor initialized with worker API: {self.worker_api.api_type}, supervisor API: {self.supervisor_api.api_type}"))
        if self.manager_api:
            print(info(f"Manager API: {self.manager_api.api_type}"))
        self.setup_gui()

    def clean_api_response(self, response, file_type):
            # Remove any text before and after the code block
            code_block_pattern = r'```[\w\s]*\n([\s\S]*?)\n```'
            code_blocks = re.findall(code_block_pattern, response)
            
            if code_blocks:
                # If code blocks are found, use the last one (in case there are multiple)
                cleaned_code = code_blocks[-1].strip()
            else:
                # If no code blocks are found, use the entire response
                cleaned_code = response.strip()
            
            # Remove any remaining markdown code block syntax
            cleaned_code = cleaned_code.replace('```' + file_type, '').replace('```', '')
            
            # Remove any "Here is the improved/polished version of the code:" or similar phrases
            cleaned_code = re.sub(r'^.*?(Here is|This is|Updated|Improved|Polished).*?\n', '', cleaned_code, flags=re.IGNORECASE | re.MULTILINE)
            
            return cleaned_code.strip()

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

    def save_file(self, filename, content, stage):
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
            # Step 1: Worker generates initial file structure and content
            file_list, initial_content = self.worker_generate_initial(request)

            # Step 2: Supervisor improves the files
            supervisor_improved_content = self.supervisor_improve(request, initial_content)

            # Step 3: Manager further improves the files (if available)
            if self.manager_api:
                final_content = self.manager_improve(request, supervisor_improved_content)
            else:
                final_content = supervisor_improved_content

            # Step 4: Perform quality checks and rollback if necessary
            best_content = self.perform_quality_checks(initial_content, supervisor_improved_content, final_content)

            # Update file listing and save files
            self.update_file_listing(best_content)

            messagebox.showinfo("Success", f"Application generated and improved successfully!\nFiles saved in: {self.output_folder}")
        except Exception as e:
            error_message = f"An error occurred while generating the application: {str(e)}"
            logging.error(error_message)
            messagebox.showerror("Error", error_message)
        finally:
            self.hide_loading()

    def worker_generate_initial(self, request):
        logging.info("Worker: Generating initial file structure and content...")
        file_structure_prompt = f"""Create a basic application structure based on this request: {request}
        Provide a list of files needed for this application, each on a new line. 
        Include only the file names with appropriate extensions. 
        Use standard naming conventions (e.g., snake_case for Python, camelCase for JavaScript).
        Allowed file types are: {', '.join(self.allowed_file_types.keys())}"""
        
        file_list_response = self.worker_api.chat([{"role": "user", "content": file_structure_prompt}])
        file_list = [file.strip() for file in file_list_response.split('\n') if file.strip() and '.' in file]
        file_list = [file for file in file_list if file.split('.')[-1] in self.allowed_file_types]

        initial_content = {}
        for file in file_list:
            file_type = self.allowed_file_types.get(file.split('.')[-1], "Unknown")
            content_prompt = f"""Generate the initial content for the file '{file}' ({file_type}) for the application: {request}
            Ensure the code is complete, syntactically correct, and follows best practices for {file_type}.
            Provide only the code, no explanations."""
            content = self.worker_api.chat([{"role": "user", "content": content_prompt}])
            initial_content[file] = content
            self.save_file(file, content, "initial")

        return file_list, initial_content

    def supervisor_improve(self, request, content):
        logging.info("Supervisor: Improving and integrating files...")
        improved_content = {}
        for file, file_content in content.items():
            file_type = self.allowed_file_types.get(file.split('.')[-1], "Unknown")
            improve_prompt = f"""Improve the following {file_type} code for the file '{file}' in the application: {request}

Current content:
{file_content}

Please provide an improved version of this code. Focus on:
1. Enhancing functionality and features
2. Improving code structure and organization
3. Implementing best practices and design patterns
4. Optimizing performance
5. Enhancing error handling and security

IMPORTANT: 
- Ensure that your improvements do not break existing functionality or reduce the quality of the code.
- If you cannot meaningfully improve the code without risking its functionality, return the original code unchanged.
- Provide ONLY the improved code, without any explanations or markdown formatting.

"""

            improved_file_content = self.supervisor_api.chat([{"role": "user", "content": improve_prompt}])
            cleaned_content = self.clean_api_response(improved_file_content, file_type)
            improved_content[file] = cleaned_content
            self.save_file(file, cleaned_content, "supervisor_improved")

        return improved_content
    
    def manager_improve(self, request, content):
        logging.info("Manager: Further improving the application...")
        final_content = {}
        for file, file_content in content.items():
            file_type = self.allowed_file_types.get(file.split('.')[-1], "Unknown")
            improve_prompt = f"""Further improve the following {file_type} code for the file '{file}' in the application: {request}

Current content:
{file_content}

Please provide a final, polished version of this code. Focus on:
1. Ensuring all features are fully implemented
2. Optimizing code efficiency and performance
3. Enhancing code readability and maintainability
4. Implementing advanced error handling and logging
5. Ensuring security best practices are followed
6. Adding helpful comments and documentation

IMPORTANT: 
- Ensure that your improvements do not break existing functionality or reduce the quality of the code.
- If you cannot meaningfully improve the code without risking its functionality, return the original code unchanged.
- Provide ONLY the improved code, without any explanations or markdown formatting.

"""

            final_file_content = self.manager_api.chat([{"role": "user", "content": improve_prompt}])
            cleaned_content = self.clean_api_response(final_file_content, file_type)
            final_content[file] = cleaned_content
            self.save_file(file, cleaned_content, "manager_improved")

        return final_content
    
    def perform_quality_checks(self, initial_content, supervisor_content, manager_content):
        logging.info("Performing quality checks...")
        best_content = {}
        for file in initial_content.keys():
            initial = initial_content[file]
            supervisor = supervisor_content[file]
            manager = manager_content[file] if manager_content else supervisor

            # Perform quality checks
            initial_score = self.evaluate_code_quality(file, initial)
            supervisor_score = self.evaluate_code_quality(file, supervisor)
            manager_score = self.evaluate_code_quality(file, manager)

            logging.info(f"Quality scores for {file}: Initial: {initial_score}, Supervisor: {supervisor_score}, Manager: {manager_score}")

            # Choose the best version
            if manager_score >= supervisor_score and manager_score >= initial_score:
                best_content[file] = manager
                stage = "manager"
            elif supervisor_score >= initial_score:
                best_content[file] = supervisor
                stage = "supervisor"
            else:
                best_content[file] = initial
                stage = "initial"

            logging.info(f"Chosen version for {file}: {stage}")
            self.save_file(file, best_content[file], f"final_{stage}")

        return best_content
    
    def evaluate_code_quality(self, filename, content):
        file_type = filename.split('.')[-1]
        evaluate_prompt = f"""Evaluate the quality of the following {file_type} code:

{content}

Please rate the code on a scale from 1 to 10 (10 being the highest quality) based on the following criteria:
1. Functionality (Does it work as intended?)
2. Code structure and organization
3. Adherence to best practices
4. Performance and efficiency
5. Error handling and security
6. Readability and maintainability

Provide only a single number as your rating, with no explanation."""

        try:
            rating = float(self.worker_api.chat([{"role": "user", "content": evaluate_prompt}]))
            return rating
        except ValueError:
            logging.error(f"Failed to get a numerical rating for {filename}. Defaulting to 5.")
            return 5.0

    def parse_improved_content(self, content):
        improved_files = {}
        current_file = None
        current_content = []
        for line in content.split('\n'):
            if line.startswith("File: "):
                if current_file and current_content:
                    improved_files[current_file] = '\n'.join(current_content)
                    self.save_file(current_file, improved_files[current_file], "improved")
                current_file = line[6:].strip()
                current_content = []
            else:
                current_content.append(line)
        if current_file and current_content:
            improved_files[current_file] = '\n'.join(current_content)
            self.save_file(current_file, improved_files[current_file], "improved")
        return improved_files

    def format_content_for_review(self, content):
        return "\n\n".join([f"File: {file}\nContent:\n{file_content}" for file, file_content in content.items()])

    def update_file_listing(self, content):
        self.file_list.delete(*self.file_list.get_children())
        for file in content:
            self.file_list.insert('', 'end', text=file, values=(file,))
        self.files = content

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
