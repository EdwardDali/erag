import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

import tkinter as tk
from tkinter import messagebox, ttk, filedialog, simpledialog
import threading
import asyncio
import os
from src.file_processing import process_file, append_to_db
from src.talk2doc import RAGSystem
from src.embeddings_utils import compute_and_save_embeddings, load_or_compute_embeddings
from sentence_transformers import SentenceTransformer
from src.create_graph import create_knowledge_graph, create_knowledge_graph_from_raw
from src.settings import settings
from src.search_utils import SearchUtils
from src.create_knol import KnolCreator
from src.web_sum import WebSum
from src.web_rag import WebRAG
from src.route_query import RouteQuery
from src.api_model import get_available_models, update_settings, configure_api
from src.talk2model import Talk2Model
from src.create_sum import run_create_sum
from src.talk2url import Talk2URL
from src.talk2git import Talk2Git
from src.create_q import run_create_q
from src.server import ServerManager
from src.gen_a import run_gen_a
from src.look_and_feel import error, success, warning, info, highlight

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class ERAGGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("ERAG")
        self.api_type_var = tk.StringVar(master)
        self.api_type_var.set("ollama")  # Default to ollama
        self.model_var = tk.StringVar(master)
        self.rag_system = None
        self.model = SentenceTransformer(settings.model_name)
        self.db_embeddings = None
        self.db_indexes = None
        self.db_content = None
        self.knowledge_graph = None
        self.web_rag = None
        self.is_initializing = True  # Flag to track initialization
        self.talk2url = None
        self.server_manager = ServerManager()  # Initialize the ServerManager
        self.project_root = project_root

        # Create output folder if it doesn't exist
        os.makedirs(settings.output_folder, exist_ok=True)

        # Create the notebook
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.create_widgets()

        # Set up the window close event
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main")

        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="Settings")

        self.server_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.server_tab, text="llama.cpp server")  # Renamed tab

        self.create_main_tab()
        self.create_settings_tab()
        self.create_server_tab() 

    def create_main_tab(self):
        self.create_model_frame()
        self.create_upload_frame()
        self.create_embeddings_frame()
        self.create_agent_frame()  # New line
        self.create_doc_rag_frame()
        self.create_web_rag_frame()

    def create_upload_frame(self):
        upload_frame = tk.LabelFrame(self.main_tab, text="Upload")
        upload_frame.pack(fill="x", padx=10, pady=5)

        file_types = ["DOCX", "JSON", "PDF", "Text"]
        for file_type in file_types:
            button = tk.Button(upload_frame, text=f"Upload {file_type}", 
                               command=lambda ft=file_type: self.upload_and_chunk(ft))
            button.pack(side="left", padx=5, pady=5)
            ToolTip(button, f"Upload and process a {file_type} file")

    def create_embeddings_frame(self):
        embeddings_frame = tk.LabelFrame(self.main_tab, text="Embeddings and Graph")
        embeddings_frame.pack(fill="x", padx=10, pady=5)

        execute_embeddings_button = tk.Button(embeddings_frame, text="Execute Embeddings", 
                                              command=self.execute_embeddings)
        execute_embeddings_button.pack(side="left", padx=5, pady=5)
        ToolTip(execute_embeddings_button, "Compute and save embeddings for uploaded documents")

        create_knowledge_graph_button = tk.Button(embeddings_frame, text="Create Knowledge Graph", 
                                                  command=self.create_knowledge_graph)
        create_knowledge_graph_button.pack(side="left", padx=5, pady=5)
        ToolTip(create_knowledge_graph_button, "Create a knowledge graph from processed documents")

        create_knowledge_graph_raw_button = tk.Button(embeddings_frame, text="Create Knowledge Graph from Raw", 
                                                      command=self.create_knowledge_graph_from_raw)
        create_knowledge_graph_raw_button.pack(side="left", padx=5, pady=5)
        ToolTip(create_knowledge_graph_raw_button, "Create a knowledge graph from a raw document file")

    def create_model_frame(self):
        model_frame = tk.LabelFrame(self.main_tab, text="Model Selection")
        model_frame.pack(fill="x", padx=10, pady=5)

        # API Type selection
        api_label = tk.Label(model_frame, text="API Type:")
        api_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        
        api_options = ["ollama", "llama"]
        api_menu = ttk.Combobox(model_frame, textvariable=self.api_type_var, values=api_options, state="readonly")
        api_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        api_menu.bind("<<ComboboxSelected>>", self.update_model_list)

        # Model selection
        model_label = tk.Label(model_frame, text="Model:")
        model_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        
        self.model_menu = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly")
        self.model_menu.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.model_menu.bind("<<ComboboxSelected>>", self.update_model_setting)

        # Initialize model list
        self.update_model_list()

    def create_agent_frame(self):
        agent_frame = tk.LabelFrame(self.main_tab, text="Model and Agent")
        agent_frame.pack(fill="x", padx=10, pady=5)

        talk2model_button = tk.Button(agent_frame, text="Talk2Model", command=self.run_talk2model)
        talk2model_button.pack(side="left", padx=5, pady=5)
        ToolTip(talk2model_button, "Start a conversation with the selected model")

        route_query_button = tk.Button(agent_frame, text="Route Query", command=self.run_route_query)
        route_query_button.pack(side="left", padx=5, pady=5)
        ToolTip(route_query_button, "Route a query to the appropriate system or model")

    def run_talk2model(self):
        try:
            api_type = self.api_type_var.get()
            model = self.model_var.get()
            
            # Create and run the Talk2Model instance in a separate thread
            talk2model = Talk2Model(api_type, model)
            threading.Thread(target=talk2model.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"Talk2Model started with {api_type} API and {model} model. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting Talk2Model: {str(e)}")


    def update_model_list(self, event=None):
        api_type = self.api_type_var.get()
        if api_type == "ollama":
            models = get_available_models(api_type)
        elif api_type == "llama":
            models = self.server_manager.get_gguf_models()
        else:
            models = []

        self.model_menu['values'] = models
        if models:
            if api_type == "ollama" and settings.ollama_model in models:
                self.model_var.set(settings.ollama_model)
            elif api_type == "llama" and self.server_manager.current_model in models:
                self.model_var.set(self.server_manager.current_model)
            else:
                self.model_var.set(models[0])
            if not self.is_initializing:
                self.update_model_setting()
        else:
            self.model_var.set("")
        
        if self.is_initializing:
            self.is_initializing = False
            self.update_model_setting(show_message=False)

    def update_model_setting(self, event=None, show_message=True):
        api_type = self.api_type_var.get()
        model = self.model_var.get()
        if model:
            update_settings(settings, api_type, model)
            if api_type == "llama":
                self.server_manager.set_current_model(model)
            if show_message:
                messagebox.showinfo("Model Selected", f"Selected API: {api_type}, Model: {model}")
        elif show_message:
            messagebox.showwarning("Model Selection", "No model selected")

    def create_doc_rag_frame(self):
        rag_frame = tk.LabelFrame(self.main_tab, text="Doc Rag")
        rag_frame.pack(fill="x", padx=10, pady=5)

        talk2doc_button = tk.Button(rag_frame, text="Talk2Doc", command=self.run_model)
        talk2doc_button.pack(side="left", padx=5, pady=5)
        ToolTip(talk2doc_button, "Start a conversation with the RAG system using uploaded documents")

        create_knol_button = tk.Button(rag_frame, text="Create Knol", command=self.create_knol)              
        create_knol_button.pack(side="left", padx=5, pady=5)
        ToolTip(create_knol_button, "Create a knowledge artifact (Knol) from processed documents")

        create_sum_button = tk.Button(rag_frame, text="Create Sum", command=self.run_create_sum)
        create_sum_button.pack(side="left", padx=5, pady=5)
        ToolTip(create_sum_button, "Create a summary of an uploaded document")

        create_q_button = tk.Button(rag_frame, text="Create Q", command=self.run_create_q)
        create_q_button.pack(side="left", padx=5, pady=5)
        ToolTip(create_q_button, "Create questions based on an input document")

        # New 'Gen A' button
        gen_a_button = tk.Button(rag_frame, text="Gen A", command=self.run_gen_a)
        gen_a_button.pack(side="left", padx=5, pady=5)
        ToolTip(gen_a_button, "Generate answers based on existing questions")

    def create_web_rag_frame(self):
        rag_frame = tk.LabelFrame(self.main_tab, text="Web Rag")
        rag_frame.pack(fill="x", padx=10, pady=5)

        web_rag_button = tk.Button(rag_frame, text="Web Rag", command=self.run_web_rag)
        web_rag_button.pack(side="left", padx=5, pady=5)
        ToolTip(web_rag_button, "Start a conversation with the RAG system using web content")

        web_sum_button = tk.Button(rag_frame, text="Web Sum", command=self.run_web_sum)
        web_sum_button.pack(side="left", padx=5, pady=5)
        ToolTip(web_sum_button, "Summarize content from web pages")

        talk2urls_button = tk.Button(rag_frame, text="Talk2URLs", command=self.run_talk2urls)
        talk2urls_button.pack(side="left", padx=5, pady=5)
        ToolTip(talk2urls_button, "Use LLM to interact with content from specific URLs")

        # Add the new Talk2Git button
        talk2git_button = tk.Button(rag_frame, text="Talk2Git", command=self.run_talk2git)
        talk2git_button.pack(side="left", padx=5, pady=5)
        ToolTip(talk2git_button, "Interact with content from a GitHub repository")

    def create_settings_tab(self):
        # Create a main frame to hold the three columns
        main_frame = ttk.Frame(self.settings_tab)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)

        # Left Column
        left_column = ttk.Frame(main_frame)
        left_column.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Middle Column
        middle_column = ttk.Frame(main_frame)
        middle_column.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Right Column
        right_column = ttk.Frame(main_frame)
        right_column.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        # Create frames for different setting categories
        upload_frame = self.create_labelframe(left_column, "Upload Settings", 0)
        embeddings_frame = self.create_labelframe(left_column, "Embeddings Settings", 1)
        graph_frame = self.create_labelframe(left_column, "Graph Settings", 2)
        model_frame = self.create_labelframe(left_column, "Model Settings", 3)

        knol_frame = self.create_labelframe(middle_column, "Knol Creation Settings", 0)
        search_frame = self.create_labelframe(middle_column, "Search Settings", 1)
        file_frame = self.create_labelframe(middle_column, "File Settings", 2)
        web_sum_frame = self.create_labelframe(middle_column, "Web Sum Settings", 3)

        web_rag_frame = self.create_labelframe(right_column, "Web RAG Settings", 0)
        summarization_frame = self.create_labelframe(right_column, "Summarization Settings", 1)
        api_frame = self.create_labelframe(right_column, "API Settings", 2)
        question_gen_frame = self.create_labelframe(right_column, "Question Generation Settings", 3)
        talk2url_frame = self.create_labelframe(right_column, "Talk2URL Settings", 4)
        github_frame = self.create_labelframe(right_column, "GitHub Settings", 5)
        


        # Create and layout settings fields
        self.create_settings_fields(upload_frame, [
            ("Chunk Size", "file_chunk_size"),
            ("Overlap Size", "file_overlap_size"),
        ])

        self.create_settings_fields(embeddings_frame, [
            ("Batch Size", "batch_size"),
            ("Embeddings File Path", "embeddings_file_path"),
            ("DB File Path", "db_file_path"),
        ])

        self.create_settings_fields(graph_frame, [
            ("Graph Chunk Size", "graph_chunk_size"),
            ("Graph Overlap Size", "graph_overlap_size"),
            ("NLP Model", "nlp_model"),
            ("Similarity Threshold", "similarity_threshold"),
            ("Min Entity Occurrence", "min_entity_occurrence"),
            ("Knowledge Graph File Path", "knowledge_graph_file_path"),
        ])

        # Create checkbox for enable_semantic_edges
        self.create_checkbox(graph_frame, "Enable Semantic Edges", "enable_semantic_edges", 
                             len(graph_frame.grid_slaves()), 0)

        self.create_settings_fields(model_frame, [
            ("Max History Length", "max_history_length"),
            ("Conversation Context Size", "conversation_context_size"),
            ("Update Threshold", "update_threshold"),
            ("Ollama Model", "ollama_model"),
            ("Temperature", "temperature"),
            ("Model Name", "model_name"),
        ])

        self.create_settings_fields(knol_frame, [
            ("Number of Questions", "num_questions"),
        ])

        self.create_settings_fields(web_sum_frame, [
            ("Web Sum URLs to Crawl", "web_sum_urls_to_crawl"),
            ("Summary Size", "summary_size"),
            ("Final Summary Size", "final_summary_size"),
        ])

        self.create_settings_fields(web_rag_frame, [
            ("Web Rag URLs to Crawl", "web_rag_urls_to_crawl"),
            ("Initial Context Size", "initial_context_size"),
            ("Web RAG File", "web_rag_file"),
            ("Web RAG Chunk Size", "web_rag_chunk_size"),
            ("Web RAG Overlap Size", "web_rag_overlap_size"),
        ])

        self.create_settings_fields(search_frame, [
            ("Top K", "top_k"),
            ("Entity Relevance Threshold", "entity_relevance_threshold"),
            ("Lexical Weight", "lexical_weight"),
            ("Semantic Weight", "semantic_weight"),
            ("Graph Weight", "graph_weight"),
            ("Text Weight", "text_weight"),
        ])

        self.create_settings_fields(question_gen_frame, [
            ("Initial Question Chunk Size", "initial_question_chunk_size"),
            ("Question Chunk Levels", "question_chunk_levels"),
            ("Excluded Question Levels", "excluded_question_levels"),
            ("Questions Per Chunk", "questions_per_chunk"),  # New field for questions per chunk
        ])

        # Create checkboxes for boolean settings
        checkbox_frame = ttk.Frame(search_frame)
        checkbox_frame.grid(row=len(search_frame.grid_slaves()), column=0, columnspan=2, sticky="w", padx=5, pady=5)
        self.create_checkbox(checkbox_frame, "Enable Lexical Search", "enable_lexical_search", 0, 0)
        self.create_checkbox(checkbox_frame, "Enable Semantic Search", "enable_semantic_search", 0, 1)
        self.create_checkbox(checkbox_frame, "Enable Graph Search", "enable_graph_search", 1, 0)
        self.create_checkbox(checkbox_frame, "Enable Text Search", "enable_text_search", 1, 1)

        self.create_settings_fields(file_frame, [
            ("Results File Path", "results_file_path"),
        ])

        # Create checkbox for talk2url_limit_content_size
        self.create_checkbox(talk2url_frame, "Limit Content Size", "talk2url_limit_content_size", 0, 0)
        
        # Create settings field for Content Size Per URL
        ttk.Label(talk2url_frame, text="Content Size Per URL").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        content_size_var = tk.StringVar(value=str(settings.talk2url_content_size_per_url))
        content_size_entry = ttk.Entry(talk2url_frame, textvariable=content_size_var)
        content_size_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        setattr(self, "talk2url_content_size_per_url_var", content_size_var)

        self.create_settings_fields(summarization_frame, [
            ("Chunk Size", "summarization_chunk_size"),
            ("Summary Size", "summarization_summary_size"),
            ("Combining Number", "summarization_combining_number"),
            ("Final Chunk Size", "summarization_final_chunk_size"),
        ])

        self.create_settings_fields(github_frame, [
            ("GitHub Token", "github_token"),
            ("File Analysis Limit", "file_analysis_limit"),
        ])



        # Add buttons for settings management
        button_frame = ttk.Frame(self.settings_tab)
        button_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        ttk.Button(button_frame, text="Apply Settings", command=self.apply_settings).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_settings).pack(side="left", padx=5)

    def create_labelframe(self, parent, text, row):
        frame = ttk.LabelFrame(parent, text=text)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        return frame

    def create_settings_fields(self, parent, fields):
        for i, (label, key) in enumerate(fields):
            ttk.Label(parent, text=label).grid(row=i, column=0, sticky="e", padx=5, pady=2)
            value = getattr(settings, key)
            if isinstance(value, str) and value.startswith(str(self.project_root)):
                # Convert absolute path to relative path for display
                value = os.path.relpath(value, self.project_root)
            var = tk.StringVar(value=str(value))
            if key == "github_token":
                entry = ttk.Entry(parent, textvariable=var, show="*")
            else:
                entry = ttk.Entry(parent, textvariable=var)
            entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            setattr(self, f"{key}_var", var)

    def create_checkbox(self, parent, text, key, row, column):
        var = tk.BooleanVar(value=getattr(settings, key))
        ttk.Checkbutton(parent, text=text, variable=var).grid(row=row, column=column, sticky="w", padx=5, pady=2)
        setattr(self, f"{key}_var", var)

    def apply_settings(self):
        for key in dir(settings):
            if not key.startswith('_') and hasattr(self, f"{key}_var"):
                value = getattr(self, f"{key}_var").get()
                if isinstance(getattr(settings, key), bool):
                    value = bool(value)
                elif isinstance(getattr(settings, key), int):
                    value = int(value)
                elif isinstance(getattr(settings, key), float):
                    value = float(value)
                elif key == "excluded_question_levels":
                    value = [int(x.strip()) for x in value.split(',') if x.strip().isdigit()]
                elif isinstance(getattr(settings, key), str) and value.startswith(('output', 'Output')):
                    # Convert relative path back to absolute path
                    value = os.path.join(self.project_root, value)
                settings.update_setting(key, value)
        
        settings.apply_settings()
        messagebox.showinfo("Settings", "Settings applied successfully")

    def reset_settings(self):
        settings.reset_to_defaults()
        self.update_settings_display()
        messagebox.showinfo("Settings", "Settings reset to defaults")

    def update_settings_display(self):
        for key in dir(settings):
            if not key.startswith('_') and hasattr(self, f"{key}_var"):
                getattr(self, f"{key}_var").set(str(getattr(settings, key)))

    def run_route_query(self):
        try:
            api_type = self.api_type_var.get()
            model = self.model_var.get()
            
            # Create the RouteQuery instance with just the api_type
            route_query = RouteQuery(api_type)
            
            # Apply settings to RouteQuery
            settings.apply_settings()
            
            # Run the Route Query in a separate thread to keep the GUI responsive
            threading.Thread(target=route_query.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"Route Query system started with {api_type} API and {model} model. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the Route Query system: {str(e)}")

    def run_create_sum(self):
        try:
            file_path = filedialog.askopenfilename(title="Select a book to summarize",
                                                   filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("All files", "*.*")])
            if not file_path:
                messagebox.showwarning("Warning", "No file selected.")
                return

            api_type = self.api_type_var.get()
            model = self.model_var.get()
            client = configure_api(api_type)

            # Apply settings before running the summarization
            self.apply_settings()

            # Run the summarization in a separate thread
            threading.Thread(target=self._create_sum_thread, args=(file_path, api_type, client), daemon=True).start()

            messagebox.showinfo("Info", f"Summarization started for {os.path.basename(file_path)}. Check the console for progress.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the summarization process: {str(e)}")

    def _create_sum_thread(self, file_path, api_type, client):
        try:
            result = run_create_sum(file_path, api_type, client)
            print(result)
            messagebox.showinfo("Success", result)
        except Exception as e:
            error_message = f"An error occurred during summarization: {str(e)}"
            print(error_message)
            messagebox.showerror("Error", error_message)

    def create_knol(self):
        try:
            api_type = self.api_type_var.get()
            creator = KnolCreator(api_type)
            
            if os.path.exists(settings.db_file_path):
                with open(settings.db_file_path, "r", encoding="utf-8") as db_file:
                    creator.db_content = db_file.readlines()
            else:
                creator.db_content = None

            if os.path.exists(settings.embeddings_file_path):
                creator.db_embeddings, _, _ = load_or_compute_embeddings(
                    self.model, 
                    settings.db_file_path, 
                    settings.embeddings_file_path
                )
            else:
                creator.db_embeddings = None

            if os.path.exists(settings.knowledge_graph_file_path):
                creator.knowledge_graph = self.knowledge_graph
            else:
                creator.knowledge_graph = None

            if all([creator.db_embeddings is not None, creator.db_content is not None, creator.knowledge_graph is not None]):
                creator.search_utils = SearchUtils(creator.model, creator.db_embeddings, creator.db_content, creator.knowledge_graph)
            else:
                creator.search_utils = None
                print("Some components are missing. The knol creation process will not use RAG capabilities.")

            threading.Thread(target=creator.run_knol_creator, daemon=True).start()
            messagebox.showinfo("Info", "Knol creation process started. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the knol creation process: {str(e)}")

    def upload_and_chunk(self, file_type: str):
        try:
            content = process_file(file_type)
            if content:
                append_to_db(content)
                messagebox.showinfo("Success", f"{file_type} file content processed and appended to db.txt with table of contents in db_content.txt.")
            else:
                messagebox.showwarning("Warning", "No file selected or file was empty.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the file: {str(e)}")


    def execute_embeddings(self):
        try:
            # Ensure we're using the correct path from settings
            db_file_path = settings.db_file_path
            
            if not os.path.exists(db_file_path):
                messagebox.showwarning("Warning", f"{db_file_path} not found. Please upload some documents first.")
                return

            # Process db.txt
            self.db_embeddings, self.db_indexes, self.db_content = load_or_compute_embeddings(
                self.model, 
                db_file_path, 
                settings.embeddings_file_path
            )
            messagebox.showinfo("Success", f"Embeddings computed and saved successfully to {settings.embeddings_file_path}. Shape: {self.db_embeddings.shape}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while computing embeddings: {str(e)}")

    def create_knowledge_graph(self):
        try:
            if not os.path.exists(settings.db_file_path) or not os.path.exists(settings.embeddings_file_path):
                messagebox.showwarning("Warning", f"{settings.db_file_path} or {settings.embeddings_file_path} not found. Please upload documents and execute embeddings first.")
                return

            self.knowledge_graph = create_knowledge_graph()
            if self.knowledge_graph:
                doc_nodes = [n for n, d in self.knowledge_graph.nodes(data=True) if d['type'] == 'document']
                chunk_nodes = [n for n, d in self.knowledge_graph.nodes(data=True) if d['type'] == 'chunk']
                entity_nodes = [n for n, d in self.knowledge_graph.nodes(data=True) if d['type'] == 'entity']
                messagebox.showinfo("Success", f"Knowledge graph created with {len(doc_nodes)} document nodes, {len(chunk_nodes)} chunk nodes, {len(entity_nodes)} entity nodes, and {self.knowledge_graph.number_of_edges()} edges. Saved as {settings.knowledge_graph_file_path}.")
            else:
                messagebox.showwarning("Warning", "Failed to create knowledge graph.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while creating the knowledge graph: {str(e)}")

    def create_knowledge_graph_from_raw(self):
        try:
            raw_file_path = filedialog.askopenfilename(title="Select Raw Document File",
                                                       filetypes=[("Text Files", "*.txt")])
            if not raw_file_path:
                messagebox.showwarning("Warning", "No file selected.")
                return

            self.knowledge_graph = create_knowledge_graph_from_raw(raw_file_path)
            if self.knowledge_graph:
                doc_nodes = [n for n, d in self.knowledge_graph.nodes(data=True) if d['type'] == 'document']
                chunk_nodes = [n for n, d in self.knowledge_graph.nodes(data=True) if d['type'] == 'chunk']
                entity_nodes = [n for n, d in self.knowledge_graph.nodes(data=True) if d['type'] == 'entity']
                messagebox.showinfo("Success", f"Knowledge graph created from raw documents with {len(doc_nodes)} document nodes, {len(chunk_nodes)} chunk nodes, {len(entity_nodes)} entity nodes, and {self.knowledge_graph.number_of_edges()} edges. Saved as {settings.knowledge_graph_file_path}.")
            else:
                messagebox.showwarning("Warning", "Failed to create knowledge graph from raw documents.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while creating the knowledge graph from raw documents: {str(e)}")

    def run_model(self):
        try:
            api_type = self.api_type_var.get()
            self.rag_system = RAGSystem(api_type)
            
            # Apply settings to RAGSystem
            settings.apply_settings()
            
            # Run the CLI in a separate thread to keep the GUI responsive
            threading.Thread(target=self.rag_system.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"RAG system started with {api_type} API. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the RAG system: {str(e)}")

    def run_talk2urls(self):
        try:
            api_type = self.api_type_var.get()
            self.talk2url = Talk2URL(api_type)
            
            # Apply settings to Talk2URL
            settings.apply_settings()
            
            # Run Talk2URL in a separate thread to keep the GUI responsive
            threading.Thread(target=self.talk2url.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"Talk2URLs system started with {api_type} API. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the Talk2URLs system: {str(e)}")

    def run_talk2git(self):
        try:
            api_type = self.api_type_var.get()
            github_token = settings.github_token  # Get the GitHub token from settings
            self.talk2git = Talk2Git(api_type, github_token)
            
            # Apply settings to Talk2Git
            settings.apply_settings()
            
            # Run Talk2Git in a separate thread to keep the GUI responsive
            threading.Thread(target=self.talk2git.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"Talk2Git system started with {api_type} API. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the Talk2Git system: {str(e)}")

    def run_create_q(self):
        try:
            file_path = filedialog.askopenfilename(title="Select a document to create questions from",
                                                   filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("All files", "*.*")])
            if not file_path:
                messagebox.showwarning("Warning", "No file selected.")
                return

            api_type = self.api_type_var.get()
            model = self.model_var.get()
            client = configure_api(api_type)

            # Apply settings before running the question creation
            self.apply_settings()

            # Run the question creation in a separate thread
            threading.Thread(target=self._create_q_thread, args=(file_path, api_type, client), daemon=True).start()

            messagebox.showinfo("Info", f"Question creation started for {os.path.basename(file_path)}. Check the console for progress.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the question creation process: {str(e)}")

    def _create_q_thread(self, file_path, api_type, client):
        try:
            result = run_create_q(file_path, api_type, client)
            print(result)
            messagebox.showinfo("Success", "Questions created successfully. Check the output file.")
        except Exception as e:
            error_message = f"An error occurred during question creation: {str(e)}"
            print(error_message)
            messagebox.showerror("Error", error_message)



    def run_web_sum(self):
        try:
            api_type = self.api_type_var.get()
            web_sum = WebSum(api_type)
            
            # Apply settings to WebSum
            settings.apply_settings()
            
            # Run the Web Sum in a separate thread to keep the GUI responsive
            threading.Thread(target=web_sum.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"Web Sum system started with {api_type} API. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the Web Sum system: {str(e)}")

    def run_web_rag(self):
        try:
            api_type = self.api_type_var.get()
            self.web_rag = WebRAG(api_type)
            
            # Apply settings to WebRAG
            settings.apply_settings()
            
            # Run the Web RAG in a separate thread to keep the GUI responsive
            threading.Thread(target=self.web_rag.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"Web RAG system started with {api_type} API. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the Web RAG system: {str(e)}")


    def run_gen_a(self):
        try:
            # Open file dialog to select the questions file
            questions_file = filedialog.askopenfilename(title="Select Questions File",
                                                        filetypes=[("Text files", "*.txt")])
            if not questions_file:
                print(error("No file selected. Exiting."))
                return

            # Read questions and count them
            with open(questions_file, 'r', encoding='utf-8') as file:
                questions = [line.strip() for line in file if line.strip()]
            
            print(f"\n{success(f'Question file imported correctly, {len(questions)} questions identified.')}")

            # Display options to the user in the console
            print(f"\n{info('Choose the answer generation method:')}")
            print(warning("1. Talk2Doc"))
            print(warning("2. WebRAG"))
            print(warning("3. Hybrid (Talk2Doc + WebRAG)"))
            print(warning("4. Exit"))
            
            while True:
                choice = input(success("Enter your choice (1, 2, 3, or 4): ")).strip()
                if choice in ['1', '2', '3', '4']:
                    break
                print(error("Invalid choice. Please enter 1, 2, 3, or 4."))

            if choice == '4':
                print(info("Exiting the answer generation process."))
                return

            gen_method = {
                '1': 'talk2doc',
                '2': 'web_rag',
                '3': 'hybrid'
            }[choice]

            api_type = self.api_type_var.get()
            client = configure_api(api_type)

            # Apply settings silently
            self.apply_settings()

            # Run the answer generation in a separate thread
            threading.Thread(target=self._gen_a_thread, args=(questions_file, gen_method, api_type, client), daemon=True).start()

            print(info(f"Answer generation started using {gen_method} method. Check the console for progress."))
        except Exception as e:
            print(error(f"An error occurred while starting the answer generation process: {str(e)}"))

    def _gen_a_thread(self, questions_file, gen_method, api_type, client):
        try:
            from src.gen_a import run_gen_a
            result = run_gen_a(questions_file, gen_method, api_type, client)
            print(result)
            messagebox.showinfo("Success", "Answers generated successfully. Check the output file.")
        except Exception as e:
            error_message = f"An error occurred during answer generation: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)
            

    def create_server_tab(self):
        # Enable/Disable on start
        enable_frame = ttk.Frame(self.server_tab)
        enable_frame.pack(fill="x", padx=10, pady=5)
        self.enable_var = tk.BooleanVar(value=self.server_manager.enable_on_start)
        ttk.Checkbutton(enable_frame, text="Enable server on start", variable=self.enable_var, 
                        command=self.toggle_server_on_start).pack(side="left")

        # Server executable location
        exe_frame = ttk.Frame(self.server_tab)
        exe_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(exe_frame, text="Server Executable:").pack(side="left")
        self.exe_var = tk.StringVar(value=self.server_manager.server_executable)
        ttk.Entry(exe_frame, textvariable=self.exe_var).pack(side="left", expand=True, fill="x")
        ttk.Button(exe_frame, text="Browse", command=self.browse_server_exe).pack(side="left")

        # Model folder selection
        folder_frame = ttk.Frame(self.server_tab)
        folder_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(folder_frame, text="Model Folder:").pack(side="left")
        self.folder_var = tk.StringVar(value=self.server_manager.model_folder)
        ttk.Entry(folder_frame, textvariable=self.folder_var).pack(side="left", expand=True, fill="x")
        ttk.Button(folder_frame, text="Browse", command=self.browse_model_folder).pack(side="left")

        # Additional arguments
        args_frame = ttk.Frame(self.server_tab)
        args_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(args_frame, text="Additional Arguments:").pack(side="left")
        self.args_var = tk.StringVar(value=self.server_manager.additional_args)
        ttk.Entry(args_frame, textvariable=self.args_var).pack(side="left", expand=True, fill="x")

        # Output mode selection
        output_frame = ttk.Frame(self.server_tab)
        output_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(output_frame, text="Output Mode:").pack(side="left")
        self.output_mode_var = tk.StringVar(value=self.server_manager.output_mode)
        ttk.Radiobutton(output_frame, text="File", variable=self.output_mode_var, value="file", command=self.set_output_mode).pack(side="left")
        ttk.Radiobutton(output_frame, text="Window", variable=self.output_mode_var, value="window", command=self.set_output_mode).pack(side="left")

        # Restart server button
        ttk.Button(self.server_tab, text="Restart Server", command=self.restart_server).pack(pady=10)

    def toggle_server_on_start(self):
        self.server_manager.enable_on_start = self.enable_var.get()
        self.server_manager.save_config()
        self.check_server_status()

    def check_server_status(self):
        if self.server_manager.enable_on_start:
            if (self.server_manager.server_executable and
                self.server_manager.model_folder and
                self.server_manager.get_gguf_models()):
                # All conditions met, server can be started
                messagebox.showinfo("Server Status", "Server will start automatically on next launch.")
            else:
                # Missing required settings
                messagebox.showwarning("Server Status", "Cannot enable server start. Please ensure server executable, model folder, and at least one model are set.")
                self.enable_var.set(False)
                self.server_manager.enable_on_start = False
                self.server_manager.save_config()

    def browse_server_exe(self):
        path = filedialog.askopenfilename(title="Select server executable", 
                                          filetypes=[("Executable files", "*.exe")])
        if path:
            self.exe_var.set(path)
            self.server_manager.server_executable = path
            self.server_manager.save_config()

    def browse_model_folder(self):
        folder = filedialog.askdirectory(title="Select model folder")
        if folder:
            self.folder_var.set(folder)
            self.server_manager.set_model_folder(folder)
            self.update_model_list()  # Update both main and server model lists

    def set_output_mode(self):
        self.server_manager.set_output_mode(self.output_mode_var.get())

    def restart_server(self):
        self.server_manager.server_executable = self.exe_var.get()
        self.server_manager.additional_args = self.args_var.get()
        self.server_manager.save_config()
        self.server_manager.restart_server()

    def on_closing(self):
        settings.save_settings()
        self.server_manager.stop_server()
        self.master.destroy()

    def run_model(self):
        try:
            api_type = self.api_type_var.get()
            model = self.model_var.get()
            
            if api_type == "ollama":
                self.rag_system = RAGSystem(api_type)
                # Apply settings to RAGSystem
                settings.apply_settings()
                # Run the CLI in a separate thread to keep the GUI responsive
                threading.Thread(target=self.rag_system.run, daemon=True).start()
            elif api_type == "llama":
                # Ensure the server is running with the selected model
                self.server_manager.set_current_model(model)
                self.server_manager.restart_server()
                # Start the llama.cpp client (you'll need to implement this)
                threading.Thread(target=self.run_llama_client, daemon=True).start()
            
            messagebox.showinfo("Info", f"System started with {api_type} API and model: {model}. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the system: {str(e)}")

    def run_llama_client(self):
        try:
            from src.talk2doc import RAGSystem
            rag_system = RAGSystem("llama")
            rag_system.run()
        except Exception as e:
            print(f"An error occurred while running the llama.cpp client: {str(e)}")

def main():
    root = tk.Tk()
    gui = ERAGGUI(root)
    if gui.server_manager.enable_on_start:
        if (gui.server_manager.server_executable and
            gui.server_manager.model_folder and
            gui.server_manager.get_gguf_models()):
            gui.server_manager.start_server()
        else:
            messagebox.showwarning("Server Start Failed", "Cannot start server. Please check server settings.")
    root.mainloop()

if __name__ == "__main__":
    settings.load_settings()
    main()
