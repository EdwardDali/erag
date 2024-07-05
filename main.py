import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import threading
import os
from file_processing import process_file, append_to_db
from talk2doc import RAGSystem
from embeddings_utils import compute_and_save_embeddings, load_or_compute_embeddings
from sentence_transformers import SentenceTransformer
from create_graph import create_knowledge_graph, create_knowledge_graph_from_raw
from settings import settings
from search_utils import SearchUtils
from create_knol import KnolCreator
from web_sum import WebSum
from web_rag import WebRAG
from route_query import RouteQuery
from api_model import get_available_models, update_settings, configure_api
from talk2model import Talk2Model
from create_sum import run_create_sum



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

        self.create_main_tab()
        self.create_settings_tab()

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

    def create_embeddings_frame(self):
        embeddings_frame = tk.LabelFrame(self.main_tab, text="Embeddings and Graph")
        embeddings_frame.pack(fill="x", padx=10, pady=5)

        execute_embeddings_button = tk.Button(embeddings_frame, text="Execute Embeddings", 
                                              command=self.execute_embeddings)
        execute_embeddings_button.pack(side="left", padx=5, pady=5)

        create_knowledge_graph_button = tk.Button(embeddings_frame, text="Create Knowledge Graph", 
                                                  command=self.create_knowledge_graph)
        create_knowledge_graph_button.pack(side="left", padx=5, pady=5)

        create_knowledge_graph_raw_button = tk.Button(embeddings_frame, text="Create Knowledge Graph from Raw", 
                                                      command=self.create_knowledge_graph_from_raw)
        create_knowledge_graph_raw_button.pack(side="left", padx=5, pady=5)

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

        route_query_button = tk.Button(agent_frame, text="Route Query", command=self.run_route_query)
        route_query_button.pack(side="left", padx=5, pady=5)

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
        models = get_available_models(api_type)
        self.model_menu['values'] = models
        if models:
            if api_type == "ollama" and settings.ollama_model in models:
                self.model_var.set(settings.ollama_model)
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
            if show_message:
                messagebox.showinfo("Model Selected", f"Selected API: {api_type}, Model: {model}")
        elif show_message:
            messagebox.showwarning("Model Selection", "No model selected")

    def create_doc_rag_frame(self):
        rag_frame = tk.LabelFrame(self.main_tab, text="Doc Rag")
        rag_frame.pack(fill="x", padx=10, pady=5)

        talk2doc_button = tk.Button(rag_frame, text="Talk2Doc", command=self.run_model)
        talk2doc_button.pack(side="left", padx=5, pady=5)

        create_knol_button = tk.Button(rag_frame, text="Create Knol", command=self.create_knol)              
        create_knol_button.pack(side="left", padx=5, pady=5)

        # Add the new Create Sum button
        create_sum_button = tk.Button(rag_frame, text="Create Sum", command=self.run_create_sum)
        create_sum_button.pack(side="left", padx=5, pady=5)

    def create_web_rag_frame(self):

        rag_frame = tk.LabelFrame(self.main_tab, text="Web Rag")
        rag_frame.pack(fill="x", padx=10, pady=5)

        web_sum_button = tk.Button(rag_frame, text="Web Sum", command=self.run_web_sum)
        web_sum_button.pack(side="left", padx=5, pady=5)

        web_rag_button = tk.Button(rag_frame, text="Web Rag", command=self.run_web_rag)
        web_rag_button.pack(side="left", padx=5, pady=5)

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

        self.create_settings_fields(summarization_frame, [
            ("Chunk Size", "summarization_chunk_size"),
            ("Summary Size", "summarization_summary_size"),
            ("Combining Number", "summarization_combining_number"),
            ("Final Chunk Size", "summarization_final_chunk_size"),
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
            var = tk.StringVar(value=str(value))
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
            client = configure_api(api_type)
            route_query = RouteQuery(api_type, client)
            
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
            if not os.path.exists(settings.db_file_path):
                messagebox.showwarning("Warning", f"{settings.db_file_path} not found. Please upload some documents first.")
                return

            # Process db.txt
            self.db_embeddings, self.db_indexes, self.db_content = load_or_compute_embeddings(
                self.model, 
                settings.db_file_path, 
                settings.embeddings_file_path
            )
            messagebox.showinfo("Success", f"Embeddings computed and saved successfully. Shape: {self.db_embeddings.shape}")

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

    def on_closing(self):
        settings.save_settings()
        self.master.destroy()

def main():
    root = tk.Tk()
    ERAGGUI(root)
    root.mainloop()

if __name__ == "__main__":
    settings.load_settings()
    main()
