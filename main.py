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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
from dotenv import load_dotenv, set_key  # Add set_key here
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
from src.api_model import get_available_models, update_settings, EragAPI, create_erag_api
from src.talk2model import Talk2Model
from src.create_sum import run_create_sum
from src.talk2url import Talk2URL
from src.talk2git import Talk2Git
from src.create_q import run_create_q
from src.server import ServerManager
from src.gen_a import run_gen_a
from src.look_and_feel import error, success, warning, info, highlight
from src.talk2sd import Talk2SD
from src.x_da import ExploratoryDataAnalysis
from src.file_processing import upload_multiple_files, FileProcessor
from src.self_knol import SelfKnolCreator
from src.ax_da_b1 import AdvancedExploratoryDataAnalysisB1  
from src.ax_da_b2 import AdvancedExploratoryDataAnalysisB2  
from src.i_da import InnovativeDataAnalysis
from src.ax_da_b3 import AdvancedExploratoryDataAnalysisB3
from src.ax_da_b4 import AdvancedExploratoryDataAnalysisB4
from src.fn_processing import process_financial_data
from src.f_da import FinancialExploratoryDataAnalysis
from src.ax_da_b5 import AdvancedExploratoryDataAnalysisB5
from src.ax_da_b6 import AdvancedExploratoryDataAnalysisB6
from src.ax_da_b7 import AdvancedExploratoryDataAnalysisB7
from src.merge_sd import merge_structured_data
from src.code_editor import CodeEditor

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()


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
        self.erag_api = None
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.file_processor = FileProcessor()
        self.supervisor_model_var = tk.StringVar(master)
        self.manager_model_var = tk.StringVar(master)
        self.default_manager_model_var = tk.StringVar(master)
        self.embedding_class_var = tk.StringVar(master)
        self.embedding_class_var.set(settings.default_embedding_class)
        self.embedding_model_var = tk.StringVar(master)
        self.is_initializing = True

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
        self.notebook.add(self.server_tab, text="llama.cpp server")

        self.create_main_tab()
        self.create_settings_tab()
        self.create_server_tab()
        self.is_initializing = False

    def create_main_tab(self):
        self.create_embedding_model_frame()
        self.create_model_frame()
        self.create_upload_frame()
        self.create_embeddings_frame()
        self.create_agent_frame()
        self.create_doc_rag_frame()
        self.create_web_rag_frame()
        self.create_structured_data_rag_frame()
        self.create_financial_data_rag_frame()

    def create_embedding_model_frame(self):
        embedding_model_frame = tk.LabelFrame(self.main_tab, text="Embedding Model Selection")
        embedding_model_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(embedding_model_frame, text="Embedding Class:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.embedding_class_var = tk.StringVar(value=settings.default_embedding_class)
        embedding_class_options = ["ollama", "sentence_transformers"]
        embedding_class_menu = ttk.Combobox(embedding_model_frame, textvariable=self.embedding_class_var, values=embedding_class_options, state="readonly", width=20)
        embedding_class_menu.grid(row=0, column=1, padx=5, pady=5)
        embedding_class_menu.bind("<<ComboboxSelected>>", self.update_embedding_model_list)

        tk.Label(embedding_model_frame, text="Embedding Model:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.embedding_model_var = tk.StringVar(value=settings.default_embedding_model)
        self.embedding_model_menu = ttk.Combobox(embedding_model_frame, textvariable=self.embedding_model_var, state="readonly", width=30)
        self.embedding_model_menu.grid(row=0, column=3, padx=5, pady=5)
        self.embedding_model_menu.bind("<<ComboboxSelected>>", self.update_embedding_model_setting)

        # Initialize model lists
        self.update_embedding_model_list()



    def create_structured_data_rag_frame(self):
        rag_frame = tk.LabelFrame(self.main_tab, text="Structured Data RAG")
        rag_frame.pack(fill="x", padx=10, pady=5)

        talk2sd_button = tk.Button(rag_frame, text="Talk2SD", command=self.run_talk2sd)
        talk2sd_button.pack(side="left", padx=5, pady=5)
        ToolTip(talk2sd_button, "Start a conversation with the structured data using SQL queries")

        xda_button = tk.Button(rag_frame, text="XDA", command=self.run_xda)
        xda_button.pack(side="left", padx=5, pady=5)
        ToolTip(xda_button, "Perform Exploratory Data Analysis on a selected SQLite database")

        axda_b1_button = tk.Button(rag_frame, text="A-XDA-B1", command=self.run_axda_b1)
        axda_b1_button.pack(side="left", padx=5, pady=5)
        ToolTip(axda_b1_button, "Perform Advanced Exploratory Data Analysis (Batch 1) on a selected SQLite database")

        axda_b2_button = tk.Button(rag_frame, text="A-XDA-B2", command=self.run_axda_b2)
        axda_b2_button.pack(side="left", padx=5, pady=5)
        ToolTip(axda_b2_button, "Perform Advanced Exploratory Data Analysis (Batch 2) on a selected SQLite database")

        axda_b3_button = tk.Button(rag_frame, text="A-XDA-B3", command=self.run_axda_b3)
        axda_b3_button.pack(side="left", padx=5, pady=5)
        ToolTip(axda_b3_button, "Perform Advanced Exploratory Data Analysis (Batch 3) on a selected SQLite database")

        axda_b4_button = tk.Button(rag_frame, text="A-XDA-B4", command=self.run_axda_b4)
        axda_b4_button.pack(side="left", padx=5, pady=5)
        ToolTip(axda_b4_button, "Perform Advanced Exploratory Data Analysis (Batch 4) on a selected SQLite database")

        axda_b5_button = tk.Button(rag_frame, text="A-XDA-B5", command=self.run_axda_b5)
        axda_b5_button.pack(side="left", padx=5, pady=5)
        ToolTip(axda_b5_button, "Perform Advanced Exploratory Data Analysis (Batch 5) on a selected SQLite database")

        axda_b6_button = tk.Button(rag_frame, text="A-XDA-B6", command=self.run_axda_b6)
        axda_b6_button.pack(side="left", padx=5, pady=5)
        ToolTip(axda_b6_button, "Perform Advanced Exploratory Data Analysis (Batch 6) on a selected SQLite database")

        axda_b7_button = tk.Button(rag_frame, text="A-XDA-B7", command=self.run_axda_b7)
        axda_b7_button.pack(side="left", padx=5, pady=5)
        ToolTip(axda_b7_button, "Perform Advanced Exploratory Data Analysis (Batch 7) on a selected SQLite database")

        ixda_button = tk.Button(rag_frame, text="I-XDA", command=self.run_ixda)
        ixda_button.pack(side="left", padx=5, pady=5)
        ToolTip(ixda_button, "Perform Innovative Exploratory Data Analysis on a selected SQLite database")

    def create_financial_data_rag_frame(self):
        financial_frame = tk.LabelFrame(self.main_tab, text="Financial Data RAG")
        financial_frame.pack(fill="x", padx=10, pady=5)

        fxda_button = tk.Button(financial_frame, text="F-XDA", command=self.run_fxda)
        fxda_button.pack(side="left", padx=5, pady=5)
        ToolTip(fxda_button, "Perform Financial Exploratory Data Analysis on a selected SQLite database")




    def create_upload_frame(self):
        upload_frame = tk.LabelFrame(self.main_tab, text="Upload Data")
        upload_frame.pack(fill="x", padx=10, pady=5)

        file_types = ["DOCX", "JSON", "PDF", "Text"]
        for file_type in file_types:
            button = tk.Button(upload_frame, text=f"Upload {file_type}", 
                               command=lambda ft=file_type: upload_multiple_files(ft))
            button.pack(side="left", padx=5, pady=5)
            ToolTip(button, f"Upload and process multiple unstructured data {file_type} files")

        # Add new button for merging structured data
        merge_sd_button = tk.Button(upload_frame, text="Merge SD", 
                                    command=self.merge_structured_data)
        merge_sd_button.pack(side="left", padx=5, pady=5)
        ToolTip(merge_sd_button, "Merge two structured data files (CSV or XLSX)")

        # Add new button for structured data
        structured_data_button = tk.Button(upload_frame, text="Upload Structured Data", 
                                           command=self.upload_structured_data)
        structured_data_button.pack(side="left", padx=5, pady=5)
        ToolTip(structured_data_button, "Upload and process structured data (CSV or XLSX)")

        # Add new button for financial data
        financial_data_button = tk.Button(upload_frame, text="Upload Financial Data", 
                                          command=self.upload_financial_data)
        financial_data_button.pack(side="left", padx=5, pady=5)
        ToolTip(financial_data_button, "Upload and process financial data (CSV or XLSX)")

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

        #  API Type selection
        tk.Label(model_frame, text="API Type:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        api_options = ["ollama", "llama", "groq", "gemini"]
        api_menu = ttk.Combobox(model_frame, textvariable=self.api_type_var, values=api_options, state="readonly", width=15)
        api_menu.grid(row=0, column=1, padx=5, pady=5)
        api_menu.bind("<<ComboboxSelected>>", self.update_model_list)

        # Worker Model selection
        tk.Label(model_frame, text="Worker Model:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.model_menu = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=30)
        self.model_menu.grid(row=0, column=3, padx=5, pady=5)
        self.model_menu.bind("<<ComboboxSelected>>", self.update_model_setting)

        # Supervisor Model selection
        tk.Label(model_frame, text="Supervisor Model:").grid(row=0, column=4, padx=5, pady=5, sticky="e")
        self.supervisor_model_menu = ttk.Combobox(model_frame, textvariable=self.supervisor_model_var, state="readonly", width=30)
        self.supervisor_model_menu.grid(row=0, column=5, padx=5, pady=5)

        # Update the Manager Model selection to use the default from settings
        tk.Label(model_frame, text="Manager Model:").grid(row=0, column=6, padx=5, pady=5, sticky="e")
        self.manager_model_var = tk.StringVar(self.master, value=settings.default_manager_model if settings.default_manager_model else 'None')
        self.manager_model_menu = ttk.Combobox(model_frame, textvariable=self.manager_model_var, state="readonly", width=30)
        self.manager_model_menu.grid(row=0, column=7, padx=5, pady=5)

         # Add Re-ranker Model selection
        tk.Label(model_frame, text="Re-ranker Model:").grid(row=0, column=8, padx=5, pady=5, sticky="e")
        self.reranker_model_var = tk.StringVar(value=settings.reranker_model)
        self.reranker_model_menu = ttk.Combobox(model_frame, textvariable=self.reranker_model_var, state="readonly", width=30)
        self.reranker_model_menu.grid(row=0, column=9, padx=5, pady=5)



        # Initialize model list
        self.update_model_list()


    def update_embedding_model_list(self, event=None):
        embedding_class = self.embedding_class_var.get()
        if embedding_class == "sentence_transformers":
            models = [settings.sentence_transformer_model]
        elif embedding_class == "ollama":
            models = [settings.ollama_embedding_model]
        else:
            models = []

        self.embedding_model_menu['values'] = models
        if models:
            self.embedding_model_var.set(models[0])
        
        # Update settings
        settings.update_setting("default_embedding_class", embedding_class)
        settings.update_setting("default_embedding_model", self.embedding_model_var.get())
        
        # Print changes to console
        print(success(f"Updated embedding model list for embedding class: {embedding_class}"))
        print(success(f"Available embedding models: {', '.join(models)}"))
        print(success(f"Selected embedding model: {self.embedding_model_var.get()}"))

        # Update EragAPI if it exists and we're not initializing
        if hasattr(self, 'erag_api') and self.erag_api is not None and not self.is_initializing:
            self.erag_api.embedding_class = embedding_class
            self.erag_api.embedding_model = self.embedding_model_var.get()
            print(info("EragAPI updated with new embedding settings"))

        # Show message box only if we're not initializing
        if not self.is_initializing:
            messagebox.showinfo("Embedding Settings Updated", 
                                f"Embedding class: {embedding_class}\n"
                                f"Embedding model: {self.embedding_model_var.get()}")
    
    def get_ollama_embedding_models(self):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            models = result.stdout.strip().split('\n')[1:]  # Skip the header
            # Filter for embedding models (you may need to adjust this filtering logic)
            embedding_models = [model.split()[0] for model in models if 'embedding' in model.lower()]
            return embedding_models if embedding_models else ["chroma/all-minilm-l6-v2-f32:latest"]
        except subprocess.CalledProcessError:
            print(error("Error running 'ollama list' command"))
            return ["chroma/all-minilm-l6-v2-f32:latest"]

    def update_embedding_model_setting(self, event=None):
        embedding_model = self.embedding_model_var.get()
        
        # Update settings
        settings.update_setting("default_embedding_model", embedding_model)
        
        # Print changes to console
        print(success(f"Selected embedding model: {embedding_model}"))

        # Update EragAPI if it exists and we're not initializing
        if hasattr(self, 'erag_api') and self.erag_api is not None and not self.is_initializing:
            self.erag_api.embedding_model = embedding_model
            print(info("EragAPI updated with new embedding model"))

        # Show message box only if we're not initializing
        if not self.is_initializing:
            messagebox.showinfo("Embedding Model Updated", 
                                f"Embedding model: {embedding_model}")

    def update_model_list(self, event=None):
        api_type = self.api_type_var.get()
        
        if api_type == "llama":
            models = self.server_manager.get_gguf_models()
        else:
            models = get_available_models(api_type)

        # Add 'None' option for Manager model
        manager_models = ['None'] + models

        self.model_menu['values'] = models
        self.supervisor_model_menu['values'] = models
        self.manager_model_menu['values'] = manager_models
        self.reranker_model_menu['values'] = models  # Add this line for re-ranker model

        if models:
            # Set worker model
            if api_type == "ollama":
                if settings.ollama_model in models:
                    self.model_var.set(settings.ollama_model)
                else:
                    self.model_var.set(models[0])
                    settings.update_setting("ollama_model", models[0])
            elif api_type == "llama" and self.server_manager.current_model in models:
                self.model_var.set(self.server_manager.current_model)
            elif api_type == "groq":
                if settings.groq_model in models:
                    self.model_var.set(settings.groq_model)
                else:
                    new_default = models[0]
                    self.model_var.set(new_default)
                    settings.update_setting("groq_model", new_default)
            elif api_type == "gemini":
                if settings.gemini_model in models:
                    self.model_var.set(settings.gemini_model)
                else:
                    new_default = models[0]
                    self.model_var.set(new_default)
                    settings.update_setting("gemini_model", new_default)
            else:
                self.model_var.set(models[0])

            # Set supervisor model (default to the same as worker model)
            self.supervisor_model_var.set(self.model_var.get())

            # Set manager model (use the default from settings if available)
            default_manager = settings.default_manager_model if settings.default_manager_model else 'None'
            if default_manager in manager_models:
                self.manager_model_var.set(default_manager)
            else:
                self.manager_model_var.set('None')

            # Set re-ranker model
            if settings.reranker_model in models:
                self.reranker_model_var.set(settings.reranker_model)
            else:
                self.reranker_model_var.set(models[0])
                settings.update_setting("reranker_model", models[0])

            if not self.is_initializing:
                self.update_model_setting()
        else:
            self.model_var.set("")
            self.supervisor_model_var.set("")
            self.manager_model_var.set('None')
            self.reranker_model_var.set("")
        
        if self.is_initializing:
            self.is_initializing = False
            self.update_model_setting(show_message=False)
        
        # Update the API type in settings
        settings.update_setting("api_type", api_type)
        
        print(success(f"Updated model list for API type: {api_type}"))
        print(success(f"Available models: {', '.join(models)}"))
        print(success(f"Selected worker model: {self.model_var.get()}"))
        print(success(f"Selected supervisor model: {self.supervisor_model_var.get()}"))
        print(success(f"Selected manager model: {self.manager_model_var.get()}"))
        print(success(f"Selected re-ranker model: {self.reranker_model_var.get()}"))

    def update_embedding_model_setting(self, event=None):
        embedding_class = self.embedding_class_var.get()
        embedding_model = self.embedding_model_var.get()
        
        settings.update_setting("default_embedding_class", embedding_class)
        settings.update_setting("default_embedding_model", embedding_model)
        
        print(info(f"Embedding class updated to: {embedding_class}"))
        print(info(f"Embedding model updated to: {embedding_model}"))
        
        # Update the EragAPI instance with the new embedding settings
        if hasattr(self, 'erag_api'):
            self.erag_api.embedding_class = embedding_class
            self.erag_api.embedding_model = embedding_model
            print(info("EragAPI updated with new embedding settings"))

    def create_agent_frame(self):
        agent_frame = tk.LabelFrame(self.main_tab, text="Model and Agent")
        agent_frame.pack(fill="x", padx=10, pady=5)

        talk2model_button = tk.Button(agent_frame, text="Talk2Model", command=self.run_talk2model)
        talk2model_button.pack(side="left", padx=5, pady=5)
        ToolTip(talk2model_button, "Start a conversation with the selected model")

        route_query_button = tk.Button(agent_frame, text="Route Query", command=self.run_route_query)
        route_query_button.pack(side="left", padx=5, pady=5)
        ToolTip(route_query_button, "Route a query to the appropriate system or model")

        # Add the new Create Self Knol button
        create_self_knol_button = tk.Button(agent_frame, text="Create Self Knol", command=self.create_self_knol)
        create_self_knol_button.pack(side="left", padx=5, pady=5)
        ToolTip(create_self_knol_button, "Create a knowledge artifact (Self Knol) using only the model's knowledge")

        # Add the new CodeEditor button
        code_editor_button = tk.Button(agent_frame, text="CodeEditor", command=self.run_code_editor)
        code_editor_button.pack(side="left", padx=5, pady=5)
        ToolTip(code_editor_button, "Open the AI-powered Code Editor")

    def run_code_editor(self):
        try:
            api_type = self.api_type_var.get()
            model = self.model_var.get()
            embedding_class = self.embedding_class_var.get()
            embedding_model = self.embedding_model_var.get()
            
            self.check_api_keys()  # Ensure API keys are set
            
            print(info(f"Initializing EragAPI with: API type: {api_type}, Model: {model}, Embedding class: {embedding_class}, Embedding model: {embedding_model}"))
            
            erag_api = create_erag_api(api_type, model, embedding_class, embedding_model)
            
            code_editor = CodeEditor(erag_api)
            code_editor.run()
            
            print(info(f"AI-powered Application Generator started with {api_type} API and {model} model."))
        except Exception as e:
            error_message = f"An error occurred while starting the AI-powered Application Generator: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)


    def run_talk2model(self):
        try:
            api_type = self.api_type_var.get()
            model = self.model_var.get()
            embedding_class = self.embedding_class_var.get()
            embedding_model = self.embedding_model_var.get()
            
            # Create and run the Talk2Model instance in a separate thread
            erag_api = create_erag_api(api_type, model, embedding_class, embedding_model)
            talk2model = Talk2Model(erag_api)
            threading.Thread(target=talk2model.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"Talk2Model started with {api_type} API and {model} model.\n"
                                        f"Embedding Class: {embedding_class}\n"
                                        f"Embedding Model: {embedding_model}\n"
                                        f"Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting Talk2Model: {str(e)}")



    def update_model_setting(self, event=None, show_message=True):
        api_type = self.api_type_var.get()
        model = self.model_var.get()
        embedding_class = self.embedding_class_var.get()
        embedding_model = self.embedding_model_var.get()
        reranker_model = self.reranker_model_var.get()
        
        if model:
            update_settings(settings, api_type, model)
            if api_type == "llama":
                if not self.server_manager.set_current_model(model):
                    messagebox.showwarning("Model Selection", f"Failed to set model: {model}")
                    return
            
            # Update the EragAPI instance with the new model and embedding settings
            self.erag_api = create_erag_api(api_type, model, embedding_class, embedding_model, reranker_model)
            
            print(info(f"EragAPI initialized with {api_type} backend for model: {model}"))
            print(info(f"Embedding class updated to: {embedding_class}"))
            print(info(f"Embedding model updated to: {embedding_model}"))
            print(info(f"Re-ranker model updated to: {reranker_model}"))
            
            if show_message:
                messagebox.showinfo("Model Selected", 
                                    f"Selected model: {model}\n"
                                    f"Using EragAPI with {api_type} backend\n"
                                    f"Embedding class: {embedding_class}\n"
                                    f"Embedding model: {embedding_model}\n"
                                    f"Re-ranker model: {reranker_model}")
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

        gen_a_button = tk.Button(rag_frame, text="Gen A", command=self.run_gen_a)
        gen_a_button.pack(side="left", padx=5, pady=5)
        ToolTip(gen_a_button, "Generate answers based on existing questions")

        # New Gen DSet button
        gen_dset_button = tk.Button(rag_frame, text="Gen DSet", command=self.run_gen_dset)
        gen_dset_button.pack(side="left", padx=5, pady=5)
        ToolTip(gen_dset_button, "Generate a dataset from Q&A pairs using LLM")

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
        # Create a main frame to hold the four columns
        main_frame = ttk.Frame(self.settings_tab)
        main_frame.grid(row=0, column=0, sticky="nsew")
        for i in range(4):
            main_frame.columnconfigure(i, weight=1)

        # Create four columns
        columns = [ttk.Frame(main_frame) for _ in range(4)]
        for i, column in enumerate(columns):
            column.grid(row=0, column=i, sticky="nsew", padx=5, pady=5)

        # Create frames for different setting categories
        upload_frame = self.create_labelframe(columns[0], "Upload Settings", 0)
        embeddings_frame = self.create_labelframe(columns[0], "Embeddings Settings", 1)
        graph_frame = self.create_labelframe(columns[0], "Graph Settings", 2)

        knol_frame = self.create_labelframe(columns[1], "Knol Creation Settings", 0)
        search_frame = self.create_labelframe(columns[1], "Search Settings", 1)
        rerank_frame = self.create_labelframe(columns[1], "Re-ranking Settings", 2)
        file_frame = self.create_labelframe(columns[1], "File Settings", 3)
        xdas_frame = self.create_labelframe(columns[1], "XDAs Settings", 4)

        web_sum_frame = self.create_labelframe(columns[2], "Web Sum Settings", 0)
        web_rag_frame = self.create_labelframe(columns[2], "Web RAG Settings", 1)
        summarization_frame = self.create_labelframe(columns[2], "Summarization Settings", 2)
        dataset_frame = self.create_labelframe(columns[2], "Dataset Generation Settings", 3)

        api_frame = self.create_labelframe(columns[3], "API Settings", 0)
        question_gen_frame = self.create_labelframe(columns[3], "Question Generation Settings", 1)
        talk2url_frame = self.create_labelframe(columns[3], "Talk2URL Settings", 2)
        github_frame = self.create_labelframe(columns[3], "GitHub Settings", 3)
        


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

        self.create_settings_fields(knol_frame, [
            ("Number of Questions", "num_questions"),
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

        self.create_settings_fields(summarization_frame, [
            ("Chunk Size", "summarization_chunk_size"),
            ("Summary Size", "summarization_summary_size"),
            ("Combining Number", "summarization_combining_number"),
            ("Final Chunk Size", "summarization_final_chunk_size"),
        ])

        self.create_settings_fields(question_gen_frame, [
            ("Initial Question Chunk Size", "initial_question_chunk_size"),
            ("Question Chunk Levels", "question_chunk_levels"),
            ("Excluded Question Levels", "excluded_question_levels"),
            ("Questions Per Chunk", "questions_per_chunk"),  # New field for questions per chunk
        ])

         # API Settings
        self.create_settings_fields(api_frame, [
            ("Default Embedding Class", "default_embedding_class"),
            ("Default Embedding Model", "default_embedding_model"),  # Change this line
            ("Sentence Transformer Model", "sentence_transformer_model"),
            ("Ollama Embedding Model", "ollama_embedding_model"),
            ("Default Ollama Model", "ollama_model"),
            ("Default Llama Model", "llama_model"),
            ("Default Groq Model", "groq_model"),
            ("Default Gemini Model", "gemini_model"),
            ("Default Manager Model", "default_manager_model"),
            ("Temperature", "temperature"),
            ("Max History Length", "max_history_length"),
            ("Conversation Context Size", "conversation_context_size"),
            ("Update Threshold", "update_threshold"),
        ])

        self.create_settings_fields(rerank_frame, [
            ("Re-rank Top K", "rerank_top_k"),
        ])

         # Add this new checkbox
        self.create_checkbox(xdas_frame, "Save Results to TXT", "save_results_to_txt", 
                             len(xdas_frame.grid_slaves()), 0)

        # Groq API Key field
        ttk.Label(api_frame, text="Groq API Key:").grid(row=len(api_frame.grid_slaves()), column=0, sticky="e", padx=5, pady=2)
        self.groq_api_key_var = tk.StringVar(value=self.groq_api_key)
        ttk.Entry(api_frame, textvariable=self.groq_api_key_var, show="*").grid(row=len(api_frame.grid_slaves())-1, column=1, sticky="w", padx=5, pady=2)

        # Gemini API Key field
        ttk.Label(api_frame, text="Gemini API Key:").grid(row=len(api_frame.grid_slaves()), column=0, sticky="e", padx=5, pady=2)
        self.gemini_api_key_var = tk.StringVar(value=self.gemini_api_key)
        ttk.Entry(api_frame, textvariable=self.gemini_api_key_var, show="*").grid(row=len(api_frame.grid_slaves())-1, column=1, sticky="w", padx=5, pady=2)


        # GitHub Token field
        ttk.Label(github_frame, text="GitHub Token:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.github_token_var = tk.StringVar(value=self.github_token)
        ttk.Entry(github_frame, textvariable=self.github_token_var, show="*").grid(row=0, column=1, sticky="w", padx=5, pady=2)


        # Create checkbox for talk2url_limit_content_size
        self.create_checkbox(talk2url_frame, "Limit Content Size", "talk2url_limit_content_size", 0, 0)
        
        # Create settings field for Content Size Per URL
        ttk.Label(talk2url_frame, text="Content Size Per URL").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        content_size_var = tk.StringVar(value=str(settings.talk2url_content_size_per_url))
        content_size_entry = ttk.Entry(talk2url_frame, textvariable=content_size_var)
        content_size_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        setattr(self, "talk2url_content_size_per_url_var", content_size_var)

        self.create_settings_fields(github_frame, [
            ("File Analysis Limit", "file_analysis_limit"),
        ])

        # Dataset fields
        ttk.Label(dataset_frame, text="Dataset Fields:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.dataset_fields_vars = {}
        for i, field in enumerate(["id", "question", "answer", "domain", "difficulty", "keywords", "language", "answer_type"]):
            var = tk.BooleanVar(value=field in settings.dataset_fields)
            ttk.Checkbutton(dataset_frame, text=field, variable=var).grid(row=i+1, column=0, sticky="w", padx=20)
            self.dataset_fields_vars[field] = var

        # Dataset output formats
        ttk.Label(dataset_frame, text="Output Formats:").grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.dataset_output_formats_vars = {}
        for i, format in enumerate(["jsonl", "csv", "parquet"]):
            var = tk.BooleanVar(value=format in settings.dataset_output_formats)
            ttk.Checkbutton(dataset_frame, text=format, variable=var).grid(row=i+1, column=1, sticky="w", padx=20)
            self.dataset_output_formats_vars[format] = var

        # Dataset output file
        ttk.Label(dataset_frame, text="Output File Base Name:").grid(row=9, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self.dataset_output_file_var = tk.StringVar(value=os.path.basename(settings.dataset_output_file))
        ttk.Entry(dataset_frame, textvariable=self.dataset_output_file_var).grid(row=10, column=0, columnspan=2, sticky="ew", padx=5, pady=2)



        # Add GitHub Token field separately
        ttk.Label(github_frame, text="GitHub Token:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.github_token_var = tk.StringVar(value=self.github_token)
        ttk.Entry(github_frame, textvariable=self.github_token_var, show="*").grid(row=1, column=1, sticky="w", padx=5, pady=2)


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
                
                # Special handling for default_manager_model
                if key == "default_manager_model":
                    value = None if value.lower() == 'none' else value
                
                settings.update_setting(key, value)

        
        # Update Groq API Key and GitHub Token in .env file
        self.update_env_file("GROQ_API_KEY", self.groq_api_key_var.get())
        self.update_env_file("GEMINI_API_KEY", self.gemini_api_key_var.get())
        self.update_env_file("GITHUB_TOKEN", self.github_token_var.get())

        # Apply dataset generation settings
        settings.dataset_fields = [field for field, var in self.dataset_fields_vars.items() if var.get()]
        settings.dataset_output_formats = [format for format, var in self.dataset_output_formats_vars.items() if var.get()]
        settings.dataset_output_file = os.path.join(settings.output_folder, self.dataset_output_file_var.get())

        settings.apply_settings()
        messagebox.showinfo("Settings", "Settings applied successfully")

        # Update the model list in the main GUI to reflect any changes
        self.update_model_list()

        settings.save_results_to_txt = self.save_results_to_txt_var.get()

        # Apply embedding model settings
        settings.update_setting("default_embedding_class", self.embedding_class_var.get())
        settings.update_setting("default_embedding_model", self.embedding_model_var.get())
       
    def update_env_file(self, key: str, value: str) -> None:
        """
        Update or add a key-value pair in the .env file and current environment.

        Args:
            key (str): The key to update or add.
            value (str): The value to set for the key.

        This method updates the .env file located in the project root directory.
        If the file doesn't exist, it will be created. If the key already exists
        in the file, its value will be updated. If the key doesn't exist, it will
        be added to the file. The method also updates the current session's
        environment variables.
        """
        env_path = os.path.join(self.project_root, '.env')
        set_key(env_path, key, value)
        os.environ[key] = value  # Update the environment variable in the current session

    def reset_settings(self):
        settings.reset_to_defaults()
        self.update_settings_display()
        messagebox.showinfo("Settings", "Settings reset to defaults")

    def update_settings_display(self):
        for key in dir(settings):
            if not key.startswith('_') and hasattr(self, f"{key}_var"):
                getattr(self, f"{key}_var").set(str(getattr(settings, key)))

                # Update dataset generation settings display
        for field, var in self.dataset_fields_vars.items():
            var.set(field in settings.dataset_fields)
        for format, var in self.dataset_output_formats_vars.items():
            var.set(format in settings.dataset_output_formats)
        self.dataset_output_file_var.set(os.path.basename(settings.dataset_output_file))

    def run_route_query(self):
        try:
            api_type = self.api_type_var.get()
            model = self.model_var.get()
            
            # Create the RouteQuery instance with EragAPI
            erag_api = EragAPI(api_type)
            route_query = RouteQuery(erag_api)
            
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

            # Check Groq API key if using Groq
            if api_type == "groq":
                self.check_groq_api_key()
                if not self.groq_api_key:
                    return  # Exit if no API key is provided

            erag_api = create_erag_api(api_type, model)

            # Apply settings before running the summarization
            self.apply_settings()

            # Run the summarization in a separate thread
            threading.Thread(target=self._create_sum_thread, args=(file_path, api_type, erag_api), daemon=True).start()

            messagebox.showinfo("Info", f"Summarization started for {os.path.basename(file_path)}. Check the console for progress.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the summarization process: {str(e)}")

    def _create_sum_thread(self, file_path, api_type, erag_api):
        try:
            result = run_create_sum(file_path, api_type, erag_api)
            print(result)
            messagebox.showinfo("Success", result)
        except Exception as e:
            error_message = f"An error occurred during summarization: {str(e)}"
            print(error_message)
            messagebox.showerror("Error", error_message)

    def create_knol(self):
        try:
            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            manager_model = self.manager_model_var.get()
            
            # Create separate EragAPI instances for worker, supervisor, and manager
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            manager_erag_api = create_erag_api(api_type, manager_model) if manager_model != 'None' else None
            
            # Create KnolCreator instance
            creator = KnolCreator(worker_erag_api, supervisor_erag_api, manager_erag_api)
            
            # Apply settings to KnolCreator
            settings.apply_settings()
            
            architecture_info = (
                "This Knol Creation module supports a Worker-Supervisor-Manager Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisor Model: {supervisor_model}\n"
                f"Manager Model: {manager_model}\n\n"
                "The Worker Model performs the initial knol creation and question answering, "
                "the Supervisor Model improves the knol and enhances the answers, "
                f"and the Manager Model {'reviews the final result, provides feedback, ' if manager_model != 'None' else 'is not used in this process.'}"
                f"{'and initiates further iterations if necessary.' if manager_model != 'None' else ''}"
            )
            
            messagebox.showinfo("Knol Creation Process Started", 
                                f"{architecture_info}\n\n"
                                f"Knol creation process started with {api_type} API.\n"
                                "Check the console for interaction and progress updates.")
            
            # Run the knol creator in a separate thread
            threading.Thread(target=creator.run_knol_creator, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the knol creation process: {str(e)}")


    def upload_multiple_files(self, file_type: str):
            try:
                files_added = self.file_processor.upload_multiple_files(file_type)
                if files_added > 0:
                    print(info(f"{files_added} {file_type} queued for processing. Check the console for progress. File content processed and appended to db.txt with table of contents in db_content.txt."))
                else:
                    print(info("No files selected for processing."))
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while queueing files for processing: {str(e)}")


    def execute_embeddings(self):
        try:
            # Ensure we're using the correct path from settings
            db_file_path = settings.db_file_path
            
            if not os.path.exists(db_file_path):
                messagebox.showwarning("Warning", f"{db_file_path} not found. Please upload some documents first.")
                return

            # Create the appropriate model based on settings
            if settings.default_embedding_class == "ollama":
                model = create_erag_api(settings.api_type, embedding_class=settings.default_embedding_class, embedding_model=settings.default_embedding_model)
            else:
                model = SentenceTransformer(settings.default_embedding_model)

            # Process db.txt
            self.db_embeddings, self.db_indexes, self.db_content = load_or_compute_embeddings(
                model, 
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

    def create_self_knol(self):
        try:
            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            manager_model = self.manager_model_var.get()
            
            # Create separate EragAPI instances for worker, supervisor, and manager
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            manager_erag_api = create_erag_api(api_type, manager_model) if manager_model != 'None' else None
            
            # Create SelfKnolCreator instance
            creator = SelfKnolCreator(worker_erag_api, supervisor_erag_api, manager_erag_api)
            
            # Apply settings to SelfKnolCreator
            settings.apply_settings()
            
            architecture_info = (
                "This Self Knol Creation module supports a Worker-Supervisor-Manager Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisor Model: {supervisor_model}\n"
                f"Manager Model: {manager_model}\n\n"
                "The Worker Model performs the initial knol creation, "
                "the Supervisor Model improves and expands the knol, "
                f"and the Manager Model {'reviews the final result and provides feedback.' if manager_model != 'None' else 'is not used in this process.'}"
            )
            
            messagebox.showinfo("Self Knol Creation Process Started", 
                                f"{architecture_info}\n\n"
                                f"Self Knol creation process started with {api_type} API.\n"
                                "Check the console for interaction and progress updates.")
            
            # Run the self knol creator in a separate thread
            threading.Thread(target=creator.run_self_knol_creator, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the self knol creation process: {str(e)}")


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


    def run_talk2urls(self):
        try:
            api_type = self.api_type_var.get()
            model = self.model_var.get()
            erag_api = create_erag_api(api_type, model)
            self.talk2url = Talk2URL(erag_api)
            
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
            model = self.model_var.get()
            github_token = os.getenv("GITHUB_TOKEN")  # Get token from environment
            erag_api = create_erag_api(api_type, model)
            self.talk2git = Talk2Git(erag_api, github_token)
            
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
            erag_api = create_erag_api(api_type, model)

            # Apply settings before running the question creation
            self.apply_settings()

            # Run the question creation in a separate thread
            threading.Thread(target=self._create_q_thread, args=(file_path, api_type, erag_api), daemon=True).start()

            messagebox.showinfo("Info", f"Question creation started for {os.path.basename(file_path)}. Check the console for progress.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the question creation process: {str(e)}")


    def _create_q_thread(self, file_path, api_type, erag_api):
        try:
            result = run_create_q(file_path, api_type, erag_api)
            print(result)
            messagebox.showinfo("Success", "Questions created successfully. Check the output file.")
        except Exception as e:
            error_message = f"An error occurred during question creation: {str(e)}"
            print(error_message)
            messagebox.showerror("Error", error_message)



    def run_web_sum(self):
        try:
            api_type = self.api_type_var.get()
            model = self.model_var.get()
            erag_api = create_erag_api(api_type, model)
            web_sum = WebSum(erag_api)
            
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
            model = self.model_var.get()
            erag_api = create_erag_api(api_type, model)
            self.web_rag = WebRAG(erag_api)
            
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
            model = self.model_var.get()
            erag_api = create_erag_api(api_type, model)

            # Apply settings silently
            self.apply_settings()

            # Run the answer generation in a separate thread
            threading.Thread(target=self._gen_a_thread, args=(questions_file, gen_method, api_type, erag_api), daemon=True).start()

            print(info(f"Answer generation started using {gen_method} method. Check the console for progress."))
        except Exception as e:
            print(error(f"An error occurred while starting the answer generation process: {str(e)}"))



    def _gen_a_thread(self, questions_file, gen_method, api_type, erag_api):
        try:
            from src.gen_a import run_gen_a
            result = run_gen_a(questions_file, gen_method, api_type, erag_api)
            print(result)
            messagebox.showinfo("Success", "Answers generated successfully. Check the output file.")
        except Exception as e:
            error_message = f"An error occurred during answer generation: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)


    def run_gen_dset(self):
        try:
            file_path = filedialog.askopenfilename(title="Select Q&A file",
                                                   filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if not file_path:
                messagebox.showwarning("Warning", "No file selected.")
                return

            api_type = self.api_type_var.get()
            model = self.model_var.get()

            # Apply settings before running the dataset generation
            self.apply_settings()

            # Run the dataset generation
            from src.gen_dset import run_gen_dset
            print(info("Starting dataset generation. Please check the console for progress updates."))
            result = run_gen_dset(file_path, api_type, model)
            print(result)
            messagebox.showinfo("Success", "Dataset generated successfully. Check the console for details and output file locations.")
        except Exception as e:
            error_message = f"An error occurred during dataset generation: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def _gen_dset_thread(self, file_path, api_type, erag_api):
        try:
            from src.gen_dset import run_gen_dset
            result = run_gen_dset(file_path, api_type, erag_api)
            print(result)
            messagebox.showinfo("Success", "Dataset generated successfully. Check the output files.")
        except Exception as e:
            error_message = f"An error occurred during dataset generation: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def upload_structured_data(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Structured Data File",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if not file_path:
                messagebox.showwarning("Warning", "No file selected.")
                return

            from src.sd_processing import process_structured_data
            result = process_structured_data(file_path)
            
            if result:
                messagebox.showinfo("Success", "Structured data processed and saved to SQLite database.")
            else:
                messagebox.showwarning("Warning", "Failed to process structured data.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the structured data: {str(e)}")

    def run_talk2sd(self):
        try:
            api_type = self.api_type_var.get()
            model = self.model_var.get()
            
            # Create EragAPI instance
            erag_api = create_erag_api(api_type, model)
            
            from src.talk2sd import Talk2SD
            talk2sd = Talk2SD(erag_api)
            
            # Apply settings to Talk2SD
            settings.apply_settings()
            
            # Run Talk2SD in a separate thread to keep the GUI responsive
            threading.Thread(target=talk2sd.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"Talk2SD system started with {api_type} API. Check the console for interaction.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while starting the Talk2SD system: {str(e)}")

    def choose_table(self, tables):
        # Filter out system tables
        user_tables = [table for table in tables if table.lower() not in ['information_schema', 'sqlite_master', 'sqlite_sequence', 'sqlite_stat1']]
        
        if not user_tables:
            messagebox.showwarning("Warning", "No user tables found in the database.")
            return None

        # Create a new top-level window
        table_window = tk.Toplevel(self.master)
        table_window.title("Choose a Table")
        table_window.geometry("300x200")

        # Create a listbox to display tables
        listbox = tk.Listbox(table_window)
        listbox.pack(expand=True, fill="both")

        # Populate the listbox with table names
        for table in user_tables:
            listbox.insert(tk.END, table)

        # Variable to store the selected table
        selected_table = tk.StringVar()

        # Function to set the selected table and close the window
        def on_select():
            selection = listbox.curselection()
            if selection:
                selected_table.set(user_tables[selection[0]])
                table_window.destroy()

        # Add a select button
        select_button = tk.Button(table_window, text="Select", command=on_select)
        select_button.pack()

        # Wait for the window to be closed
        self.master.wait_window(table_window)

        return selected_table.get()


    def run_xda(self):
        try:
            db_path = filedialog.askopenfilename(
                title="Select SQLite Database",
                filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                messagebox.showwarning("Warning", "No database selected.")
                return

            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            
            # Create separate EragAPI instances for worker and supervisor
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            
            # Create ExploratoryDataAnalysis instance with both APIs
            xda = ExploratoryDataAnalysis(worker_erag_api, supervisor_erag_api, db_path)
            
            # Apply settings to XDA
            settings.apply_settings()
            
            # Run XDA in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_xda_thread, args=(xda,), daemon=True).start()
            
            output_folder = os.path.join(os.path.dirname(db_path), "xda_output")
            
            # Create an informative message about the Worker-Supervisor Model architecture
            architecture_info = (
                "This XDA module supports a Worker Model and Supervisory Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisory Model: {supervisor_model}\n\n"
                "The Worker Model performs the initial analysis, while the Supervisory Model reviews and enhances the results, "
                "providing a more comprehensive and refined analysis."
            )
            
            messagebox.showinfo("XDA Process Started", 
                                f"{architecture_info}\n\n"
                                f"Exploratory Data Analysis started on the selected database {os.path.basename(db_path)}.\n"
                                f"You will be prompted to select a table in the console.\n"
                                f"Check the console for progress updates and AI interpretations.\n"
                                f"Results will be saved in {output_folder}")
        except Exception as e:
            error_message = f"An error occurred while starting the XDA process: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_xda_thread(self, xda):
        try:
            xda.run()
            print(success("Exploratory Data Analysis completed successfully."))
            
            # Generate PDF report
            pdf_path = xda.generate_pdf_report()
            
            print(success(f"PDF report generated successfully: {pdf_path}"))
            messagebox.showinfo("Success", f"Exploratory Data Analysis completed successfully.\n"
                                           f"PDF report generated: {pdf_path}")
        except Exception as e:
            error_message = f"An error occurred during Exploratory Data Analysis: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    
    def run_axda_b1(self):
        try:
            db_path = filedialog.askopenfilename(
                title="Select SQLite Database",
                filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                messagebox.showwarning("Warning", "No database selected.")
                return

            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            
            # Create separate EragAPI instances for worker and supervisor
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            
            # Create AdvancedExploratoryDataAnalysisB1 instance with both APIs
            axda_b1 = AdvancedExploratoryDataAnalysisB1(worker_erag_api, supervisor_erag_api, db_path)
            
            # Apply settings to A-XDA-B1
            settings.apply_settings()
            
            # Run A-XDA-B1 in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_axda_b1_thread, args=(axda_b1,), daemon=True).start()
            
            output_folder = os.path.join(os.path.dirname(db_path), "axda_b1_output")
            
            # Create an informative message about the Worker-Supervisor Model architecture
            architecture_info = (
                "This A-XDA-B1 module supports a Worker Model and Supervisory Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisory Model: {supervisor_model}\n\n"
                "The Worker Model performs the initial analysis, while the Supervisory Model reviews and enhances the results, "
                "providing a more comprehensive and refined analysis."
            )
            
            messagebox.showinfo("A-XDA-B1 Process Started", 
                                f"{architecture_info}\n\n"
                                f"Advanced Exploratory Data Analysis (Batch 1) started on the selected database {os.path.basename(db_path)}.\n"
                                f"Check the console for progress updates and AI interpretations.\n"
                                f"Results will be saved in {output_folder}")
        except Exception as e:
            error_message = f"An error occurred while starting the A-XDA-B1 process: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b1_thread(self, axda_b1):
        try:
            axda_b1.run()
            print(success("Advanced Exploratory Data Analysis (Batch 1) completed successfully."))
            
            # Generate PDF report
            pdf_path = axda_b1.generate_pdf_report()
            
            print(success(f"PDF report generated successfully: {pdf_path}"))
            messagebox.showinfo("Success", f"Advanced Exploratory Data Analysis (Batch 1) completed successfully.\n"
                                           f"PDF report generated: {pdf_path}")
        except Exception as e:
            error_message = f"An error occurred during Advanced Exploratory Data Analysis (Batch 1): {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b2(self):
        try:
            db_path = filedialog.askopenfilename(
                title="Select SQLite Database",
                filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                messagebox.showwarning("Warning", "No database selected.")
                return

            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            
            # Create separate EragAPI instances for worker and supervisor
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            
            # Create AdvancedExploratoryDataAnalysisB1 instance with both APIs
            axda_b2 = AdvancedExploratoryDataAnalysisB2(worker_erag_api, supervisor_erag_api, db_path)
            
            # Apply settings to A-XDA-B1
            settings.apply_settings()
            
            # Run A-XDA-B2 in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_axda_b2_thread, args=(axda_b2,), daemon=True).start()
            
            output_folder = os.path.join(os.path.dirname(db_path), "axda_b1_output")
            
            # Create an informative message about the Worker-Supervisor Model architecture
            architecture_info = (
                "This A-XDA-B2 module supports a Worker Model and Supervisory Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisory Model: {supervisor_model}\n\n"
                "The Worker Model performs the initial analysis, while the Supervisory Model reviews and enhances the results, "
                "providing a more comprehensive and refined analysis."
            )
            
            messagebox.showinfo("A-XDA-B2 Process Started", 
                                f"{architecture_info}\n\n"
                                f"Advanced Exploratory Data Analysis (Batch 2) started on the selected database {os.path.basename(db_path)}.\n"
                                f"Check the console for progress updates and AI interpretations.\n"
                                f"Results will be saved in {output_folder}")
        except Exception as e:
            error_message = f"An error occurred while starting the A-XDA-B2 process: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b2_thread(self, axda_b2):
        try:
            axda_b2.run()
            print(success("Advanced Exploratory Data Analysis (Batch 2) completed successfully."))
            
            # Generate PDF report
            pdf_path = axda_b2.generate_pdf_report()
            
            print(success(f"PDF report generated successfully: {pdf_path}"))
            messagebox.showinfo("Success", f"Advanced Exploratory Data Analysis (Batch 2) completed successfully.\n"
                                           f"PDF report generated: {pdf_path}")
        except Exception as e:
            error_message = f"An error occurred during Advanced Exploratory Data Analysis (Batch 2): {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)


    def run_ixda(self):
        try:
            db_path = filedialog.askopenfilename(
                title="Select SQLite Database",
                filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                messagebox.showwarning("Warning", "No database selected.")
                return

            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            
            # Create separate EragAPI instances for worker and supervisor
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            
            # Create InnovativeDataAnalysis instance with both APIs
            ixda = InnovativeDataAnalysis(worker_erag_api, supervisor_erag_api, db_path)
            
            # Apply settings to I-XDA
            settings.apply_settings()
            
            # Run I-XDA in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_ixda_thread, args=(ixda,), daemon=True).start()
            
            output_folder = os.path.join(os.path.dirname(db_path), "ixda_output")
            
            # Create an informative message about the Worker-Supervisor Model architecture
            architecture_info = (
                "This I-XDA module supports a Worker Model and Supervisory Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisory Model: {supervisor_model}\n\n"
                "The Worker Model performs the initial innovative analysis, while the Supervisory Model reviews and enhances the results, "
                "providing a more comprehensive and refined analysis using advanced techniques."
            )
            
            messagebox.showinfo("I-XDA Process Started", 
                                f"{architecture_info}\n\n"
                                f"Innovative Exploratory Data Analysis started on the selected database {os.path.basename(db_path)}.\n"
                                f"Check the console for progress updates and AI interpretations.\n"
                                f"Results will be saved in {output_folder}")
        except Exception as e:
            error_message = f"An error occurred while starting the I-XDA process: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_ixda_thread(self, ixda):
        try:
            ixda.run()
            print(success("Innovative Exploratory Data Analysis completed successfully."))
            
            # Generate PDF report
            pdf_path = ixda.generate_pdf_report()
            
            print(success(f"PDF report generated successfully: {pdf_path}"))
            messagebox.showinfo("Success", f"Innovative Exploratory Data Analysis completed successfully.\n"
                                           f"PDF report generated: {pdf_path}")
        except Exception as e:
            error_message = f"An error occurred during Innovative Exploratory Data Analysis: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)


    def run_axda_b3(self):
        try:
            db_path = filedialog.askopenfilename(
                title="Select SQLite Database",
                filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                messagebox.showwarning("Warning", "No database selected.")
                return

            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            
            # Create separate EragAPI instances for worker and supervisor
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            
            # Create AdvancedExploratoryDataAnalysisB3 instance with both APIs
            axda_b3 = AdvancedExploratoryDataAnalysisB3(worker_erag_api, supervisor_erag_api, db_path)
            
            # Apply settings to A-XDA-B3
            settings.apply_settings()
            
            # Run A-XDA-B3 in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_axda_b3_thread, args=(axda_b3,), daemon=True).start()
            
            output_folder = os.path.join(os.path.dirname(db_path), "axda_b3_output")
            
            # Create an informative message about the Worker-Supervisor Model architecture
            architecture_info = (
                "This A-XDA-B3 module supports a Worker Model and Supervisory Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisory Model: {supervisor_model}\n\n"
                "The Worker Model performs the initial analysis, while the Supervisory Model reviews and enhances the results, "
                "providing a more comprehensive and refined analysis."
            )
            
            messagebox.showinfo("A-XDA-B3 Process Started", 
                                f"{architecture_info}\n\n"
                                f"Advanced Exploratory Data Analysis (Batch 3) started on the selected database {os.path.basename(db_path)}.\n"
                                f"Check the console for progress updates and AI interpretations.\n"
                                f"Results will be saved in {output_folder}")
        except Exception as e:
            error_message = f"An error occurred while starting the A-XDA-B3 process: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b3_thread(self, axda_b3):
        try:
            axda_b3.run()
            print(success("Advanced Exploratory Data Analysis (Batch 3) completed successfully."))
            
            # Generate PDF report
            pdf_path = axda_b3.generate_pdf_report()
            
            print(success(f"PDF report generated successfully: {pdf_path}"))
            messagebox.showinfo("Success", f"Advanced Exploratory Data Analysis (Batch 3) completed successfully.\n"
                                           f"PDF report generated: {pdf_path}")
        except Exception as e:
            error_message = f"An error occurred during Advanced Exploratory Data Analysis (Batch 3): {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b4(self):
        try:
            db_path = filedialog.askopenfilename(
                title="Select SQLite Database",
                filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                messagebox.showwarning("Warning", "No database selected.")
                return

            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            
            # Create separate EragAPI instances for worker and supervisor
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            
            # Create AdvancedExploratoryDataAnalysisB4 instance with both APIs
            axda_b4 = AdvancedExploratoryDataAnalysisB4(worker_erag_api, supervisor_erag_api, db_path)
            
            # Apply settings to A-XDA-B4
            settings.apply_settings()
            
            # Run A-XDA-B4 in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_axda_b4_thread, args=(axda_b4,), daemon=True).start()
            
            output_folder = os.path.join(os.path.dirname(db_path), "axda_b4_output")
            
            # Create an informative message about the Worker-Supervisor Model architecture
            architecture_info = (
                "This A-XDA-B4 module supports a Worker Model and Supervisory Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisory Model: {supervisor_model}\n\n"
                "The Worker Model performs the initial analysis, while the Supervisory Model reviews and enhances the results, "
                "providing a more comprehensive and refined analysis."
            )
            
            messagebox.showinfo("A-XDA-B4 Process Started", 
                                f"{architecture_info}\n\n"
                                f"Advanced Exploratory Data Analysis (Batch 4) started on the selected database {os.path.basename(db_path)}.\n"
                                f"Check the console for progress updates and AI interpretations.\n"
                                f"Results will be saved in {output_folder}")
        except Exception as e:
            error_message = f"An error occurred while starting the A-XDA-B4 process: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b4_thread(self, axda_b4):
        try:
            axda_b4.run()
            print(success("Advanced Exploratory Data Analysis (Batch 4) completed successfully."))
            
            # Generate PDF report
            pdf_path = axda_b4.generate_pdf_report()
            
            print(success(f"PDF report generated successfully: {pdf_path}"))
            messagebox.showinfo("Success", f"Advanced Exploratory Data Analysis (Batch 4) completed successfully.\n"
                                           f"PDF report generated: {pdf_path}")
        except Exception as e:
            error_message = f"An error occurred during Advanced Exploratory Data Analysis (Batch 4): {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b5(self):
        try:
            db_path = filedialog.askopenfilename(
                title="Select SQLite Database",
                filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                messagebox.showwarning("Warning", "No database selected.")
                return

            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            
            # Create separate EragAPI instances for worker and supervisor
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            
            # Create AdvancedExploratoryDataAnalysisB5 instance with both APIs
            axda_b5 = AdvancedExploratoryDataAnalysisB5(worker_erag_api, supervisor_erag_api, db_path)
            
            # Apply settings to A-XDA-B5
            settings.apply_settings()
            
            # Run A-XDA-B5 in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_axda_b5_thread, args=(axda_b5,), daemon=True).start()
            
            output_folder = os.path.join(os.path.dirname(db_path), "axda_b5_output")
            
            # Create an informative message about the Worker-Supervisor Model architecture
            architecture_info = (
                "This A-XDA-B5 module supports a Worker Model and Supervisory Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisory Model: {supervisor_model}\n\n"
                "The Worker Model performs the initial analysis, while the Supervisory Model reviews and enhances the results, "
                "providing a more comprehensive and refined analysis."
            )
            
            messagebox.showinfo("A-XDA-B5 Process Started", 
                                f"{architecture_info}\n\n"
                                f"Advanced Exploratory Data Analysis (Batch 5) started on the selected database {os.path.basename(db_path)}.\n"
                                f"Check the console for progress updates and AI interpretations.\n"
                                f"Results will be saved in {output_folder}")
        except Exception as e:
            error_message = f"An error occurred while starting the A-XDA-B5 process: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b5_thread(self, axda_b5):
        try:
            axda_b5.run()
            print(success("Advanced Exploratory Data Analysis (Batch 5) completed successfully."))
            
            # Generate PDF report
            pdf_path = axda_b5.generate_pdf_report()
            
            print(success(f"PDF report generated successfully: {pdf_path}"))
            messagebox.showinfo("Success", f"Advanced Exploratory Data Analysis (Batch 5) completed successfully.\n"
                                           f"PDF report generated: {pdf_path}")
        except Exception as e:
            error_message = f"An error occurred during Advanced Exploratory Data Analysis (Batch 5): {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b6(self):
        try:
            db_path = filedialog.askopenfilename(
                title="Select SQLite Database",
                filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                messagebox.showwarning("Warning", "No database selected.")
                return

            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            
            # Create separate EragAPI instances for worker and supervisor
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            
            # Create AdvancedExploratoryDataAnalysisB6 instance with both APIs
            axda_b6 = AdvancedExploratoryDataAnalysisB6(worker_erag_api, supervisor_erag_api, db_path)
            
            # Apply settings to A-XDA-B6
            settings.apply_settings()
            
            # Run A-XDA-B6 in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_axda_b6_thread, args=(axda_b6,), daemon=True).start()
            
            output_folder = os.path.join(os.path.dirname(db_path), "axda_b6_output")
            
            # Create an informative message about the Worker-Supervisor Model architecture
            architecture_info = (
                "This A-XDA-B6 module supports a Worker Model and Supervisory Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisory Model: {supervisor_model}\n\n"
                "The Worker Model performs the initial analysis, while the Supervisory Model reviews and enhances the results, "
                "providing a more comprehensive and refined analysis."
            )
            
            messagebox.showinfo("A-XDA-B6 Process Started", 
                                f"{architecture_info}\n\n"
                                f"Advanced Exploratory Data Analysis (Batch 6) started on the selected database {os.path.basename(db_path)}.\n"
                                f"Check the console for progress updates and AI interpretations.\n"
                                f"Results will be saved in {output_folder}")
        except Exception as e:
            error_message = f"An error occurred while starting the A-XDA-B6 process: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b6_thread(self, axda_b6):
        try:
            axda_b6.run()
            print(success("Advanced Exploratory Data Analysis (Batch 6) completed successfully."))
            
            # Generate PDF report
            pdf_path = axda_b6.generate_pdf_report()
            
            print(success(f"PDF report generated successfully: {pdf_path}"))
            messagebox.showinfo("Success", f"Advanced Exploratory Data Analysis (Batch 6) completed successfully.\n"
                                           f"PDF report generated: {pdf_path}")
        except Exception as e:
            error_message = f"An error occurred during Advanced Exploratory Data Analysis (Batch 6): {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b7(self):
        try:
            db_path = filedialog.askopenfilename(
                title="Select SQLite Database",
                filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                messagebox.showwarning("Warning", "No database selected.")
                return

            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            
            # Create separate EragAPI instances for worker and supervisor
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            
            # Create AdvancedExploratoryDataAnalysisB7 instance with both APIs
            axda_b7 = AdvancedExploratoryDataAnalysisB7(worker_erag_api, supervisor_erag_api, db_path)
            
            # Apply settings to A-XDA-B7
            settings.apply_settings()
            
            # Run A-XDA-B7 in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_axda_b7_thread, args=(axda_b7,), daemon=True).start()
            
            output_folder = os.path.join(os.path.dirname(db_path), "axda_b7_output")
            
            # Create an informative message about the Worker-Supervisor Model architecture
            architecture_info = (
                "This A-XDA-B7 module supports a Worker Model and Supervisory Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisory Model: {supervisor_model}\n\n"
                "The Worker Model performs the initial analysis, while the Supervisory Model reviews and enhances the results, "
                "providing a more comprehensive and refined analysis."
            )
            
            messagebox.showinfo("A-XDA-B7 Process Started", 
                                f"{architecture_info}\n\n"
                                f"Advanced Exploratory Data Analysis (Batch 7) started on the selected database {os.path.basename(db_path)}.\n"
                                f"Check the console for progress updates and AI interpretations.\n"
                                f"Results will be saved in {output_folder}")
        except Exception as e:
            error_message = f"An error occurred while starting the A-XDA-B7 process: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_axda_b7_thread(self, axda_b7):
        try:
            axda_b7.run()
            print(success("Advanced Exploratory Data Analysis (Batch 7) completed successfully."))
            
            # Generate PDF report
            pdf_path = axda_b7.generate_pdf_report()
            
            print(success(f"PDF report generated successfully: {pdf_path}"))
            messagebox.showinfo("Success", f"Advanced Exploratory Data Analysis (Batch 7) completed successfully.\n"
                                           f"PDF report generated: {pdf_path}")
        except Exception as e:
            error_message = f"An error occurred during Advanced Exploratory Data Analysis (Batch 7): {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)


    def upload_financial_data(self):
        try:
            file_paths = filedialog.askopenfilenames(
                title="Select Financial Data Files",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if not file_paths:
                messagebox.showwarning("Warning", "No files selected.")
                return

            company_name = simpledialog.askstring("Company Name", "Enter the name of the company:")
            if not company_name:
                messagebox.showwarning("Warning", "Company name is required.")
                return

            from src.fn_processing import process_financial_data
            result = process_financial_data(file_paths, company_name)
            
            if result:
                messagebox.showinfo("Success", f"Financial data for {company_name} processed and saved to SQLite database.")
            else:
                messagebox.showwarning("Warning", f"Failed to process financial data for {company_name}.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the financial data: {str(e)}")

    def run_fxda(self):
        try:
            db_path = filedialog.askopenfilename(
                title="Select Financial SQLite Database",
                filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                messagebox.showwarning("Warning", "No database selected.")
                return

            api_type = self.api_type_var.get()
            worker_model = self.model_var.get()
            supervisor_model = self.supervisor_model_var.get()
            
            # Create separate EragAPI instances for worker and supervisor
            worker_erag_api = create_erag_api(api_type, worker_model)
            supervisor_erag_api = create_erag_api(api_type, supervisor_model)
            
            # Create FinancialExploratoryDataAnalysis instance with both APIs
            fxda = FinancialExploratoryDataAnalysis(worker_erag_api, supervisor_erag_api, db_path)
            
            # Apply settings to F-XDA
            settings.apply_settings()
            
            # Run F-XDA in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_fxda_thread, args=(fxda,), daemon=True).start()
            
            output_folder = os.path.join(os.path.dirname(db_path), "fxda_output")
            
            # Create an informative message about the Worker-Supervisor Model architecture
            architecture_info = (
                "This F-XDA module supports a Worker Model and Supervisory Model architecture:\n\n"
                f"Worker Model: {worker_model}\n"
                f"Supervisory Model: {supervisor_model}\n\n"
                "The Worker Model performs the initial financial analysis, while the Supervisory Model reviews and enhances the results, "
                "providing a more comprehensive and refined analysis using advanced financial techniques."
            )
            
            messagebox.showinfo("F-XDA Process Started", 
                                f"{architecture_info}\n\n"
                                f"Financial Exploratory Data Analysis started on the selected database {os.path.basename(db_path)}.\n"
                                f"Check the console for progress updates and AI interpretations.\n"
                                f"Results will be saved in {output_folder}")
        except Exception as e:
            error_message = f"An error occurred while starting the F-XDA process: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def run_fxda_thread(self, fxda):
        try:
            fxda.run()
            print(success("Financial Exploratory Data Analysis completed successfully."))
            
            # Generate PDF report
            pdf_path = fxda.generate_pdf_report()
            
            if pdf_path:
                print(success(f"PDF report generated successfully: {pdf_path}"))
                messagebox.showinfo("Success", f"Financial Exploratory Data Analysis completed successfully.\n"
                                               f"PDF report generated: {pdf_path}")
            else:
                print(error("Failed to generate PDF report."))
                messagebox.showwarning("Warning", "Financial Exploratory Data Analysis completed, but PDF report generation failed.")
        except Exception as e:
            error_message = f"An error occurred during Financial Exploratory Data Analysis: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def merge_structured_data(self):
        try:
            file1 = filedialog.askopenfilename(
                title="Select First Structured Data File",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if not file1:
                return

            file2 = filedialog.askopenfilename(
                title="Select Second Structured Data File",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if not file2:
                return

            # Ensure the output folder exists
            output_folder = os.path.join(self.project_root, "output")
            os.makedirs(output_folder, exist_ok=True)

            print("\nFiles selected. Starting merge process...")
            print("Please check the console for information and prompts.")
            result = merge_structured_data(file1, file2, output_folder)
            
            if not result:
                messagebox.showwarning("Warning", "Failed to merge structured data files. Check the console for details.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while merging the structured data files: {str(e)}")


            

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
            embedding_class = self.embedding_class_var.get()
            embedding_model = self.embedding_model_var.get()
            
            # Check API keys before proceeding
            self.check_api_keys()
            
            # Ensure the EragAPI instance is up-to-date
            self.erag_api = create_erag_api(api_type, model, embedding_class, embedding_model)
            
            print(info(f"EragAPI initialized with {self.erag_api.api_type} backend"))
            print(info(f"Model: {self.erag_api.model}"))
            print(info(f"Embedding class: {self.erag_api.embedding_class}"))
            print(info(f"Embedding model: {self.erag_api.embedding_model}"))
        
            # API-specific checks
            if api_type == "llama":
                if not self.server_manager.can_start_server():
                    raise Exception("Server cannot be started. Please check your settings.")
                if not self.server_manager.start_server():
                    raise Exception("Failed to start the llama.cpp server.")
            elif api_type == "groq" and not self.groq_api_key:
                raise Exception("Groq API Key is not set. Please set it in the settings.")
            elif api_type == "gemini" and not self.gemini_api_key:
                raise Exception("Gemini API Key is not set. Please set it in the settings.")
            
            # Create and run the RAG system
            from src.talk2doc import RAGSystem
            self.rag_system = RAGSystem(self.erag_api)
            settings.apply_settings()
            
            # Run the RAG system in a separate thread
            threading.Thread(target=self.rag_system.run, daemon=True).start()
            
            messagebox.showinfo("Info", f"System started with EragAPI (backed by {self.erag_api.api_type}) and model: {model}. Check the console for interaction.")
        except Exception as e:
            error_message = f"An error occurred while starting the system: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)

    def check_api_keys(self):
        if self.api_type_var.get() == "groq" and not self.groq_api_key:
            self.groq_api_key = simpledialog.askstring("Groq API Key", "Please enter your Groq API Key:", show='*')
            if self.groq_api_key:
                self.update_env_file("GROQ_API_KEY", self.groq_api_key)
                messagebox.showinfo("Success", "Groq API Key has been saved.")
            else:
                messagebox.showwarning("Warning", "Groq API Key is required to use the Groq API.")
        elif self.api_type_var.get() == "gemini" and not self.gemini_api_key:
            self.gemini_api_key = simpledialog.askstring("Gemini API Key", "Please enter your Gemini API Key:", show='*')
            if self.gemini_api_key:
                self.update_env_file("GEMINI_API_KEY", self.gemini_api_key)
                messagebox.showinfo("Success", "Gemini API Key has been saved.")
            else:
                messagebox.showwarning("Warning", "Gemini API Key is required to use the Gemini API.")


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
