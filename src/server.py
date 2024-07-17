import subprocess
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext
import threading
import glob

class ServerManager:
    def __init__(self):
        self.config_file = Path(__file__).parent.parent / "output" / "server_config.json"
        self.load_config()
        self.server_process = None
        self.output_window = None
        self.log_file = Path(__file__).parent.parent / "output" / "server_log.txt"

    def load_config(self):
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            self.enable_on_start = config.get('enable_on_start', False)
            self.server_executable = config.get('server_executable', '')
            self.model_folder = config.get('model_folder', '')
            self.current_model = config.get('current_model', '')
            self.additional_args = config.get('additional_args', '-c 8192')
            self.output_mode = config.get('output_mode', 'file')
        else:
            self.enable_on_start = False
            self.server_executable = ''
            self.model_folder = ''
            self.current_model = ''
            self.additional_args = '-c 8192'
            self.output_mode = 'file'

    def save_config(self):
        config = {
            'enable_on_start': self.enable_on_start,
            'server_executable': self.server_executable,
            'model_folder': self.model_folder,
            'current_model': self.current_model,
            'additional_args': self.additional_args,
            'output_mode': self.output_mode
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def get_gguf_models(self):
        if not self.model_folder:
            return []
        model_files = glob.glob(os.path.join(self.model_folder, "*.gguf"))
        return [os.path.basename(f) for f in model_files]

    def start_server(self):
        if self.server_process is None or self.server_process.poll() is not None:
            if not self.current_model:
                print("No model selected. Please select a model before starting the server.")
                return False
            
            model_path = os.path.join(self.model_folder, self.current_model)
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False

            command = [
                self.server_executable,
                '-m', model_path,
                *self.additional_args.split()
            ]
            try:
                if self.output_mode == 'file':
                    with open(self.log_file, 'w') as log:
                        self.server_process = subprocess.Popen(command, stdout=log, stderr=log)
                elif self.output_mode == 'window':
                    self.server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    self.create_output_window()
                    threading.Thread(target=self.read_output, daemon=True).start()
                print("Server started successfully.")
                return True
            except Exception as e:
                print(f"Failed to start server: {e}")
                return False
        return True  # Server was already running

    def stop_server(self):
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("Server stopped successfully.")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("Server forcefully terminated.")
            self.server_process = None
        if self.output_window:
            self.output_window.destroy()
            self.output_window = None

    def restart_server(self):
        self.stop_server()
        self.start_server()

    def create_output_window(self):
        if not self.output_window:
            self.output_window = tk.Toplevel()
            self.output_window.title("Server Output")
            self.output_text = scrolledtext.ScrolledText(self.output_window, wrap=tk.WORD)
            self.output_text.pack(expand=True, fill='both')

    def read_output(self):
        for line in self.server_process.stdout:
            if self.output_window:
                self.output_text.insert(tk.END, line)
                self.output_text.see(tk.END)
            else:
                break

    def set_output_mode(self, mode):
        if mode in ['file', 'window']:
            self.output_mode = mode
            self.save_config()

    def set_model_folder(self, folder):
        self.model_folder = folder
        self.save_config()

    def set_current_model(self, model_name):
        if model_name in self.get_gguf_models():
            self.current_model = model_name
            self.save_config()
            return True
        else:
            print(f"Model {model_name} not found in the models folder.")
            return False

    def can_start_server(self):
        return (self.server_executable and
                self.model_folder and
                self.get_gguf_models() and
                self.current_model in self.get_gguf_models())

# You can add any additional utility functions or classes here if needed

if __name__ == "__main__":
    # This block can be used for testing the ServerManager independently
    manager = ServerManager()
    print("Available GGUF models:", manager.get_gguf_models())
    if manager.can_start_server():
        print("Server can be started.")
    else:
        print("Server cannot be started. Please check the configuration.")
