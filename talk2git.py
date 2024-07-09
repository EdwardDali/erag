import logging
import os
import requests
from urllib.parse import urlparse
from settings import settings
from api_model import configure_api
from talk2doc import ANSIColor, SearchUtils
from sentence_transformers import SentenceTransformer, util
import torch
import base64
from typing import List, Dict, Tuple
import time
from collections import deque

class Talk2Git:
    def __init__(self, api_type: str, github_token: str = ""):
        self.api_type = api_type
        self.client = configure_api(api_type)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Talk2Git/1.0",
            "Accept": "application/vnd.github.v3+json"
        })
        if github_token:
            self.session.headers["Authorization"] = f"token {github_token}"
        self.conversation_history = []
        self.repo_url = ""
        self.repo_contents = {}
        self.github_api_url = "https://api.github.com"
        self.model = SentenceTransformer(settings.model_name)
        self.db_embeddings = None
        self.db_content = None
        self.project_name = ""
        self.conversation_context = deque(maxlen=settings.conversation_context_size * 2)
        self.search_utils = None
        self.file_index = {}  # New attribute to store file names without extensions

    def parse_github_url(self, url):
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub repository URL")
        owner, repo = path_parts[:2]
        self.project_name = repo
        return owner, repo

    def fetch_repo_contents(self, owner, repo, path=""):
        url = f"{self.github_api_url}/repos/{owner}/{repo}/contents/{path}"
        return self.make_github_request(url)

    def fetch_file_content(self, url):
        response = self.make_github_request(url)
        content = response['content']
        decoded_content = base64.b64decode(content)
        try:
            return decoded_content.decode('utf-8')
        except UnicodeDecodeError:
            return f"[Binary content, size: {len(decoded_content)} bytes]"

    def make_github_request(self, url):
        while True:
            response = self.session.get(url)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403 and 'rate limit exceeded' in response.text:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - int(time.time()), 0) + 1
                print(f"{ANSIColor.YELLOW.value}Rate limit exceeded. Waiting for {wait_time} seconds...{ANSIColor.RESET.value}")
                time.sleep(wait_time)
            else:
                response.raise_for_status()

    def process_repo(self, repo_url):
        self.repo_url = repo_url
        owner, repo = self.parse_github_url(repo_url)
        self.traverse_repo(owner, repo)
        self.create_repo_file()
        self.compute_embeddings()
        return f"Processed {len(self.repo_contents)} files from the repository."

    def traverse_repo(self, owner, repo, path=""):
        contents = self.fetch_repo_contents(owner, repo, path)
        for item in contents:
            if item['type'] == 'file':
                file_content = self.fetch_file_content(item['url'])
                file_path = item['path']
                self.repo_contents[file_path] = file_content
                # Add file to index without extension
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                self.file_index[file_name] = file_path
                print(f"{ANSIColor.NEON_GREEN.value}Successfully processed {file_path}{ANSIColor.RESET.value}")
            elif item['type'] == 'dir':
                self.traverse_repo(owner, repo, item['path'])

    def create_repo_file(self):
        repo_file_path = os.path.join(settings.output_folder, f"{self.project_name}_contents.txt")
        with open(repo_file_path, "w", encoding="utf-8") as f:
            for file_path, content in self.repo_contents.items():
                f.write(f"File: {file_path}\n\n")
                f.write(content)
                f.write("\n\n" + "="*50 + "\n\n")
        print(f"{ANSIColor.NEON_GREEN.value}Repository contents saved to {repo_file_path}{ANSIColor.RESET.value}")

    def compute_embeddings(self):
        self.db_content = list(self.repo_contents.values())
        self.db_embeddings = self.model.encode(self.db_content, convert_to_tensor=True)
        embeddings_file = os.path.join(settings.output_folder, f"{self.project_name}_embeddings.pt")
        torch.save({"embeddings": self.db_embeddings, "content": self.db_content}, embeddings_file)
        print(f"{ANSIColor.NEON_GREEN.value}Embeddings saved to {embeddings_file}{ANSIColor.RESET.value}")
        
        # Initialize SearchUtils after computing embeddings
        self.search_utils = SearchUtils(self.model, self.db_embeddings, self.db_content, None)

    def generate_response(self, user_input: str) -> str:
        # Check if the user is asking about a specific file
        for file_name in self.file_index:
            if file_name.lower() in user_input.lower():
                user_input = user_input.replace(file_name, self.file_index[file_name])
        lexical_context, semantic_context, graph_context, text_context = self.search_utils.get_relevant_context(user_input, list(self.conversation_context))
        
        lexical_str = "\n".join(lexical_context)
        semantic_str = "\n".join(semantic_context)
        graph_str = "\n".join(graph_context)
        text_str = "\n".join(text_context)

        combined_context = f"""Conversation Context:\n{' '.join(self.conversation_context)}

Lexical Search Results:
{lexical_str}

Semantic Search Results:
{semantic_str}

Text Search Results:
{text_str}"""

        system_message = """You are an AI assistant tasked with answering questions about a GitHub repository. Follow these guidelines:
1. Use the provided repository content to inform your answers.
2. If asked about specific code or files, provide relevant code snippets or explain the code structure.
3. Structure your answer in a clear, organized manner.
4. Stay focused on the specific question asked.
5. Be concise but comprehensive.
6. If you're not sure about something or if the information is not in the provided content, say so."""

        messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history,
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {user_input}\n\nPlease prioritize the Conversation Context when answering, followed by the most relevant information from either the lexical, semantic, or text search results. If none of the provided context is relevant, you can answer based on your general knowledge about software development and GitHub repositories."}
        ]

        try:
            response = self.client.chat.completions.create(
                model=settings.ollama_model if self.api_type == "ollama" else settings.llama_model,
                messages=messages,
                temperature=settings.temperature
            ).choices[0].message.content

            return response
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I'm sorry, but I encountered an error while trying to answer your question."

    def update_conversation_context(self, user_input: str, assistant_response: str):
        self.conversation_context.append(user_input)
        self.conversation_context.append(assistant_response)

    def static_code_analysis(self):
        analysis_results = []
        for file_path, content in self.repo_contents.items():
            print(f"{ANSIColor.CYAN.value}Analyzing {file_path}...{ANSIColor.RESET.value}")
            
            # Prepare the prompt for the LLM
            prompt = f"""Perform a static code analysis on the following file:

File: {file_path}

Content:
{content[:1000]}  # Limit content to first 1000 characters to avoid token limits

Please provide a brief analysis covering the following aspects:
1. Code structure and organization
2. Potential bugs or issues
3. Adherence to best practices
4. Suggestions for improvements

Your analysis should be concise but informative."""

            try:
                response = self.client.chat.completions.create(
                    model=settings.ollama_model if self.api_type == "ollama" else settings.llama_model,
                    messages=[
                        {"role": "system", "content": "You are an expert code reviewer performing static code analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2  # Lower temperature for more focused responses
                ).choices[0].message.content

                analysis_results.append(f"Analysis for {file_path}:\n\n{response}\n\n{'='*50}\n")
                print(f"{ANSIColor.NEON_GREEN.value}Analysis complete for {file_path}{ANSIColor.RESET.value}")
            except Exception as e:
                logging.error(f"Error analyzing {file_path}: {str(e)}")
                analysis_results.append(f"Error analyzing {file_path}: {str(e)}\n\n{'='*50}\n")

        # Save analysis results
        analysis_file = os.path.join(settings.output_folder, f"{self.project_name}_static_analysis.txt")
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write("\n".join(analysis_results))
        
        print(f"{ANSIColor.NEON_GREEN.value}Static code analysis completed. Results saved to {analysis_file}{ANSIColor.RESET.value}")

    def run(self):
        print(f"{ANSIColor.YELLOW.value}Welcome to Talk2Git. Type 'exit' to quit, 'clear' to clear conversation history, 'change repo' to analyze a different repository, or 'analyze' to perform static code analysis.{ANSIColor.RESET.value}")

        while True:
            if not self.repo_url:
                repo_url = input(f"{ANSIColor.YELLOW.value}Enter the GitHub repository URL: {ANSIColor.RESET.value}").strip()
                if not repo_url:
                    continue
                print(f"{ANSIColor.CYAN.value}Processing repository...{ANSIColor.RESET.value}")
                processing_result = self.process_repo(repo_url)
                print(f"{ANSIColor.NEON_GREEN.value}{processing_result} You can now ask questions about the repository or perform static code analysis.{ANSIColor.RESET.value}")
                continue

            user_input = input(f"{ANSIColor.YELLOW.value}Enter your question or command: {ANSIColor.RESET.value}").strip()

            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using Talk2Git. Goodbye!{ANSIColor.RESET.value}")
                break
            elif user_input.lower() == 'clear':
                self.conversation_history.clear()
                self.conversation_context.clear()
                print(f"{ANSIColor.CYAN.value}Conversation history and current repository cleared.{ANSIColor.RESET.value}")
                continue
            elif user_input.lower() == 'change repo':
                self.repo_url = ""
                self.repo_contents.clear()
                self.db_embeddings = None
                self.db_content = None
                self.conversation_history.clear()
                self.conversation_context.clear()
                self.search_utils = None
                self.file_index.clear()
                print(f"{ANSIColor.CYAN.value}Current repository cleared. Please enter a new repository URL.{ANSIColor.RESET.value}")
                continue
            elif user_input.lower() == 'analyze':
                self.static_code_analysis()
                continue

            print(f"{ANSIColor.CYAN.value}Generating response...{ANSIColor.RESET.value}")
            response = self.generate_response(user_input)
            print(f"\n{ANSIColor.NEON_GREEN.value}Response:{ANSIColor.RESET.value}\n{response}")

            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            self.update_conversation_context(user_input, response)

            output_file = os.path.join(settings.output_folder, f"{self.project_name}_talk2git_output.txt")
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"Question: {user_input}\n\n")
                f.write(f"Response: {response}\n\n")
                f.write("-" * 50 + "\n\n")

            print(f"{ANSIColor.NEON_GREEN.value}Response saved to {output_file}{ANSIColor.RESET.value}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        github_token = settings.github_token  # Get the GitHub token from settings
        talk2git = Talk2Git(api_type, github_token)
        talk2git.run()
    else:
        print("Error: No API type provided.")
        print("Usage: python talk2git.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
