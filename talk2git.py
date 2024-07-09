import logging
import os
import requests
from urllib.parse import urlparse
from settings import settings
from api_model import configure_api
from talk2doc import ANSIColor
import base64
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

class Talk2Git:
    def __init__(self, api_type: str):
        self.api_type = api_type
        self.client = configure_api(api_type)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Talk2Git/1.0",
            "Accept": "application/vnd.github.v3+json"
        })
        self.conversation_history = []
        self.repo_url = ""
        self.repo_contents = {}
        self.output_file = os.path.join(settings.output_folder, "talk2git_output.txt")
        self.github_api_url = "https://api.github.com"
        self.model = SentenceTransformer(settings.model_name)
        self.db_embeddings = None
        self.db_content = None

    def parse_github_url(self, url):
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub repository URL")
        owner, repo = path_parts[:2]
        return owner, repo

    def fetch_repo_contents(self, owner, repo, path=""):
        url = f"{self.github_api_url}/repos/{owner}/{repo}/contents/{path}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def fetch_file_content(self, url):
        response = self.session.get(url)
        response.raise_for_status()
        content = response.json()['content']
        decoded_content = base64.b64decode(content)
        try:
            return decoded_content.decode('utf-8')
        except UnicodeDecodeError:
            return f"[Binary content, size: {len(decoded_content)} bytes]"

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
                print(f"{ANSIColor.NEON_GREEN.value}Successfully processed {file_path}{ANSIColor.RESET.value}")
            elif item['type'] == 'dir':
                self.traverse_repo(owner, repo, item['path'])

    def create_repo_file(self):
        repo_file_path = os.path.join(settings.output_folder, "repo_contents.txt")
        with open(repo_file_path, "w", encoding="utf-8") as f:
            for file_path, content in self.repo_contents.items():
                f.write(f"File: {file_path}\n\n")
                f.write(content)
                f.write("\n\n" + "="*50 + "\n\n")
        print(f"{ANSIColor.NEON_GREEN.value}Repository contents saved to {repo_file_path}{ANSIColor.RESET.value}")

    def compute_embeddings(self):
        self.db_content = list(self.repo_contents.values())
        self.db_embeddings = self.model.encode(self.db_content, convert_to_tensor=True)
        embeddings_file = os.path.join(settings.output_folder, "repo_embeddings.pt")
        torch.save({"embeddings": self.db_embeddings, "content": self.db_content}, embeddings_file)
        print(f"{ANSIColor.NEON_GREEN.value}Embeddings saved to {embeddings_file}{ANSIColor.RESET.value}")

    def semantic_search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.db_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))
        return [(self.db_content[idx], score.item()) for idx, score in zip(top_results.indices, top_results.values)]

    def generate_response(self, user_input):
        relevant_content = self.semantic_search(user_input)
        context = "\n\n".join([f"Relevance {score:.2f}:\n{content}" for content, score in relevant_content])

        system_message = """You are an AI assistant tasked with answering questions about a GitHub repository. Follow these guidelines:
1. Use the provided repository content to inform your answers.
2. If asked about specific code or files, provide relevant code snippets or explain the code structure.
3. Structure your answer in a clear, organized manner.
4. Stay focused on the specific question asked.
5. Be concise but comprehensive.
6. If you're not sure about something or if the information is not in the provided content, say so."""

        user_message = f"""GitHub Repository Content:
{context}

User Question: {user_input}

Please provide a comprehensive and well-structured answer to the question based on the given repository content. Include relevant code snippets or explanations of code structure when appropriate."""

        try:
            response = self.client.chat.completions.create(
                model=settings.ollama_model if self.api_type == "ollama" else settings.llama_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.conversation_history,
                    {"role": "user", "content": user_message}
                ],
                temperature=settings.temperature
            ).choices[0].message.content

            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})

            return response
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I'm sorry, but I encountered an error while trying to answer your question."

    def run(self):
        print(f"{ANSIColor.YELLOW.value}Welcome to Talk2Git. Type 'exit' to quit, 'clear' to clear conversation history, or 'change repo' to analyze a different repository.{ANSIColor.RESET.value}")
        print(f"{ANSIColor.CYAN.value}All generated responses will be saved in: {self.output_file}{ANSIColor.RESET.value}")

        while True:
            if not self.repo_url:
                repo_url = input(f"{ANSIColor.YELLOW.value}Enter the GitHub repository URL: {ANSIColor.RESET.value}").strip()
                if not repo_url:
                    continue
                print(f"{ANSIColor.CYAN.value}Processing repository...{ANSIColor.RESET.value}")
                processing_result = self.process_repo(repo_url)
                print(f"{ANSIColor.NEON_GREEN.value}{processing_result} You can now ask questions about the repository.{ANSIColor.RESET.value}")
                continue

            user_input = input(f"{ANSIColor.YELLOW.value}Enter your question or command: {ANSIColor.RESET.value}").strip()

            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using Talk2Git. Goodbye!{ANSIColor.RESET.value}")
                break
            elif user_input.lower() == 'clear':
                self.conversation_history.clear()
                self.repo_url = ""
                self.repo_contents.clear()
                print(f"{ANSIColor.CYAN.value}Conversation history and current repository cleared.{ANSIColor.RESET.value}")
                continue
            elif user_input.lower() == 'change repo':
                self.repo_url = ""
                self.repo_contents.clear()
                self.db_embeddings = None
                self.db_content = None
                print(f"{ANSIColor.CYAN.value}Current repository cleared. Please enter a new repository URL.{ANSIColor.RESET.value}")
                continue

            print(f"{ANSIColor.CYAN.value}Generating response...{ANSIColor.RESET.value}")
            response = self.generate_response(user_input)
            print(f"\n{ANSIColor.NEON_GREEN.value}Response:{ANSIColor.RESET.value}\n{response}")

            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"Question: {user_input}\n\n")
                f.write(f"Response: {response}\n\n")
                f.write("-" * 50 + "\n\n")

            print(f"{ANSIColor.NEON_GREEN.value}Response saved to {self.output_file}{ANSIColor.RESET.value}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        talk2git = Talk2Git(api_type)
        talk2git.run()
    else:
        print("Error: No API type provided.")
        print("Usage: python talk2git.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
