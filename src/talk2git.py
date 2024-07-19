import logging
import os
import requests
from urllib.parse import urlparse
from src.settings import settings
from src.api_model import EragAPI, create_erag_api
from src.look_and_feel import success, info, warning, error
import base64
import time
from dotenv import load_dotenv
from src.print_pdf import PDFReportGenerator

class Talk2Git:
    def __init__(self, erag_api: EragAPI, github_token: str = ""):
        self.erag_api = erag_api
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Talk2Git/1.0",
            "Accept": "application/vnd.github.v3+json"
        })
        
        # Load environment variables
        load_dotenv()
        
        # Use GitHub token from .env file if not provided
        if not github_token:
            github_token = os.getenv("GITHUB_TOKEN")
        
        if github_token:
            self.session.headers["Authorization"] = f"token {github_token}"
        else:
            print(warning("GitHub token not found. Some operations may be limited."))
        
        self.repo_url = ""
        self.repo_contents = {}
        self.github_api_url = "https://api.github.com"
        self.project_name = ""
        self.pdf_content = []
        self.findings = []
        self.image_paths = []
        self.project_summary = ""
        self.pdf_generator = PDFReportGenerator(settings.output_folder, erag_api.model)

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
                print(warning(f"Rate limit exceeded. Waiting for {wait_time} seconds..."))
                time.sleep(wait_time)
            else:
                response.raise_for_status()

    def process_repo(self, repo_url):
        self.repo_url = repo_url
        owner, repo = self.parse_github_url(repo_url)
        self.traverse_repo(owner, repo)
        self.create_repo_file()
        return f"Processed {len(self.repo_contents)} files from the repository."

    def traverse_repo(self, owner, repo, path=""):
        contents = self.fetch_repo_contents(owner, repo, path)
        for item in contents:
            if item['type'] == 'file':
                file_content = self.fetch_file_content(item['url'])
                file_path = item['path']
                self.repo_contents[file_path] = file_content
                print(success(f"Successfully processed {file_path}"))
            elif item['type'] == 'dir':
                self.traverse_repo(owner, repo, item['path'])

    def create_repo_file(self):
        repo_file_path = os.path.join(settings.output_folder, f"{self.project_name}_contents.txt")
        with open(repo_file_path, "w", encoding="utf-8") as f:
            for file_path, content in self.repo_contents.items():
                f.write(f"File: {file_path}\n\n")
                f.write(content)
                f.write("\n\n" + "="*50 + "\n\n")
        print(success(f"Repository contents saved to {repo_file_path}"))

    def save_results(self, analysis_type: str, text_content: str):
        # Save text file
        text_file = os.path.join(settings.output_folder, f"{self.project_name}_{analysis_type}.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(success(f"{analysis_type} results saved to {text_file}"))

        # Generate and save PDF report
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.project_summary,
            self.findings,
            self.pdf_content,
            self.image_paths,
            f"{self.project_name}_{analysis_type}"  # Pass the desired PDF filename (without extension)
        )
        if pdf_file:
            print(success(f"PDF report for {analysis_type} generated: {pdf_file}"))
        else:
            print(error(f"Failed to generate PDF report for {analysis_type}"))

        # Clear the lists for the next analysis
        self.pdf_content.clear()
        self.findings.clear()
        self.image_paths.clear()

    def run_all_analyses(self):
        print(info("Running all analyses..."))
        self.static_code_analysis()
        self.summarize_project()
        self.analyze_dependencies()
        self.detect_code_smells()
        
        # Combine all results for a final "All Analyses" report
        all_text_content = ""
        all_pdf_content = []
        all_findings = []
        
        for analysis_type in ["static_code_analysis", "project_summary", "dependency_analysis", "code_smell_detection"]:
            text_file = os.path.join(settings.output_folder, f"{self.project_name}_{analysis_type}.txt")
            if os.path.exists(text_file):
                with open(text_file, "r", encoding="utf-8") as f:
                    all_text_content += f"\n\n{'='*50}\n\n{analysis_type.upper()}\n\n"
                    all_text_content += f.read()
            
            pdf_file = os.path.join(settings.output_folder, f"{self.project_name}_{analysis_type}.pdf")
            if os.path.exists(pdf_file):
                all_pdf_content.append((f"{analysis_type.capitalize()} Report", "", f"See detailed report: {pdf_file}"))
            
            all_findings.append(f"Completed {analysis_type}")
        
        self.save_results("all_analyses", all_text_content)
        print(success("All analyses completed and combined report generated."))

    def static_code_analysis(self):
        analysis_results = []
        for file_path, content in self.repo_contents.items():
            print(info(f"Analyzing {file_path}..."))
            
            prompt = f"""Perform a static code analysis on the following file:

File: {file_path}

Content:
{content[:settings.file_analysis_limit]}  # Use the file_analysis_limit setting

Please provide a brief analysis covering the following aspects:
1. Code structure and organization
2. Potential bugs or issues
3. Adherence to best practices
4. Suggestions for improvements
5. Any security concerns

Your analysis should be concise but informative."""

            try:
                messages = [
                    {"role": "system", "content": "You are an expert code reviewer performing static code analysis."},
                    {"role": "user", "content": prompt}
                ]
                response = self.erag_api.chat(messages, temperature=0.2)

                analysis_results.append(f"Analysis for {file_path}:\n\n{response}\n\n{'='*50}\n")
                print(success(f"Analysis complete for {file_path}"))
                
                # Add to pdf_content and findings
                self.pdf_content.append((f"Static Code Analysis for {file_path}", "", response))
                self.findings.append(f"Static Code Analysis: {file_path}")
            except Exception as e:
                logging.error(f"Error analyzing {file_path}: {str(e)}")
                analysis_results.append(f"Error analyzing {file_path}: {str(e)}\n\n{'='*50}\n")

        # Save results
        self.save_results("static_code_analysis", "\n".join(analysis_results))

    def summarize_project(self):
        file_summaries = {}
        for file_path, content in self.repo_contents.items():
            print(info(f"Summarizing {file_path}..."))
            
            prompt = f"""Summarize the purpose and main functionality of the following file:

File: {file_path}

Content:
{content[:settings.file_analysis_limit]}  # Use the file_analysis_limit setting

Please provide a concise summary (2-3 sentences) describing the file's main purpose and functionality."""

            try:
                messages = [
                    {"role": "system", "content": "You are an expert programmer summarizing code files."},
                    {"role": "user", "content": prompt}
                ]
                response = self.erag_api.chat(messages, temperature=0.2)

                file_summaries[file_path] = response
                print(success(f"Summary complete for {file_path}"))
                
                # Add to pdf_content
                self.pdf_content.append((f"Summary for {file_path}", "", response))
            except Exception as e:
                logging.error(f"Error summarizing {file_path}: {str(e)}")
                file_summaries[file_path] = f"Error summarizing file: {str(e)}"

        # Create overall project summary
        project_summary_prompt = f"""Based on the following file summaries, provide an overall summary of the project:

{chr(10).join([f"{path}: {summary}" for path, summary in file_summaries.items()])}

Please provide a concise summary (3-5 sentences) describing the overall purpose and functionality of the project."""

        try:
            messages = [
                {"role": "system", "content": "You are an expert programmer summarizing software projects."},
                {"role": "user", "content": project_summary_prompt}
            ]
            self.project_summary = self.erag_api.chat(messages, temperature=0.2)
        except Exception as e:
            logging.error(f"Error creating project summary: {str(e)}")
            self.project_summary = f"Error creating project summary: {str(e)}"

        # Prepare text content
        text_content = f"Project Summary:\n{self.project_summary}\n\n{'='*50}\n\nFile Summaries:\n\n"
        for file_path, summary in file_summaries.items():
            text_content += f"File: {file_path}\nSummary: {summary}\n\n"

        # Save results
        self.save_results("project_summary", text_content)


    def analyze_dependencies(self):
        dependency_files = [file for file in self.repo_contents.keys() if file.endswith(('requirements.txt', 'package.json', 'pom.xml'))]
        
        if not dependency_files:
            message = "No dependency files found in the repository."
            print(warning(message))
            self.save_results("dependency_analysis", message)
            return

        analysis_results = []
        for file in dependency_files:
            content = self.repo_contents[file][:settings.file_analysis_limit]
            print(info(f"Analyzing dependencies in {file}..."))
            
            prompt = f"""Analyze the dependencies in the following file:

File: {file}

Content:
{content}

Please provide a brief analysis covering the following aspects:
1. List of main dependencies and their versions
2. Potential outdated dependencies
3. Possible security vulnerabilities (based on known common vulnerabilities)
4. Suggestions for dependency updates or replacements

Your analysis should be concise but informative."""

            try:
                messages = [
                    {"role": "system", "content": "You are an expert in software dependencies and security analysis."},
                    {"role": "user", "content": prompt}
                ]
                response = self.erag_api.chat(messages, temperature=0.2)

                analysis_results.append(f"Analysis for {file}:\n\n{response}\n\n{'='*50}\n")
                print(success(f"Analysis complete for {file}"))
                
                # Add to pdf_content and findings
                self.pdf_content.append((f"Dependency Analysis for {file}", "", response))
                self.findings.append(f"Dependency Analysis: {file}")
            except Exception as e:
                logging.error(f"Error analyzing dependencies in {file}: {str(e)}")
                analysis_results.append(f"Error analyzing dependencies in {file}: {str(e)}\n\n{'='*50}\n")

        # Save results
        self.save_results("dependency_analysis", "\n".join(analysis_results))

    def detect_code_smells(self):
        code_files = [file for file in self.repo_contents.keys() if file.endswith(('.py', '.js', '.java', '.cpp', '.c', '.h', '.cs'))]
        
        if not code_files:
            message = "No supported code files found in the repository."
            print(warning(message))
            self.save_results("code_smell_detection", message)
            return

        analysis_results = []
        for file in code_files:
            content = self.repo_contents[file][:settings.file_analysis_limit]
            print(info(f"Detecting code smells in {file}..."))
            
            prompt = f"""Analyze the following code for potential code smells:

File: {file}

Content:
{content}

Please provide a brief analysis covering the following aspects:
1. Identify any common code smells (e.g., long methods, large classes, duplicate code)
2. Highlight areas of the code that might be difficult to understand or maintain
3. Suggest potential refactoring opportunities
4. Comment on the overall code quality and structure

Your analysis should be concise but informative."""

            try:
                messages = [
                    {"role": "system", "content": "You are an expert code reviewer specializing in identifying code smells and suggesting improvements."},
                    {"role": "user", "content": prompt}
                ]
                response = self.erag_api.chat(messages, temperature=0.2)

                analysis_results.append(f"Analysis for {file}:\n\n{response}\n\n{'='*50}\n")
                print(success(f"Analysis complete for {file}"))
                
                # Add to pdf_content and findings
                self.pdf_content.append((f"Code Smell Detection for {file}", "", response))
                self.findings.append(f"Code Smell Detection: {file}")
            except Exception as e:
                logging.error(f"Error detecting code smells in {file}: {str(e)}")
                analysis_results.append(f"Error detecting code smells in {file}: {str(e)}\n\n{'='*50}\n")

        # Save results
        self.save_results("code_smell_detection", "\n".join(analysis_results))


    def run(self):
        print(info("Welcome to Talk2Git."))

        while True:
            if not self.repo_url:
                repo_url = input(info("Enter the GitHub repository URL: ")).strip()
                if not repo_url:
                    continue
                print(info("Processing repository..."))
                processing_result = self.process_repo(repo_url)
                print(success(processing_result))

            self.display_menu()
            choice = input(info("Enter your choice (1-7): ")).strip()

            if choice == '1':
                self.static_code_analysis()
            elif choice == '2':
                self.summarize_project()
            elif choice == '3':
                self.analyze_dependencies()
            elif choice == '4':
                self.detect_code_smells()
            elif choice == '5':
                self.run_all_analyses()
            elif choice == '6':
                self.repo_url = ""
                self.repo_contents.clear()
                print(info("Current repository cleared. Please enter a new repository URL."))
            elif choice == '7':
                print(success("Thank you for using Talk2Git. Goodbye!"))
                break
            else:
                print(error("Invalid choice. Please enter a number between 1 and 7."))

    def display_menu(self):
        print(info("\nTalk2Git Menu:"))
        print("1. Analyze repository")
        print("2. Summarize project")
        print("3. Analyze dependencies")
        print("4. Detect code smells")
        print("5. All (Run all analyses)")
        print("6. Change repository")
        print("7. Exit")

def main(api_type: str):
    erag_api = create_erag_api(api_type)
    github_token = settings.github_token
    talk2git = Talk2Git(erag_api, github_token)
    talk2git.run()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        main(api_type)
    else:
        print(error("Error: No API type provided."))
        print(warning("Usage: python src/talk2git.py <api_type>"))
        print(info("Available API types: ollama, llama"))
        sys.exit(1)
