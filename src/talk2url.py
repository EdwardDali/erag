import logging
from bs4 import BeautifulSoup
import requests
from src.settings import settings
from src.api_model import configure_api, LlamaClient
from src.color_scheme import Colors, colorize
import colorama
import os
import re
import unicodedata

# Initialize colorama
colorama.init(autoreset=True)

class Talk2URL:
    def __init__(self, api_type: str):
        self.api_type = api_type
        if api_type == "llama":
            self.client = LlamaClient()
        else:
            self.client = configure_api(api_type)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        })
        self.conversation_history = []
        self.current_urls = []
        self.url_contents = {}
        self.output_file = os.path.join(settings.output_folder, "talk2url_output.txt")

    def crawl_page(self, url):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, features="lxml")
            
            for script in soup(["script", "style"]):
                script.extract()
            text_content = ' '.join(soup.stripped_strings)
            
            # Remove non-printable characters and normalize Unicode
            text_content = ''.join(char for char in text_content if char.isprintable())
            text_content = unicodedata.normalize('NFKD', text_content).encode('ascii', 'ignore').decode('ascii')
            
            logging.info(f"Successfully crawled {url}")
            return text_content
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Error crawling {url}: {str(e)}")
            return None

    def process_urls(self, urls):
        for url in urls:
            content = self.crawl_page(url)
            if content:
                self.url_contents[url] = content
                print(colorize(f"Successfully processed content from {url}", Colors.SUCCESS))
            else:
                print(colorize(f"Failed to process content from {url}", Colors.ERROR))
        
        return f"Processed {len(self.url_contents)} URLs successfully."

    def generate_response(self, user_input):
        if settings.talk2url_limit_content_size:
            all_content = "\n\n".join([f"Content from {url}:\n{content[:settings.talk2url_content_size_per_url]}..." 
                                       for url, content in self.url_contents.items()])
        else:
            all_content = "\n\n".join([f"Content from {url}:\n{content}" 
                                       for url, content in self.url_contents.items()])

        system_message = """You are an AI assistant tasked with answering questions based on the provided web content. Follow these guidelines:

1. Use the given context to inform your answers.
2. If the context doesn't provide a suitable answer, rely on your general knowledge.
3. Structure your answer in a clear, organized manner.
4. Stay focused on the specific question asked.
5. If asked about specific data or numbers from the web content, prioritize providing that information.
6. Be concise but comprehensive.
7. If information from multiple URLs is relevant, mention which URL(s) you're referring to in your answer.
8. If you're not sure about something or if the information is not in the provided content, say so."""

        user_message = f"""Web Content:
{all_content}

User Question: {user_input}

Please provide a comprehensive and well-structured answer to the question based on the given web content. If the content doesn't contain relevant information, you can answer based on your general knowledge."""

        try:
            if self.api_type == "llama":
                response = self.client.chat([
                    {"role": "system", "content": system_message},
                    *self.conversation_history,
                    {"role": "user", "content": user_message}
                ], temperature=settings.temperature)
            else:
                response = self.client.chat.completions.create(
                    model=settings.ollama_model,
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


    def extract_urls(self, text):
        url_pattern = re.compile(r'https?://[^\s,]+')
        urls = url_pattern.findall(text)
        return [url.rstrip(',.') for url in urls]

    def get_urls_from_user(self):
        print(colorize("Enter the target URL(s) or a file path containing URLs.", Colors.INFO))
        print(colorize("You can enter multiple URLs or paste a list of URLs.", Colors.INFO))
        print(colorize("Type '\\' on a new line when finished.", Colors.INFO))

        user_input = []
        while True:
            line = input()
            if line.strip() == '\\':
                break
            user_input.append(line)

        full_input = '\n'.join(user_input)

        if os.path.isfile(full_input.strip()):
            try:
                with open(full_input.strip(), 'r') as file:
                    file_content = file.read()
                urls = self.extract_urls(file_content)
                print(f"{ANSIColor.CYAN.value}Loaded {len(urls)} URLs from file.{ANSIColor.RESET.value}")
                return urls
            except Exception as e:
                print(f"{ANSIColor.PINK.value}Error reading file: {str(e)}. Please try again.{ANSIColor.RESET.value}")
                return []

        urls = self.extract_urls(full_input)
        if not urls:
            print(f"{ANSIColor.PINK.value}No valid URLs found. Please try again.{ANSIColor.RESET.value}")
            return []

        print(f"{ANSIColor.CYAN.value}Found {len(urls)} URLs.{ANSIColor.RESET.value}")
        return urls

    def run(self):
        print(colorize("Welcome to Talk2URLs. Type 'exit' to quit, 'clear' to clear conversation history, or 'change urls' to update the target URLs.", Colors.INFO))
        print(colorize(f"All generated responses will be saved in: {self.output_file}", Colors.INFO))

        while True:
            if not self.current_urls:
                self.current_urls = self.get_urls_from_user()
                if not self.current_urls:
                    continue
                print(colorize("Processing URLs...", Colors.INFO))
                processing_result = self.process_urls(self.current_urls)
                print(colorize(f"{processing_result} You can now ask questions about the content.", Colors.SUCCESS))
                continue

            user_input = input(colorize("Enter your question or command: ", Colors.INFO)).strip()

            if user_input.lower() == 'exit':
                print(colorize("Thank you for using Talk2URLs. Goodbye!", Colors.SUCCESS))
                break
            elif user_input.lower() == 'clear':
                self.conversation_history.clear()
                self.current_urls.clear()
                self.url_contents.clear()
                print(colorize("Conversation history and current URLs cleared.", Colors.SUCCESS))
                continue
            elif user_input.lower() == 'change urls':
                self.current_urls.clear()
                self.url_contents.clear()
                print(colorize("Current URLs cleared. Please enter new URLs.", Colors.INFO))
                continue

            print(colorize("Generating response...", Colors.INFO))
            response = self.generate_response(user_input)
            print(colorize("\nResponse:", Colors.SUCCESS))
            print(response)

            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"Question: {user_input}\n\n")
                f.write(f"Response: {response}\n\n")
                f.write("-" * 50 + "\n\n")

            print(colorize(f"Response saved to {self.output_file}", Colors.SUCCESS))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        talk2url = Talk2URL(api_type)
        talk2url.run()
    else:
        print(colorize("Error: No API type provided.", Colors.ERROR))
        print(colorize("Usage: python src/talk2url.py <api_type>", Colors.INFO))
        print(colorize("Available API types: ollama, llama", Colors.INFO))
        sys.exit(1)
