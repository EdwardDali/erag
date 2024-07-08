import logging
from bs4 import BeautifulSoup
import requests
from settings import settings
from api_model import configure_api
from talk2doc import ANSIColor
import os

class Talk2URL:
    def __init__(self, api_type: str):
        self.api_type = api_type
        self.client = configure_api(api_type)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        })
        self.conversation_history = []
        self.current_urls = []
        self.output_file = os.path.join(settings.output_folder, "talk2url_output.txt")

    def crawl_page(self, url):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, features="lxml")
            
            for script in soup(["script", "style"]):
                script.extract()
            text_content = ' '.join(soup.stripped_strings)
            
            logging.info(f"Successfully crawled {url}")
            return text_content
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Error crawling {url}: {str(e)}")
            return None

    def process_urls(self, urls):
        all_content = []
        for url in urls:
            content = self.crawl_page(url)
            if content:
                all_content.append(f"Content from {url}:\n{content}\n\n")
                print(f"{ANSIColor.NEON_GREEN.value}Successfully processed content from {url}{ANSIColor.RESET.value}")
            else:
                print(f"{ANSIColor.PINK.value}Failed to process content from {url}{ANSIColor.RESET.value}")
        return "\n".join(all_content)

    def generate_response(self, user_input, context):
        system_message = """You are an AI assistant tasked with answering questions based on the provided web content. Follow these guidelines:

1. Use the given context to inform your answers.
2. If the context doesn't provide a suitable answer, rely on your general knowledge.
3. Structure your answer in a clear, organized manner.
4. Stay focused on the specific question asked.
5. If asked about specific data or numbers from the web content, prioritize providing that information.
6. Be concise but comprehensive."""

        user_message = f"""Web Content:
{context}

User Question: {user_input}

Please provide a comprehensive and well-structured answer to the question based on the given web content. If the content doesn't contain relevant information, you can answer based on your general knowledge."""

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
        print(f"{ANSIColor.YELLOW.value}Welcome to Talk2URLs. Type 'exit' to quit, 'clear' to clear conversation history, or 'change urls' to update the target URLs.{ANSIColor.RESET.value}")
        print(f"{ANSIColor.CYAN.value}All generated responses will be saved in: {self.output_file}{ANSIColor.RESET.value}")

        while True:
            if not self.current_urls:
                urls_input = input(f"{ANSIColor.YELLOW.value}Enter the target URL(s) separated by commas: {ANSIColor.RESET.value}").strip()
                self.current_urls = [url.strip() for url in urls_input.split(',')]
                print(f"{ANSIColor.CYAN.value}Processing URLs...{ANSIColor.RESET.value}")
                self.context = self.process_urls(self.current_urls)
                print(f"{ANSIColor.NEON_GREEN.value}URLs processed. You can now ask questions about the content.{ANSIColor.RESET.value}")
                continue

            user_input = input(f"{ANSIColor.YELLOW.value}Enter your question or command: {ANSIColor.RESET.value}").strip()

            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using Talk2URLs. Goodbye!{ANSIColor.RESET.value}")
                break
            elif user_input.lower() == 'clear':
                self.conversation_history.clear()
                self.current_urls.clear()
                self.context = ""
                print(f"{ANSIColor.CYAN.value}Conversation history and current URLs cleared.{ANSIColor.RESET.value}")
                continue
            elif user_input.lower() == 'change urls':
                self.current_urls.clear()
                self.context = ""
                print(f"{ANSIColor.CYAN.value}Current URLs cleared. Please enter new URLs.{ANSIColor.RESET.value}")
                continue

            print(f"{ANSIColor.CYAN.value}Generating response...{ANSIColor.RESET.value}")
            response = self.generate_response(user_input, self.context)
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
        talk2url = Talk2URL(api_type)
        talk2url.run()
    else:
        print("Error: No API type provided.")
        print("Usage: python talk2url.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
