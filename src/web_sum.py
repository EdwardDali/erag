from src.settings import settings
from openai import OpenAI
import logging
from bs4 import BeautifulSoup
import requests
from duckduckgo_search import DDGS
import re
from src.api_model import configure_api, LlamaClient
import os
from src.color_scheme import Colors, colorize
import colorama

# Initialize colorama
colorama.init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebSum:
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
        
        # Ensure output folder exists
        os.makedirs(settings.output_folder, exist_ok=True)


    def search_and_process(self, query):
        logging.info(f"Performing search for query: {query}")
        search_results = self.perform_search(query)
        logging.info(f"Search returned {len(search_results)} URLs")
        
        relevant_urls = self.filter_relevant_urls(search_results, query)
        logging.info(f"Found {len(relevant_urls)} relevant URLs")
        
        summaries = self.process_relevant_urls(relevant_urls, query)
        
        final_summary = self.create_final_summary(summaries, query)
        
        return final_summary

    def summarize_query(self, query):
        system_message = "You are an AI assistant tasked with summarizing a question into a short phrase suitable for a filename. Provide only the summary, no additional text. The summary should be 3-5 words long."
        user_message = f"Summarize this question into a short phrase (3-5 words): {query}"

        try:
            if self.api_type == "llama":
                response = self.client.chat([
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ], temperature=settings.temperature)
            else:
                response = self.client.chat.completions.create(
                    model=settings.ollama_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=settings.temperature
                ).choices[0].message.content

            safe_filename = re.sub(r'[^a-zA-Z0-9\s]', '', response.strip())
            safe_filename = safe_filename.replace(' ', '_').lower()
            safe_filename = safe_filename[:50]  # Limit filename length

            return f"web_sum_{safe_filename}.txt"
        except Exception as e:
            logging.error(f"Error summarizing query: {str(e)}")
            return f"web_sum_query_{hash(query) % 10000}.txt"  # Fallback filename

    def perform_search(self, query):
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None, max_results=settings.web_sum_urls_to_crawl):
                results.append(r)
        return results

    def filter_relevant_urls(self, search_results, query):
        relevant_urls = []
        for result in search_results:
            if self.is_url_relevant(result, query):
                relevant_urls.append(result)
        return relevant_urls

    def is_url_relevant(self, result, query):
        system_message = "You are an AI assistant tasked with determining if a search result is relevant to a given query. Respond with 'Yes' if the result seems relevant, and 'No' if it does not."
        user_message = f"Query: {query}\n\nSearch Result Title: {result['title']}\nSearch Result Snippet: {result['body']}\n\nIs this search result relevant to the query?"

        try:
            if self.api_type == "llama":
                response = self.client.chat([
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ], temperature=0.1)
            else:
                response = self.client.chat.completions.create(
                    model=settings.ollama_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1
                ).choices[0].message.content

            return response.strip().lower() == 'yes'
        except Exception as e:
            logging.error(f"Error checking URL relevance: {str(e)}")
            return False

    def process_relevant_urls(self, relevant_urls, query):
        summaries = []
        for i, result in enumerate(relevant_urls, 1):
            content = self.crawl_page(result['url'])
            if content:
                filename = self.summarize_query(f"{query}_{i}")
                self.save_content(filename, content)
                summary = self.create_summary(content, query, i)
                summaries.append(summary)
                self.append_summary(f"web_sum_{query.replace(' ', '_')}.txt", summary, i)
        return summaries

    def crawl_page(self, url):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, features="lxml")
            
            for script in soup(["script", "style"]):
                script.extract()
            text_content = ' '.join(soup.stripped_strings)
            
            return text_content
        except requests.exceptions.RequestException as e:
            logging.error(f"Error crawling {url}: {str(e)}")
            return None

    def save_content(self, filename, content):
        try:
            file_path = os.path.join(settings.output_folder, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Content saved as {file_path}")
        except IOError as e:
            logging.error(f"Error saving content to file: {str(e)}")

    def create_summary(self, content, query, index):
        system_message = f"""You are an AI assistant tasked with summarizing web content related to a given query. 
        Create a summary of approximately {settings.summary_size} characters. Focus on the most relevant and important points related to the query: {query}."""

        user_message = f"Web content:\n{content}\n\nPlease summarize this content in relation to the query: {query}"

        try:
            if self.api_type == "llama":
                response = self.client.chat([
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ], temperature=settings.temperature)
            else:
                response = self.client.chat.completions.create(
                    model=settings.ollama_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=settings.temperature
                ).choices[0].message.content

            return f"Summary {index}:\n{response}\n\n{'='*50}\n\n"
        except Exception as e:
            logging.error(f"Error in API call for creating summary: {str(e)}")
            return f"Summary {index}: Error in creating summary.\n\n{'='*50}\n\n"

    def append_summary(self, filename, summary, index):
        try:
            file_path = os.path.join(settings.output_folder, filename)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(summary)
            logging.info(f"Summary {index} appended to {file_path}")
        except IOError as e:
            logging.error(f"Error appending summary to file: {str(e)}")

    def summarize_content(self, content):
        system_message = f"You are an AI assistant that summarizes web content. Provide a concise summary of the given text, focusing on the main points and key information. The summary should be about {settings.summary_size} characters long."
        user_message = f"Summarize the following text:\n\n{content}"

        try:
            if self.api_type == "llama":
                response = self.client.chat([
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ], temperature=settings.temperature)
            else:
                response = self.client.chat.completions.create(
                    model=settings.ollama_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=settings.temperature
                ).choices[0].message.content

            return response.strip()
        except Exception as e:
            logging.error(f"Error in summarization: {str(e)}")
            return "Error in summarization."

    def create_final_summary(self, summaries):
        system_message = f"You are an AI assistant that creates a comprehensive final summary based on multiple individual summaries. Combine the information from all provided summaries into a coherent and detailed final summary. The final summary should be about {settings.final_summary_size} characters long."
        user_message = f"Create a final summary based on the following individual summaries:\n\n" + "\n\n".join(summaries)

        try:
            if self.api_type == "llama":
                response = self.client.chat([
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ], temperature=settings.temperature)
            else:
                response = self.client.chat.completions.create(
                    model=settings.ollama_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=settings.temperature
                ).choices[0].message.content

            return response.strip()
        except Exception as e:
            logging.error(f"Error in creating final summary: {str(e)}")
            return "Error in creating final summary."

    def run(self):
        print(colorize("Welcome to the Web Summarization System. Type 'exit' to quit.", Colors.INFO))
        
        while True:
            query = input(colorize("Enter your search query: ", Colors.INFO)).strip()
            
            if query.lower() == 'exit':
                print(colorize("Thank you for using the Web Summarization System. Goodbye!", Colors.SUCCESS))
                break
            
            if not query:
                print(colorize("Please enter a valid query.", Colors.ERROR))
                continue
            
            print(colorize("Performing search...", Colors.INFO))
            search_results = self.perform_search(query)
            
            if not search_results:
                print(colorize("No search results found. Please try a different query.", Colors.WARNING))
                continue
            
            print(colorize(f"Found {len(search_results)} results. Crawling pages...", Colors.INFO))
            
            summaries = []
            for result in search_results:
                print(colorize(f"Processing: {result['title']}", Colors.INFO))
                content = self.crawl_page(result['href'])
                if content:
                    summary = self.summarize_content(content)
                    summaries.append(summary)
                    print(colorize("Summary created.", Colors.SUCCESS))
                else:
                    print(colorize("Failed to crawl page.", Colors.ERROR))
            
            if not summaries:
                print(colorize("Failed to create any summaries. Please try again with a different query.", Colors.ERROR))
                continue
            
            print(colorize("Creating final summary...", Colors.INFO))
            final_summary = self.create_final_summary(summaries)
            
            print(colorize("\nFinal Summary:", Colors.SUCCESS))
            print(final_summary)
            
            # Save the final summary to a file
            safe_query = re.sub(r'[^a-zA-Z0-9_-]', '_', query)
            filename = f"web_sum_{safe_query[:50]}.txt"
            file_path = os.path.join(settings.output_folder, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n\n")
                f.write(f"Final Summary:\n{final_summary}\n\n")
                f.write("Individual Summaries:\n\n")
                for i, summary in enumerate(summaries, 1):
                    f.write(f"Summary {i}:\n{summary}\n\n")
            
            print(colorize(f"Results saved to {file_path}", Colors.SUCCESS))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        web_sum = WebSum(api_type)
        web_sum.run()
    else:
        print(colorize("Error: No API type provided.", Colors.ERROR))
        print(colorize("Usage: python src/web_sum.py <api_type>", Colors.INFO))
        print(colorize("Available API types: ollama, llama", Colors.INFO))
        sys.exit(1)
