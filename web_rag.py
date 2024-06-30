import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urljoin
from settings import settings
from talk2doc import ANSIColor
from openai import OpenAI
import random
import time
from duckduckgo_search import DDGS
import os
import re
from search_utils import SearchUtils
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebRAG:
    def __init__(self, api_type: str):
        self.client = self.configure_api(api_type)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        })
        self.chunk_size = 500
        self.overlap_size = 100
        self.model = SentenceTransformer(settings.model_name)
        self.search_utils = None
        self.all_search_results = []
        self.conversation_history = []
        self.conversation_context = deque(maxlen=settings.conversation_context_size * 2)
        self.current_urls = set()
        self.web_rag_file = "web_rag_qa.txt"
        self.current_question_file = None
        self.context_size = 5  # Initial context size

    @staticmethod
    def configure_api(api_type: str) -> OpenAI:
        if api_type == "ollama":
            return OpenAI(base_url='http://localhost:11434/v1', api_key=settings.ollama_model)
        elif api_type == "llama":
            return OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        else:
            raise ValueError("Invalid API type")

    def search_and_process(self, query):
        logging.info(f"Performing search for query: {query}")
        self.all_search_results = self.perform_search(query)
        logging.info(f"Search returned {len(self.all_search_results)} URLs")
        
        relevant_urls = self.filter_relevant_urls(self.all_search_results[:5], query)
        logging.info(f"Found {len(relevant_urls)} relevant URLs")
        
        summarized_query = self.summarize_query(query)
        self.current_question_file = f"web_rag_{summarized_query}.txt"
        self.process_relevant_urls(relevant_urls, self.current_question_file)
        
        answer = self.generate_qa(query)
        
        # Save question and answer to web_rag_qa.txt
        with open(self.web_rag_file, "a", encoding="utf-8") as f:
            f.write(f"Question: {query}\n\n")
            f.write(f"Answer: {answer}\n\n")
            f.write("-" * 50 + "\n\n")
        
        return answer

    def summarize_query(self, query):
        system_message = "You are an AI assistant tasked with summarizing a question into a short phrase suitable for a filename. Provide only the summary, no additional text. The summary should be 3-5 words long."
        user_message = f"Summarize this question into a short phrase (3-5 words): {query}"

        try:
            response = self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3
            ).choices[0].message.content.strip()

            # Remove any non-alphanumeric characters and replace spaces with underscores
            safe_filename = re.sub(r'[^a-zA-Z0-9\s]', '', response)
            safe_filename = safe_filename.replace(' ', '_').lower()

            # Truncate if it's still too long
            safe_filename = safe_filename[:50]

            return safe_filename
        except Exception as e:
            logging.error(f"Error summarizing query: {str(e)}")
            return "web_rag_query"  # Fallback to a generic filename

    def perform_search(self, query):
        search_results = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None, max_results=settings.num_urls_to_crawl):
                search_results.append(result)  # Append the entire result object
        return search_results

    def filter_relevant_urls(self, search_results, query):
        relevant_urls = []
        for result in search_results:
            if isinstance(result, dict) and self.is_url_relevant(result, query):
                relevant_urls.append(result)
        return relevant_urls

    def is_url_relevant(self, result, query):
        system_message = "You are an AI assistant tasked with determining if a search result is relevant to a given query. Respond with 'Yes' if the result seems relevant, and 'No' if it does not."
        user_message = f"Query: {query}\n\nSearch Result Title: {result.get('title', 'No title')}\nSearch Result Snippet: {result.get('body', 'No snippet')}\n\nIs this search result relevant to the query?"

        try:
            response = self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1
            ).choices[0].message.content.strip().lower()

            return response == 'yes'
        except Exception as e:
            logging.error(f"Error checking URL relevance: {str(e)}")
            return False

    def process_relevant_urls(self, relevant_urls, filename):
        all_content = []
        with open(filename, 'w', encoding='utf-8') as f:
            for result in relevant_urls:
                content = self.crawl_page(result['href'])
                if content:
                    f.write(f"URL: {result['href']}\n")
                    f.write(f"Title: {result.get('title', 'No title')}\n")
                    f.write("Content:\n")
                    chunks = self.create_chunks(content)
                    for chunk in chunks:
                        f.write(f"{chunk}\n")
                        all_content.append(chunk)
        
        logging.info(f"Saved content to {filename}")
        self._create_embeddings(all_content)

    def process_next_urls(self):
        if not self.current_question_file:
            print(f"{ANSIColor.PINK.value}No current question context. Please perform a search first.{ANSIColor.RESET.value}")
            return False

        unprocessed_urls = [result['href'] for result in self.all_search_results if result['href'] not in self.current_urls]
        urls_to_process = unprocessed_urls[:5]
        
        if not urls_to_process:
            print(f"{ANSIColor.PINK.value}No more URLs to process.{ANSIColor.RESET.value}")
            return False

        all_content = []
        with open(self.current_question_file, "a", encoding='utf-8') as f:
            for url in urls_to_process:
                content = self.crawl_page(url)
                if content:
                    self.current_urls.add(url)
                    f.write(f"\nURL: {url}\n\n")
                    f.write("Content:\n")
                    chunks = self.create_chunks(content)
                    for chunk in chunks:
                        f.write(f"{chunk}\n")
                        all_content.append(chunk)
                    
                    print(f"{ANSIColor.NEON_GREEN.value}Successfully processed and added content from {url}{ANSIColor.RESET.value}")
                else:
                    print(f"{ANSIColor.PINK.value}Failed to process content from {url}{ANSIColor.RESET.value}")

        if all_content:
            new_embeddings = self.model.encode(all_content, show_progress_bar=False)
            self.search_utils.db_embeddings = np.vstack([self.search_utils.db_embeddings, new_embeddings])
            self.search_utils.db_content.extend(all_content)

        self.context_size *= 2  # Double the context size for the next search
        return True

    def _create_embeddings(self, all_content):
        embeddings = self.model.encode(all_content, show_progress_bar=False)
        self.search_utils = SearchUtils(self.model, embeddings, all_content, None)

    def create_chunks(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap_size
        return chunks

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

    def generate_qa(self, query):
        if self.search_utils is None:
            logging.error("Search utils not initialized. Cannot generate Q&A.")
            return "I'm sorry, but I don't have enough information to answer that question."

        lexical_results = self.search_utils.lexical_search(query)[:self.context_size]
        text_results = self.search_utils.text_search(query)[:self.context_size]
        semantic_results = self.search_utils.semantic_search(query)[:self.context_size]
        
        combined_results = lexical_results + text_results + semantic_results
        
        if not combined_results:
            return "I'm sorry, but I don't have enough relevant information to answer that question."
        
        context = "\n\n".join(combined_results)
        
        conversation_context = " ".join(self.conversation_context)
        
        system_message = """You are an AI assistant tasked with answering questions based on the provided context and conversation history. Prioritize the most recent conversation context when answering questions, but also consider other relevant information if necessary. If the given context doesn't provide a suitable answer, rely on your general knowledge."""

        user_message = f"""Conversation Context:
{conversation_context}

Search Context:
{context}

Question: {query}

Please provide a comprehensive answer to the question based on the given context and conversation history. Prioritize the Conversation Context when answering, followed by the most relevant information from the Search Context. If none of the provided context is relevant, you can answer based on your general knowledge."""

        try:
            response = self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.conversation_history,
                    {"role": "user", "content": user_message}
                ],
                temperature=settings.temperature
            ).choices[0].message.content

            self.update_conversation_history(query, response)

            return response
        except Exception as e:
            logging.error(f"Error generating Q&A: {str(e)}")
            return "I'm sorry, but I encountered an error while trying to answer your question."

    def update_conversation_history(self, query, response):
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        self.conversation_context.append(query)
        self.conversation_context.append(response)

        if len(self.conversation_history) > settings.max_history_length * 2:
            self.conversation_history = self.conversation_history[-settings.max_history_length * 2:]

    def run(self):
        print(f"{ANSIColor.YELLOW.value}Welcome to the Web RAG System. Type 'exit' to quit, 'clear' to clear conversation history, or 'check' to process more URLs and update the knowledge base.{ANSIColor.RESET.value}")

        while True:
            user_input = input(f"{ANSIColor.YELLOW.value}Enter your search query, follow-up question, or command: {ANSIColor.RESET.value}").strip()

            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using the Web RAG System. Goodbye!{ANSIColor.RESET.value}")
                break
            elif user_input.lower() == 'clear':
                self.conversation_history.clear()
                self.conversation_context.clear()
                self.current_question_file = None
                self.context_size = 5  # Reset context size
                print(f"{ANSIColor.CYAN.value}Conversation history and context cleared.{ANSIColor.RESET.value}")
                continue
            elif user_input.lower() == 'check':
                print(f"{ANSIColor.CYAN.value}Processing next 5 URLs and updating knowledge base...{ANSIColor.RESET.value}")
                if self.process_next_urls():
                    if self.current_question_file:
                        last_query = self.conversation_history[-2]['content'] if self.conversation_history else "Previous question"
                        print(f"{ANSIColor.CYAN.value}Generating new answer based on expanded information...{ANSIColor.RESET.value}")
                        new_answer = self.generate_qa(last_query)
                        print(f"\n{ANSIColor.NEON_GREEN.value}Updated Answer:{ANSIColor.RESET.value}\n{new_answer}")
                        with open(self.web_rag_file, "a", encoding="utf-8") as f:
                            f.write(f"Updated Answer:\n{new_answer}\n\n")
                            f.write("-" * 50 + "\n\n")
                    print(f"{ANSIColor.NEON_GREEN.value}Knowledge base updated. You can now ask questions with the expanded information.{ANSIColor.RESET.value}")
                continue

            if not user_input:
                print(f"{ANSIColor.PINK.value}Please enter a valid query.{ANSIColor.RESET.value}")
                continue

            if not self.search_utils:
                print(f"{ANSIColor.CYAN.value}Searching and processing web content...{ANSIColor.RESET.value}")
                answer = self.search_and_process(user_input)
                print(f"{ANSIColor.NEON_GREEN.value}Relevant content has been processed.{ANSIColor.RESET.value}")
            else:
                print(f"{ANSIColor.CYAN.value}Generating answer based on existing knowledge...{ANSIColor.RESET.value}")
                answer = self.generate_qa(user_input)

            print(f"\n{ANSIColor.NEON_GREEN.value}Answer:{ANSIColor.RESET.value}\n{answer}")

            print(f"{ANSIColor.NEON_GREEN.value}You can ask follow-up questions, start a new search, or use 'check' to process more URLs and update the knowledge base.{ANSIColor.RESET.value}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        web_rag = WebRAG(api_type)
        web_rag.run()
    else:
        print("Error: No API type provided.")
        print("Usage: python web_rag.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
