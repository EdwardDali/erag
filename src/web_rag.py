
from src.settings import settings
from openai import OpenAI
import logging
from bs4 import BeautifulSoup
import requests
from duckduckgo_search import DDGS
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import deque
from src.search_utils import SearchUtils
from src.look_and_feel import success, info, warning, error
from src.api_model import configure_api, LlamaClient
import os

class WebRAG:
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
        self.model = SentenceTransformer(settings.sentence_transformer_model)
        self.search_utils = None
        self.all_search_results = []
        self.conversation_history = []
        self.conversation_context = deque(maxlen=settings.conversation_context_size * 2)
        self.current_urls = set()
        
        # Correct the web_rag_file path
        self.web_rag_file = os.path.join(settings.output_folder, os.path.basename(settings.web_rag_file))
        
        self.current_question_file = None
        self.context_size = settings.initial_context_size
        self.processed_urls = set()  # Keep track of processed URLs
        self.current_query = None  # Store the current query
        self.search_offset = 0  # New attribute to keep track of search offset
        
        # Ensure output folder exists
        os.makedirs(settings.output_folder, exist_ok=True)


    def search_and_process(self, query):
        logging.info(f"Performing search for query: {query}")
        self.current_query = query
        self.search_offset = 0
        self.all_search_results = []
        search_results = self.perform_search(query)
        logging.info(f"Search returned {len(search_results)} URLs")
        
        relevant_urls = self.filter_relevant_urls(search_results, query)
        logging.info(f"Found {len(relevant_urls)} relevant URLs")
        
        self.current_question_file = self.summarize_query(query)
        self.process_relevant_urls(relevant_urls, self.current_question_file)
        
        answer = self.generate_qa(query)
        
        with open(self.web_rag_file, "a", encoding="utf-8") as f:
            f.write(f"Question: {query}\n\n")
            f.write(f"Answer: {answer}\n\n")
            f.write("-" * 50 + "\n\n")
        
        return answer

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

            return os.path.join(settings.output_folder, f"web_rag_{safe_filename}.txt")
        except Exception as e:
            logging.error(f"Error summarizing query: {str(e)}")
            return os.path.join(settings.output_folder, f"web_rag_query_{hash(query) % 10000}.txt")

    def perform_search(self, query, offset=0):
        if offset == 0 or not self.all_search_results:
            self.all_search_results = []
            with DDGS() as ddgs:
                for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None, max_results=settings.web_rag_urls_to_crawl * 5):
                    self.all_search_results.append(result)
        
        start = offset
        end = offset + settings.web_rag_urls_to_crawl
        return self.all_search_results[start:end]

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
        if not self.current_query:
            print(f"{ANSIColor.PINK.value}No current query context. Please perform a search first.{ANSIColor.RESET.value}")
            return False

        new_search_results = self.perform_search(self.current_query, offset=self.search_offset)
        self.search_offset += settings.web_rag_urls_to_crawl
        
        urls_to_process = [result for result in new_search_results if result['href'] not in self.processed_urls]
        
        if not urls_to_process:
            print(f"{ANSIColor.PINK.value}No new URLs found to process. Try a new search query.{ANSIColor.RESET.value}")
            return False

        all_content = []
        with open(self.current_question_file, "a", encoding='utf-8') as f:
            for result in urls_to_process:
                url = result['href']
                content = self.crawl_page(url)
                if content:
                    self.processed_urls.add(url)
                    f.write(f"\nURL: {url}\n")
                    f.write(f"Title: {result.get('title', 'No title')}\n")
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

        return True
        
    def _create_embeddings(self, all_content):
        embeddings = self.model.encode(all_content, show_progress_bar=False)
        self.search_utils = SearchUtils(self.model, embeddings, all_content, None)

    def create_chunks(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = start + settings.web_rag_chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - settings.web_rag_overlap_size
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
        
        system_message = """You are an AI assistant tasked with answering questions based on the provided context and conversation history. Follow these guidelines:

1. Prioritize the most recent conversation context when answering questions, but also consider other relevant information if necessary.
2. If the given context doesn't provide a suitable answer, rely on your general knowledge.
3. Structure your answer in a clear, organized manner:
   - Start with a brief overview or main point.
   - Use headings for main topics if applicable.
   - Use bullet points or numbered lists for details or sub-points.
4. Stay focused on the specific question asked. If the question is about a particular aspect or metric, concentrate on that in your answer.
5. If the question is asking for specific data or numbers, prioritize providing that information.
6. Be concise but comprehensive. Aim for a well-structured answer that covers the main points without unnecessary elaboration."""

        user_message = f"""Conversation Context:
{conversation_context}

Search Context:
{context}

Question: {query}

Please provide a comprehensive and well-structured answer to the question based on the given context and conversation history. Prioritize the Conversation Context when answering, followed by the most relevant information from the Search Context. If none of the provided context is relevant, you can answer based on your general knowledge."""

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

    def get_response(self, query: str) -> str:
        if not self.search_utils or not self.current_query:
            print(f"{ANSIColor.CYAN.value}Searching and processing web content...{ANSIColor.RESET.value}")
            answer = self.search_and_process(query)
        else:
            print(f"{ANSIColor.CYAN.value}Generating answer based on existing knowledge...{ANSIColor.RESET.value}")
            answer = self.generate_qa(query)

        return answer

    def run(self):
        print(info("Welcome to the Web RAG System. Type 'exit' to quit, 'clear' to clear conversation history, or 'check' to process more URLs and update the knowledge base."))
        print(info(f"All generated files will be saved in: {settings.output_folder}"))

        while True:
            user_input = input(info("Enter your search query, follow-up question, or command: ")).strip()

            if user_input.lower() == 'exit':
                print(success("Thank you for using the Web RAG System. Goodbye!"))
                break
            elif user_input.lower() == 'clear':
                self.conversation_history.clear()
                self.conversation_context.clear()
                self.current_question_file = None
                self.current_query = None
                self.context_size = settings.initial_context_size
                self.processed_urls.clear()
                self.search_utils = None
                self.search_offset = 0
                print(info("Conversation history, context, and processed URLs cleared."))
                continue
            elif user_input.lower() == 'check':
                if not self.current_query:
                    print(warning("No current query. Please perform a search first."))
                    continue
                print(info("Searching for new URLs and updating knowledge base..."))
                if self.process_next_urls():
                    print(info("Generating new answer based on expanded information..."))
                    new_answer = self.generate_qa(self.current_query)
                    print(f"\n{success('Updated Answer:')}\n{new_answer}")
                    with open(self.web_rag_file, "a", encoding="utf-8") as f:
                        f.write(f"Updated Answer:\n{new_answer}\n\n")
                        f.write("-" * 50 + "\n\n")
                    print(success("Knowledge base updated. You can now ask questions with the expanded information."))
                continue

            if not self.search_utils or not self.current_query:
                print(info("Searching and processing web content..."))
                answer = self.search_and_process(user_input)
                print(f"\n{success('Answer:')}\n{answer}")
                print(success("Relevant content has been processed. You can now ask follow-up questions or use 'check' to process more URLs."))
            else:
                print(info("Generating answer based on existing knowledge..."))
                answer = self.generate_qa(user_input)
                print(f"\n{success('Answer:')}\n{answer}")

            print(success("You can ask follow-up questions, start a new search, or use 'check' to process more URLs and update the knowledge base."))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        web_rag = WebRAG(api_type)
        web_rag.run()
    else:
        print(error("Error: No API type provided."))
        print(warning("Usage: python src/web_rag.py <api_type>"))
        print(info("Available API types: ollama, llama"))
        sys.exit(1)
