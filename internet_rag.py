# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import json
import os
import sys
import logging
from urllib.parse import urljoin, quote_plus
from settings import settings
from run_model import ANSIColor
from openai import OpenAI
import random
import time
from duckduckgo_search import DDGS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InternetRAG:
    def __init__(self, api_type: str):
        self.client = self.configure_api(api_type)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        })

    @staticmethod
    def configure_api(api_type: str) -> OpenAI:
        if api_type == "ollama":
            return OpenAI(base_url='http://localhost:11434/v1', api_key=settings.ollama_model)
        elif api_type == "llama":
            return OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        else:
            raise ValueError("Invalid API type")

    def search_and_crawl(self, query):
        logging.info(f"Performing search and crawling for query: {query}")
        search_results = self.perform_search(query)
        logging.info(f"Search returned {len(search_results)} URLs")
        crawled_data = self.crawl_pages(search_results, query)

        if crawled_data:
            self.save_raw_results(query, crawled_data)
            logging.info(f"Crawled and saved data for query: {query}")
        else:
            logging.warning(f"No data could be crawled for the query: {query}")

        return crawled_data

    def save_raw_results(self, query, crawled_data):
        filename = f"internet_rag_{query.replace(' ', '_')}_noLLM.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            for item in crawled_data:
                f.write(f"Source: DuckDuckGo\n")
                f.write(f"URL: {item['url']}\n")
                f.write(f"Content:\n{item['content']}\n\n")
                f.write("-" * 80 + "\n\n")
        logging.info(f"Raw results saved to {filename}")

    def perform_search(self, query):
        search_results = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None, max_results=settings.num_results):
                search_results.append({'url': result["href"], 'title': result["title"], 'body': result["body"]})
        
        logging.info(f"Total search results found: {len(search_results)}")
        return search_results

    def crawl_pages(self, urls, query):
        crawled_data = []
        for url_data in urls:
            page_data = self.crawl_page(url_data['url'], settings.crawl_depth, query, url_data['title'], url_data['body'])
            if page_data:
                crawled_data.append(page_data)
        logging.info(f"Crawled {len(crawled_data)} pages in total")
        return crawled_data

    def crawl_page(self, url, depth, query, title, snippet):
        if depth == 0:
            return None

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, features="lxml")
            
            # Extract text content
            for script in soup(["script", "style"]):
                script.extract()
            text_content = ' '.join(soup.stripped_strings)
            
            # Combine title, snippet, and full content
            full_content = f"Title: {title}\n\nSnippet: {snippet}\n\nFull Content:\n{text_content}"
            
            # Check if the content is relevant to the query using Ollama
            if self.is_content_relevant(full_content, query):
                logging.info(f"Successfully crawled {url}")
                return {"url": url, "content": full_content}
            else:
                logging.info(f"Skipped {url} due to irrelevant content")
                return None
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Error crawling {url}: {str(e)}")
            return None

    def is_content_relevant(self, content, query):
        system_message = "You are an AI assistant tasked with determining if a piece of text is relevant to a given query. Respond with 'Yes' if the content is relevant, and 'No' if it is not."
        user_message = f"Query: {query}\n\nContent: {content[:500]}...\n\nIs this content relevant to the query?"

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
            logging.error(f"Error checking content relevance: {str(e)}")
            return False

    def process_crawled_data(self, query):
        filename = f"internet_rag_{query.replace(' ', '_')}_noLLM.txt"
        if not os.path.exists(filename):
            return f"No data could be found for the query: {query}"

        with open(filename, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        system_message = f"""You are an AI assistant tasked with summarizing web search results. Given a query and the content from several web pages, provide a structured and comprehensive summary of the information related to the query. Focus on the most relevant and important points, and organize the information as follows:

Title: Internet RAG Results: {query}

1. [Main Topic 1]
1.1. [Subtopic 1.1]
• [Key point 1]
• [Key point 2]
• [Key point 3]
...

1.2. [Subtopic 1.2]
• [Relevant information 1]
• [Relevant information 2]
• [Relevant information 3]
...

2. [Main Topic 2]
...

Ensure that the structure is consistent and the information is detailed and accurate. Aim to provide at least 3-5 points for each subtopic, but you can include more if necessary. Stay strictly on the topic of the query and do not include information about unrelated subjects."""

        user_input = f"""Query: {query}

Web page contents:
{raw_data}

Please provide a structured and comprehensive summary of the information related to the query, following the format specified in the system message. Focus on the most relevant and important points, and organize the information into main topics and subtopics."""

        try:
            response = self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_input}
                ],
                temperature=settings.temperature
            ).choices[0].message.content

            return response
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your request."

    def run(self):
        print(f"{ANSIColor.YELLOW.value}Welcome to the Internet RAG System. Type 'exit' to quit.{ANSIColor.RESET.value}")

        while True:
            user_input = input(f"{ANSIColor.YELLOW.value}Enter your search query: {ANSIColor.RESET.value}").strip()

            if user_input.lower() == 'exit':
                print(f"{ANSIColor.NEON_GREEN.value}Thank you for using the Internet RAG System. Goodbye!{ANSIColor.RESET.value}")
                break

            if not user_input:
                print(f"{ANSIColor.PINK.value}Please enter a valid query.{ANSIColor.RESET.value}")
                continue

            print(f"{ANSIColor.CYAN.value}Searching and crawling the web...{ANSIColor.RESET.value}")
            crawled_data = self.search_and_crawl(user_input)

            if not crawled_data:
                print(f"{ANSIColor.PINK.value}Sorry, no results found for your query. Please try a different search term.{ANSIColor.RESET.value}")
                continue

            print(f"{ANSIColor.CYAN.value}Processing and structuring the crawled data...{ANSIColor.RESET.value}")
            processed_data = self.process_crawled_data(user_input)

            # Save the processed data to a file
            filename = f"internet_rag_{user_input.replace(' ', '_')}.txt"
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(processed_data)
                print(f"{ANSIColor.CYAN.value}Processed data saved as {filename}{ANSIColor.RESET.value}")
            except IOError as e:
                logging.error(f"Error saving processed data to file: {str(e)}")

            print(f"\n{ANSIColor.NEON_GREEN.value}Processed Internet RAG Results:{ANSIColor.RESET.value}")
            print(processed_data)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        internet_rag = InternetRAG(api_type)
        internet_rag.run()
    else:
        print("Error: No API type provided.")
        print("Usage: python internet_rag.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
