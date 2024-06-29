# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import json
import os
import sys
from urllib.parse import urljoin, quote_plus
from settings import settings
from run_model import ANSIColor
from openai import OpenAI

class InternetRAG:
    def __init__(self, api_type: str):
        self.crawl_database = {}
        self.load_crawl_database()
        self.client = self.configure_api(api_type)

    @staticmethod
    def configure_api(api_type: str) -> OpenAI:
        if api_type == "ollama":
            return OpenAI(base_url='http://localhost:11434/v1', api_key=settings.ollama_model)
        elif api_type == "llama":
            return OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        else:
            raise ValueError("Invalid API type")

    def load_crawl_database(self):
        if os.path.exists(settings.crawl_database_path):
            with open(settings.crawl_database_path, 'r') as f:
                self.crawl_database = json.load(f)

    def save_crawl_database(self):
        with open(settings.crawl_database_path, 'w') as f:
            json.dump(self.crawl_database, f, indent=4)

    def search_and_crawl(self, query):
        if query in self.crawl_database:
            print(f"{ANSIColor.CYAN.value}Using cached results for query: {query}{ANSIColor.RESET.value}")
            return self.crawl_database[query]

        print(f"{ANSIColor.CYAN.value}Performing search and crawling for query: {query}{ANSIColor.RESET.value}")
        search_results = self.perform_search(query)
        crawled_data = self.crawl_pages(search_results)

        self.crawl_database[query] = crawled_data
        self.save_crawl_database()

        return crawled_data

    def perform_search(self, query):
        try:
            search_url = f"https://www.google.com/search?q={quote_plus(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            search_results = []
            for result in soup.select('.yuRUbf > a'):
                href = result.get('href')
                if href.startswith('http'):
                    search_results.append(href)
                if len(search_results) >= settings.num_results:
                    break
            
            return search_results
        except Exception as e:
            print(f"{ANSIColor.PINK.value}Error performing search: {str(e)}{ANSIColor.RESET.value}")
            return []

    def crawl_pages(self, urls):
        crawled_data = []
        for url in urls:
            page_data = self.crawl_page(url, settings.crawl_depth)
            crawled_data.extend(page_data)
        return crawled_data

    def crawl_page(self, url, depth):
        if depth == 0:
            return []

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text(separator=' ', strip=True)
            
            # Extract links for further crawling
            links = [urljoin(url, link.get('href')) for link in soup.find_all('a', href=True)]
            
            # Recursive crawling
            sub_pages = []
            for link in links[:5]:  # Limit to 5 sub-links per page
                sub_pages.extend(self.crawl_page(link, depth - 1))
            
            return [{"url": url, "content": text_content}] + sub_pages
        
        except Exception as e:
            print(f"{ANSIColor.PINK.value}Error crawling {url}: {str(e)}{ANSIColor.RESET.value}")
            return []

    def process_crawled_data(self, crawled_data, query):
        system_message = f"""You are an AI assistant tasked with organizing and summarizing web search results. Given a query and the content from several web pages, provide a structured and comprehensive summary of the information related to the query. Focus on the most relevant and important points, and organize the information as follows:

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
{json.dumps(crawled_data, indent=2)}

Please provide a structured and comprehensive summary of the information related to the query, following the format specified in the system message. Focus on the most relevant and important points, and organize the information into main topics and subtopics."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]

        try:
            response = self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=messages,
                temperature=settings.temperature
            ).choices[0].message.content

            return response
        except Exception as e:
            print(f"{ANSIColor.PINK.value}Error in API call: {str(e)}{ANSIColor.RESET.value}")
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

            print(f"{ANSIColor.CYAN.value}Processing and structuring the crawled data...{ANSIColor.RESET.value}")
            processed_data = self.process_crawled_data(crawled_data, user_input)

            # Save the processed data to a file
            filename = f"internet_rag_{user_input.replace(' ', '_')}.txt"
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(processed_data)
                print(f"{ANSIColor.CYAN.value}Processed data saved as {filename}{ANSIColor.RESET.value}")
            except IOError as e:
                print(f"{ANSIColor.PINK.value}Error saving processed data to file: {str(e)}{ANSIColor.RESET.value}")

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
