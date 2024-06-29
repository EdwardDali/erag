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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        else:
            self.crawl_database = {}

    def save_crawl_database(self):
        with open(settings.crawl_database_path, 'w') as f:
            json.dump(self.crawl_database, f, indent=4)

    def search_and_crawl(self, query):
        if query in self.crawl_database and self.crawl_database[query]:
            logging.info(f"Using cached results for query: {query}")
            return self.crawl_database[query]

        logging.info(f"Performing search and crawling for query: {query}")
        search_results = self.perform_search(query)
        logging.info(f"Search returned {len(search_results)} URLs")
        crawled_data = self.crawl_pages(search_results, query)

        if crawled_data:
            self.crawl_database[query] = crawled_data
            self.save_crawl_database()
            logging.info(f"Crawled and saved data for query: {query}")
        else:
            logging.warning(f"No data could be crawled for the query: {query}")

        return crawled_data

    def perform_search(self, query):
        search_engines = [
            ("https://duckduckgo.com/html/?q={}", self.parse_duckduckgo),
            ("https://www.bing.com/search?q={}", self.parse_bing),
            ("https://search.yahoo.com/search?p={}", self.parse_yahoo),
            ("https://www.qwant.com/?q={}", self.parse_qwant),
            ("https://www.startpage.com/do/search?q={}", self.parse_startpage),
            ("https://www.google.com/search?q={}", self.parse_google)
        ]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        search_results = []
        for search_url, parse_function in search_engines:
            try:
                formatted_url = search_url.format(quote_plus(query))
                logging.info(f"Trying search engine: {formatted_url}")
                response = requests.get(formatted_url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = parse_function(soup)
                
                logging.debug(f"Raw results from {formatted_url}: {results[:5]}")  # Debug: show first 5 raw results
                
                for href in results:
                    if href and href.startswith('http') and not href.startswith('http://go.microsoft.com'):
                        search_results.append(href)
                    if len(search_results) >= settings.num_results:
                        break
                
                if search_results:
                    logging.info(f"Found {len(search_results)} search results from {formatted_url}")
                    break  # Exit the loop if we have results
                else:
                    logging.info(f"No results found from {formatted_url}")

            except Exception as e:
                logging.error(f"Error performing search with {formatted_url}: {str(e)}")
        
        logging.info(f"Total search results found: {len(search_results)}")
        return search_results

    def parse_duckduckgo(self, soup):
        return [result.get('href') for result in soup.find_all('a', class_='result__a')] or \
               [result.get('href') for result in soup.find_all('a') if result.get('href') and result.get('href').startswith('http')]

    def parse_bing(self, soup):
        return [result.get('href') for result in soup.find_all('a', class_='b_attribution')] or \
               [result.get('href') for result in soup.find_all('cite')] or \
               [result.get('href') for result in soup.find_all('a') if result.get('href') and result.get('href').startswith('http')]

    def parse_yahoo(self, soup):
        return [result.get('href') for result in soup.find_all('a', class_=' ac-algo fz-l ac-21th lh-24')] or \
               [result.get('href') for result in soup.find_all('a', class_='ac-algo')] or \
               [result.get('href') for result in soup.find_all('a') if result.get('href') and result.get('href').startswith('http')]

    def parse_qwant(self, soup):
        return [result.get('href') for result in soup.find_all('a', class_='external_link')] or \
               [result.get('href') for result in soup.find_all('a', class_='url')] or \
               [result.get('href') for result in soup.find_all('a') if result.get('href') and result.get('href').startswith('http')]

    def parse_startpage(self, soup):
        return [result.get('href') for result in soup.find_all('a', class_='w-gl__result-url')] or \
               [result.get('href') for result in soup.find_all('a', class_='result-link')] or \
               [result.get('href') for result in soup.find_all('a') if result.get('href') and result.get('href').startswith('http')]

    def parse_google(self, soup):
        return [result.get('href') for result in soup.find_all('a', class_='l')] or \
               [result.get('href') for result in soup.find_all('cite')] or \
               [result.get('href') for result in soup.find_all('a') if result.get('href') and result.get('href').startswith('http')]

    def crawl_pages(self, urls, query):
        crawled_data = []
        for url in urls:
            page_data = self.crawl_page(url, settings.crawl_depth, query)
            if page_data:
                crawled_data.extend(page_data)
        logging.info(f"Crawled {len(crawled_data)} pages in total")
        return crawled_data

    def crawl_page(self, url, depth, query):
        if depth == 0:
            return []

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text(separator=' ', strip=True)
            
            # Check if the content is relevant to the query using Ollama
            if self.is_content_relevant(text_content, query):
                # Extract links for further crawling
                links = [urljoin(url, link.get('href')) for link in soup.find_all('a', href=True)]
                
                # Recursive crawling
                sub_pages = []
                for link in links[:5]:  # Limit to 5 sub-links per page
                    sub_pages.extend(self.crawl_page(link, depth - 1, query))
                
                logging.info(f"Successfully crawled {url}")
                return [{"url": url, "content": text_content}] + sub_pages
            else:
                logging.info(f"Skipped {url} due to irrelevant content")
                return []
        
        except Exception as e:
            logging.error(f"Error crawling {url}: {str(e)}")
            return []

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

    def process_crawled_data(self, crawled_data, query):
        if not crawled_data:
            return f"No data could be crawled for the query: {query}"

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
{json.dumps(crawled_data, indent=2)}

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

            print(f"{ANSIColor.CYAN.value}Processing and structuring the crawled data...{ANSIColor.RESET.value}")
            processed_data = self.process_crawled_data(crawled_data, user_input)

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
