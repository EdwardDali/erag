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

    def search_and_process(self, query):
        logging.info(f"Performing search for query: {query}")
        search_results = self.perform_search(query)
        logging.info(f"Search returned {len(search_results)} URLs")
        
        relevant_urls = self.filter_relevant_urls(search_results, query)
        logging.info(f"Found {len(relevant_urls)} relevant URLs")
        
        summaries = self.process_relevant_urls(relevant_urls, query)
        
        final_summary = self.create_final_summary(summaries, query)
        
        return final_summary

    def perform_search(self, query):
        search_results = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None, max_results=settings.num_urls_to_crawl):
                search_results.append({'url': result["href"], 'title': result["title"], 'body': result["body"]})
        
        return search_results

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

    def process_relevant_urls(self, relevant_urls, query):
        summaries = []
        for i, result in enumerate(relevant_urls, 1):
            content = self.crawl_page(result['url'])
            if content:
                filename = f"internet_rag_{query.replace(' ', '_')}_{i}.txt"
                self.save_content(filename, content)
                summary = self.create_summary(content, query, i)
                summaries.append(summary)
                self.append_summary(f"internet_rag_{query.replace(' ', '_')}.txt", summary, i)
        return summaries

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

    def save_content(self, filename, content):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Content saved as {filename}")
        except IOError as e:
            logging.error(f"Error saving content to file: {str(e)}")

    def create_summary(self, content, query, index):
        system_message = f"""You are an AI assistant tasked with summarizing web content related to a given query. 
        Create a summary of approximately {settings.summary_size} characters. Focus on the most relevant and important points related to the query: {query}."""

        user_message = f"Web content:\n{content}\n\nPlease summarize this content in relation to the query: {query}"

        try:
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
            with open(filename, "a", encoding="utf-8") as f:
                f.write(summary)
            logging.info(f"Summary {index} appended to {filename}")
        except IOError as e:
            logging.error(f"Error appending summary to file: {str(e)}")

    def create_final_summary(self, summaries, query):
        combined_summaries = ''.join(summaries)
        system_message = f"""You are an AI assistant tasked with creating a final summary of web search results. 
        You will be given a series of summaries from different web pages, all related to the query: {query}. 
        Create a comprehensive final summary of approximately {settings.final_summary_size} characters. 
        Integrate information from all provided summaries, avoid redundancy, and ensure the final summary is well-structured and informative."""

        user_message = f"Individual summaries:\n{combined_summaries}\n\nPlease create a final comprehensive summary related to the query: {query}"

        try:
            response = self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=settings.temperature
            ).choices[0].message.content

            filename = f"internet_rag_{query.replace(' ', '_')}_final.txt"
            self.save_content(filename, response)
            return response
        except Exception as e:
            logging.error(f"Error in API call for creating final summary: {str(e)}")
            return "Error in creating final summary."

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

            print(f"{ANSIColor.CYAN.value}Searching and processing web content...{ANSIColor.RESET.value}")
            final_summary = self.search_and_process(user_input)

            if not final_summary:
                print(f"{ANSIColor.PINK.value}Sorry, no relevant results found for your query. Please try a different search term.{ANSIColor.RESET.value}")
                continue

            print(f"\n{ANSIColor.NEON_GREEN.value}Final Summary:{ANSIColor.RESET.value}")
            print(final_summary)

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
