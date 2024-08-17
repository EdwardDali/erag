import asyncio
import os
from datetime import datetime
from typing import List, Dict
import logging
from src.api_model import EragAPI, create_erag_api
from src.settings import settings
from src.look_and_feel import error, success, info, llm_response, user_input

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MixAgents:
    def __init__(self, worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, manager_erag_api: EragAPI = None):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.manager_erag_api = manager_erag_api
        self.output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "mix_agents")
        os.makedirs(self.output_folder, exist_ok=True)
        self.session_log = []
        self.session_file = None
        logging.info(f"MixAgents initialized. Output folder: {self.output_folder}")

    async def run_llm(self, erag_api: EragAPI, messages: List[Dict[str, str]]):
        """Run a single LLM call with a given EragAPI instance."""
        response = erag_api.chat(messages, temperature=settings.temperature)
        return erag_api.model, response

    async def get_responses(self, user_prompt: str):
        """Get responses from all available models."""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_prompt}
        ]
        tasks = [
            self.run_llm(self.worker_erag_api, messages),
            self.run_llm(self.supervisor_erag_api, messages)
        ]
        if self.manager_erag_api:
            tasks.append(self.run_llm(self.manager_erag_api, messages))
        
        results = await asyncio.gather(*tasks)
        return results

    def aggregate_responses(self, user_prompt: str, responses: List[tuple]):
        """Aggregate responses using the supervisor model with improved synthesis."""
        aggregator_system_prompt = """
        You are an expert information aggregator. Your task is to combine and consolidate multiple AI responses to a user query. Follow these steps:
        1. Identify all unique pieces of information from each response.
        2. Combine these pieces into a comprehensive list or explanation.
        3. Eliminate redundancies while preserving all unique information.
        4. If there are conflicting pieces of information, include all versions and note the conflict.
        5. Organize the information in a clear, structured manner that directly addresses the user's query.
        6. Do not summarize or analyze the different responses. Instead, focus on presenting all unique information in a consolidated format.
        """
        
        aggregator_messages = [
            {"role": "system", "content": aggregator_system_prompt},
            {"role": "user", "content": f"User Query: {user_prompt}\n\nResponses from models:\n" + "\n".join([f"{model}:\n{response}" for model, response in responses])}
        ]
        
        return self.supervisor_erag_api.chat(aggregator_messages, temperature=settings.temperature)

    def save_interaction(self, interaction):
        """Save a single interaction to the session file."""
        if self.session_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mix_agents_session_{timestamp}.txt"
            self.session_file = os.path.join(self.output_folder, filename)
            logging.info(f"Created new session file: {self.session_file}")

        try:
            with open(self.session_file, "a", encoding="utf-8") as f:
                f.write(f"User: {interaction['user_prompt']}\n\n")
                for response in interaction['model_responses']:
                    f.write(f"{response['model']} response:\n{response['response']}\n\n")
                f.write(f"Aggregated response:\n{interaction['aggregated_response']}\n\n")
                f.write("-" * 80 + "\n\n")
            logging.info(f"Interaction saved to {self.session_file}")
        except IOError as e:
            logging.error(f"Error saving interaction to file: {str(e)}")

    def run(self):
        print(info("Starting MixAgents. Type 'exit' to end the conversation."))
        while True:
            user_prompt = input(user_input("You: "))
            if user_prompt.lower() == 'exit':
                print(success("Thank you for using MixAgents. Goodbye!"))
                break
            
            try:
                responses = asyncio.run(self.get_responses(user_prompt))
                for model, response in responses:
                    print(info(f"{model} response: {response}"))
                
                aggregated_response = self.aggregate_responses(user_prompt, responses)
                print(llm_response(f"Aggregated response: {aggregated_response}"))
                
                # Log the interaction
                interaction = {
                    "user_prompt": user_prompt,
                    "model_responses": [{"model": model, "response": response} for model, response in responses],
                    "aggregated_response": aggregated_response
                }
                self.session_log.append(interaction)
                self.save_interaction(interaction)
            
            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                print(error(f"An error occurred: {str(e)}"))

def run_mix_agents(worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, manager_erag_api: EragAPI = None):
    mix_agents = MixAgents(worker_erag_api, supervisor_erag_api, manager_erag_api)
    mix_agents.run()