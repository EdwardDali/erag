# Standard library imports
import asyncio
from typing import List, Dict

# Local imports
from src.api_model import EragAPI, create_erag_api
from src.settings import settings
from src.look_and_feel import error, success, info, llm_response, user_input

class MixAgents:
    def __init__(self, worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, manager_erag_api: EragAPI = None):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.manager_erag_api = manager_erag_api

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

    def aggregate_responses(self, responses: List[tuple]):
        """Aggregate responses using the supervisor model."""
        aggregator_system_prompt = """You have been provided with a set of responses from various models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. Critically evaluate the information, recognizing potential biases or inaccuracies. Your response should offer a refined, accurate, and comprehensive reply to the instruction."""
        
        aggregator_messages = [
            {"role": "system", "content": aggregator_system_prompt},
            {"role": "user", "content": f"Responses from models: {', '.join([f'{model}: {response}' for model, response in responses])}"}
        ]
        
        return self.supervisor_erag_api.chat(aggregator_messages, temperature=settings.temperature)

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
                
                aggregated_response = self.aggregate_responses(responses)
                print(llm_response(f"Aggregated response: {aggregated_response}"))
            
            except Exception as e:
                print(error(f"An error occurred: {str(e)}"))

def run_mix_agents(worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, manager_erag_api: EragAPI = None):
    mix_agents = MixAgents(worker_erag_api, supervisor_erag_api, manager_erag_api)
    mix_agents.run()