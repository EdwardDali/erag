# -*- coding: utf-8 -*-

import sys
import os
import json
from run_model import ANSIColor
from openai import OpenAI

class KnolCreator:
    OLLAMA_MODEL = "phi3:instruct"
    TEMPERATURE = 0.1

    def __init__(self, api_type: str):
        self.client = self.configure_api(api_type)

    @staticmethod
    def configure_api(api_type: str) -> OpenAI:
        if api_type == "ollama":
            return OpenAI(base_url='http://localhost:11434/v1', api_key='phi3:instruct')
        elif api_type == "llama":
            return OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        else:
            raise ValueError("Invalid API type")

    def create_knol(self, subject: str) -> dict:
        system_message = """You are a knowledgeable assistant that creates structured, comprehensive, and detailed knowledge entries on various subjects. When given a topic, provide an exhaustive explanation covering key aspects, sub-topics, and important details. Structure the information as a JSON object with the following format:

{
    "title": "Knol: [Subject]",
    "content": [
        {
            "topic": "Main Topic 1",
            "subtopics": [
                {
                    "title": "Subtopic 1.1",
                    "points": [
                        "Key point 1",
                        "Key point 2",
                        "Key point 3",
                        "Key point 4",
                        "Key point 5"
                    ]
                },
                {
                    "title": "Subtopic 1.2",
                    "points": [
                        "Relevant information 1",
                        "Relevant information 2",
                        "Relevant information 3",
                        "Relevant information 4",
                        "Relevant information 5"
                    ]
                }
            ]
        },
        {
            "topic": "Main Topic 2",
            "subtopics": [
                {
                    "title": "Subtopic 2.1",
                    "points": [
                        "Important detail 1",
                        "Important detail 2",
                        "Important detail 3",
                        "Important detail 4",
                        "Important detail 5"
                    ]
                }
            ]
        }
    ]
}

Ensure that the structure is consistent and the information is detailed and accurate. Aim to provide at least 5 points for each subtopic, but you can include more if necessary."""

        user_message = f"Create a structured, comprehensive knowledge entry about {subject}. Include main topics, subtopics, and at least 5 key points with detailed information for each subtopic."

        try:
            response = self.client.chat.completions.create(
                model=self.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.TEMPERATURE
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return {"error": f"An error occurred while creating the knol: {str(e)}"}

    def improve_knol(self, knol: dict) -> dict:
        system_message = """You are a knowledgeable assistant that improves and expands existing knowledge entries. Given a JSON object representing a knol, enhance it by adding more details, restructuring if necessary, and ensuring comprehensive coverage of the subject. Maintain the following JSON structure:

{
    "title": "Knol: [Subject]",
    "content": [
        {
            "topic": "Main Topic",
            "subtopics": [
                {
                    "title": "Subtopic",
                    "points": [
                        "Detailed point 1",
                        "Detailed point 2",
                        "Detailed point 3",
                        "Detailed point 4",
                        "Detailed point 5",
                        "Detailed point 6",
                        "Detailed point 7"
                    ]
                }
            ]
        }
    ]
}

Feel free to add new topics, subtopics, or points, and reorganize the structure if it improves the overall quality and comprehensiveness of the knol. Aim to provide at least 7 points for each subtopic, but you can include more if necessary."""

        user_message = f"Improve and expand the following knol by adding more details, ensuring at least 7 points per subtopic, and restructuring if necessary:\n\n{json.dumps(knol, indent=2)}"

        try:
            response = self.client.chat.completions.create(
                model=self.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.TEMPERATURE
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return knol  # Return the original knol if improvement fails

    def add_complexity_levels(self, knol: dict) -> dict:
        system_message = """You are a knowledgeable assistant that adds explanations at different complexity levels to existing knowledge entries. Given a JSON object representing a knol, add a new section called "complexity_levels" that explains the main subject at three different levels of complexity. Maintain the following JSON structure:

{
    "title": "Knol: [Subject]",
    "content": [ ... ],
    "complexity_levels": {
        "beginner": "Explanation for a 5-year-old or complete beginner",
        "intermediate": "Explanation for someone with basic understanding of the concept",
        "expert": "Detailed explanation for an expert in the field"
    }
}

Ensure that each explanation is tailored to the appropriate audience and provides a comprehensive understanding of the subject at that level."""

        user_message = f"Add explanations at three different complexity levels (beginner, intermediate, expert) to the following knol:\n\n{json.dumps(knol, indent=2)}"

        try:
            response = self.client.chat.completions.create(
                model=self.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.TEMPERATURE
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return knol  # Return the original knol if adding complexity levels fails

def run_knol_creator(api_type: str):
    creator = KnolCreator(api_type)
    print(f"{ANSIColor.YELLOW.value}Welcome to the Knol Creation System. Type 'exit' to quit.{ANSIColor.RESET.value}")

    while True:
        user_input = input(f"{ANSIColor.YELLOW.value}Which subject do you want to create a knol about? {ANSIColor.RESET.value}").strip()

        if user_input.lower() == 'exit':
            print(f"{ANSIColor.NEON_GREEN.value}Thank you for using the Knol Creation System. Goodbye!{ANSIColor.RESET.value}")
            break

        if not user_input:
            print(f"{ANSIColor.PINK.value}Please enter a valid subject.{ANSIColor.RESET.value}")
            continue

        print(f"{ANSIColor.CYAN.value}Creating initial knol...{ANSIColor.RESET.value}")
        initial_knol = creator.create_knol(user_input)

        print(f"{ANSIColor.CYAN.value}Improving and expanding the knol...{ANSIColor.RESET.value}")
        improved_knol = creator.improve_knol(initial_knol)

        print(f"{ANSIColor.CYAN.value}Adding complexity levels...{ANSIColor.RESET.value}")
        final_knol = creator.add_complexity_levels(improved_knol)

        print(f"{ANSIColor.NEON_GREEN.value}Knol created, improved, and complexity levels added:{ANSIColor.RESET.value}")
        print(json.dumps(final_knol, indent=2))

        # Save the knol to a JSON file
        filename = f"knol_{user_input.replace(' ', '_')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(final_knol, f, indent=2, ensure_ascii=False)

        print(f"{ANSIColor.CYAN.value}Knol saved as {filename}{ANSIColor.RESET.value}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        run_knol_creator(api_type)
    else:
        print("Error: No API type provided.")
        print("Usage: python create_knol.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
