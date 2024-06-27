# -*- coding: utf-8 -*-
import sys
import os
import logging
from run_model import ANSIColor
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    def save_iteration(self, content: str, stage: str, subject: str):
        filename = f"knol_{subject.replace(' ', '_')}_{stage}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Saved {stage} knol as {filename}")
        except IOError as e:
            logging.error(f"Error saving {stage} knol to file: {str(e)}")
            
    def create_knol(self, subject: str) -> str:
        system_message = f"""You are a knowledgeable assistant that creates structured, comprehensive, and detailed knowledge entries on specific subjects. When given a topic, provide an exhaustive explanation covering key aspects, sub-topics, and important details. Structure the information as follows:

Title: Knol: {subject}

1. [Main Topic 1]
1.1. [Subtopic 1.1]
• [Key point 1]
• [Key point 2]
• [Key point 3]
• [Key point 4]
• [Key point 5]

1.2. [Subtopic 1.2]
• [Relevant information 1]
• [Relevant information 2]
• [Relevant information 3]
• [Relevant information 4]
• [Relevant information 5]

2. [Main Topic 2]
...

Ensure that the structure is consistent and the information is detailed and accurate. Aim to provide at least 5 points for each subtopic, but you can include more if necessary. Stay strictly on the topic of {subject} and do not include information about unrelated subjects."""

        user_message = f"Create a structured, comprehensive knowledge entry about {subject}. Include main topics, subtopics, and at least 5 key points with detailed information for each subtopic. Focus exclusively on {subject}."

        try:
            response = self.client.chat.completions.create(
                model=self.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.TEMPERATURE
            )
            content = response.choices[0].message.content
            self.save_iteration(content, "initial", subject)
            return content
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}")
            error_content = f"An error occurred while creating the knol: {str(e)}"
            self.save_iteration(error_content, "initial_error", subject)
            return error_content

    def improve_knol(self, knol: str, subject: str) -> str:
        system_message = f"""You are a knowledgeable assistant that improves and expands existing knowledge entries. Given a structured text representing a knol about {subject}, enhance it by adding more details, restructuring if necessary, and ensuring comprehensive coverage of the subject. Maintain the following structure:

Title: Knol: {subject}

1. [Main Topic]
1.1. [Subtopic]
• [Detailed point 1]
• [Detailed point 2]
• [Detailed point 3]
• [Detailed point 4]
• [Detailed point 5]
• [Detailed point 6]
• [Detailed point 7]

Feel free to add new topics, subtopics, or points, and reorganize the structure if it improves the overall quality and comprehensiveness of the knol. Aim to provide at least 7 points for each subtopic, but you can include more if necessary. Stay strictly on the topic of {subject} and do not include information about unrelated subjects."""

        user_message = f"Improve and expand the following knol about {subject} by adding more details, ensuring at least 7 points per subtopic, and restructuring if necessary. Focus exclusively on {subject}:\n\n{knol}"

        try:
            response = self.client.chat.completions.create(
                model=self.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.TEMPERATURE
            )
            improved_content = response.choices[0].message.content
            self.save_iteration(improved_content, "improved", subject)
            return improved_content
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}")
            error_content = f"An error occurred while improving the knol: {str(e)}"
            self.save_iteration(error_content, "improved_error", subject)
            return knol  # Return the original knol if improvement fails


    def generate_questions(self, knol: str, subject: str) -> str:
        system_message = f"""You are a knowledgeable assistant tasked with creating diverse and thought-provoking questions based on a given knowledge entry (knol) about {subject}. Generate 8 questions that cover different aspects of the knol, ranging from factual recall to critical thinking and analysis. Ensure the questions are clear, concise, and directly related to the content of the knol."""

        user_message = f"""Based on the following knol about {subject}, generate 8 diverse questions:

{knol}

Please provide 8 questions that:
1. Cover different aspects and topics from the knol
2. Include a mix of question types (e.g., factual, analytical, comparative)
3. Are clear and concise
4. Are directly related to the content of the knol
5. Encourage critical thinking and deeper understanding of the subject

Format the output as a numbered list of questions."""

        try:
            response = self.client.chat.completions.create(
                model=self.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.TEMPERATURE
            )
            questions = response.choices[0].message.content
            self.save_iteration(questions, "q", subject)
            return questions
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}")
            error_content = f"An error occurred while generating questions: {str(e)}"
            self.save_iteration(error_content, "q_error", subject)
            return error_content


    def run_knol_creator(self):
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
            initial_knol = self.create_knol(user_input)

            print(f"{ANSIColor.CYAN.value}Improving and expanding the knol...{ANSIColor.RESET.value}")
            improved_knol = self.improve_knol(initial_knol, user_input)

            print(f"{ANSIColor.CYAN.value}Generating questions based on the improved knol...{ANSIColor.RESET.value}")
            questions = self.generate_questions(improved_knol, user_input)

            print(f"{ANSIColor.NEON_GREEN.value}Knol creation process completed.{ANSIColor.RESET.value}")
            print(f"{ANSIColor.CYAN.value}You can find the results in files with '_initial', '_improved', and '_q' suffixes.{ANSIColor.RESET.value}")

            # Print the improved knol content and questions
            print(f"\n{ANSIColor.NEON_GREEN.value}Improved Knol Content:{ANSIColor.RESET.value}")
            print(improved_knol)
            print(f"\n{ANSIColor.NEON_GREEN.value}Generated Questions:{ANSIColor.RESET.value}")
            print(questions)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        creator = KnolCreator(api_type)
        creator.run_knol_creator()
    else:
        print("Error: No API type provided.")
        print("Usage: python create_knol.py <api_type>")
        print("Available API types: ollama, llama")
        sys.exit(1)
