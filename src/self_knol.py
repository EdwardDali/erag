import os
import logging
import re
from typing import List, Tuple
from src.settings import settings
from src.api_model import EragAPI, create_erag_api
from src.look_and_feel import success, info, warning, error, colorize, MAGENTA, RESET, user_input as color_user_input, llm_response as color_llm_response

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SelfKnolCreator:
    def __init__(self, worker_erag_api: EragAPI, supervisor_erag_api: EragAPI, manager_erag_api: EragAPI = None):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.manager_erag_api = manager_erag_api
        os.makedirs(settings.output_folder, exist_ok=True)

    def save_iteration(self, content: str, stage: str, subject: str, iteration: int):
        filename = f"self_knol_{subject.replace(' ', '_')}_{stage}_iteration_{iteration}.txt"
        file_path = os.path.join(settings.output_folder, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(success(f"Saved {stage} self knol (iteration {iteration}) as {file_path}"))
        except IOError as e:
            logging.error(error(f"Error saving {stage} self knol to file: {str(e)}"))

    def create_knol(self, subject: str, iteration: int, previous_knol: str = "", manager_review: str = "") -> str:
        system_message = f"""You are a knowledgeable assistant that creates and improves structured, comprehensive, and detailed knowledge entries on specific subjects. Your task is to create or enhance a knol (knowledge entry) about {subject}.

Structure the information as follows:

Title: Self Knol: {subject}

1. [Main Topic 1]
1.1. [Subtopic 1.1]
- [Key point 1]
- [Key point 2]
...

Ensure that the structure is consistent and the information is detailed and accurate. Aim to provide at least 5 points for each subtopic, but you can include more if necessary. Stay strictly on the topic of {subject} and do not include information about unrelated subjects. Use only your own knowledge without referencing external sources."""

        if iteration == 1:
            query = f"Create a structured, comprehensive knowledge entry about {subject}. Include main topics, subtopics, and at least 5 key points with detailed information for each subtopic. Focus exclusively on {subject} using only your own knowledge."
        else:
            query = f"""Improve and expand the following knowledge entry about {subject}, taking into account the manager's review. 
            Your task is to create a new, enhanced version of the knol, not just to provide comments.
            Add more details, ensure at least 5 points per subtopic, and restructure if necessary. 
            Focus on addressing the specific feedback and areas of improvement mentioned in the review.
            Incorporate all the valuable information from the previous version while expanding and improving upon it.

Previous Knol:
{previous_knol}

Manager's Review:
{manager_review}

Please provide a completely updated and improved version of the knol, incorporating all valuable information from the previous version and addressing the manager's feedback."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]

        content = self.worker_erag_api.chat(messages, temperature=settings.temperature)
        self.save_iteration(content, "initial", subject, iteration)
        return content

    def improve_knol(self, knol: str, subject: str, iteration: int, manager_review: str = "") -> str:
        system_message = f"""You are a supervisory assistant tasked with improving and expanding an existing knowledge entry (knol) about {subject}. Your role is to enhance the knol by adding more details, restructuring if necessary, and ensuring comprehensive coverage of the subject. Pay special attention to:

1. Accuracy and depth of information
2. Completeness of coverage
3. Clarity and coherence of presentation
4. Logical structure and flow
5. Appropriate depth of detail

Maintain the following structure, but feel free to add, modify, or reorganize content as needed to improve the overall quality of the knol:

Title: Self Knol: {subject}

1. [Main Topic]
1.1. [Subtopic]
- [Detailed point 1]
- [Detailed point 2]
...

Aim to provide at least 7 points for each subtopic, but you can include more if necessary. Stay strictly on the topic of {subject} and do not include information about unrelated subjects. Use only your own knowledge without referencing external sources."""

        query = f"""Improve and expand the following knol about {subject} by adding more details, ensuring at least 7 points per subtopic, and restructuring if necessary. Focus exclusively on {subject} using only your own knowledge.

Your task is to create a new, enhanced version of the knol, not just to provide comments.
Incorporate all the valuable information from the previous version while expanding and improving upon it.

Original Knol:
{knol}

"""
        if manager_review:
            query += f"""
Please pay special attention to the following manager's review from the previous iteration and address the points raised:

Manager's Review:
{manager_review}

Ensure that you create a fully updated and improved version of the knol, addressing all the feedback and suggestions provided in the manager's review.
"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]

        improved_content = self.supervisor_erag_api.chat(messages, temperature=settings.temperature)
        self.save_iteration(improved_content, "improved", subject, iteration)
        return improved_content

    def manager_review(self, knol: str, subject: str, iteration: int) -> Tuple[str, float]:
        if self.manager_erag_api is None:
            print(info("Manager review skipped as no manager model was selected."))
            return "Manager review skipped.", 10.0
        
        manager_system_message = f"""You are a managerial assistant tasked with critically reviewing and evaluating a knowledge entry (knol) about {subject}. Your role is to:

        1. Evaluate the overall quality, comprehensiveness, and accuracy of the knol
        2. Identify specific areas for improvement
        3. Pose questions that could enhance the knol's content
        4. Provide constructive feedback
        5. Rate the knol on a scale of 1 to 10

        Be thorough and critical in your evaluation, aiming to improve the knol to the highest possible standard. 
        Your review should be constructive but demanding, pointing out both strengths and weaknesses.

        IMPORTANT: You must provide a numerical grade between 1 and 10 at the end of your review, formatted as follows:
        GRADE: [Your numerical grade]

        This grade should reflect your critical evaluation, where:
        1-3: Poor quality, major revisions needed
        4-5: Below average, significant improvements required
        6-7: Average, several areas need improvement
        8-9: Good quality, minor improvements needed
        10: Excellent, meets the highest standards"""

        manager_user_input = f"""Please review the following knol about {subject} and provide:

        1. An overall evaluation (strengths and weaknesses)
        2. Specific areas for improvement
        3. Questions that could enhance the content
        4. Constructive feedback for the next iteration
        5. A rating on a scale of 1 to 10

        Remember to be critical and demanding in your evaluation. We are aiming for the highest possible quality.

        Knol Content:
        {knol}

        End your review with a numerical grade formatted as: GRADE: [Your numerical grade]"""

        manager_messages = [
            {"role": "system", "content": manager_system_message},
            {"role": "user", "content": manager_user_input}
        ]
        review = self.manager_erag_api.chat(manager_messages, temperature=settings.temperature)

        # Extract rating from the review
        grade_match = re.search(r'GRADE:\s*(\d+(?:\.\d+)?)', review)
        if grade_match:
            rating = float(grade_match.group(1))
        else:
            print(warning("No grade found in the manager's review. Assigning a default grade of 5."))
            rating = 5.0

        self.save_iteration(review, "manager_review", subject, iteration)
        return review, rating

    def run_self_knol_creator(self):
        print(info("Welcome to the Self Knol Creation System. Type 'exit' to quit."))
        
        while True:
            user_input = input(color_user_input("Which subject do you want to create a self knol about? ")).strip()

            if user_input.lower() == 'exit':
                print(success("Thank you for using the Self Knol Creation System. Goodbye!"))
                break

            if not user_input:
                print(error("Please enter a valid subject."))
                continue

            iteration = 1
            previous_knol = ""
            previous_review = ""

            while True:
                print(info(f"Creating self knol about {user_input} (Iteration {iteration})..."))
                initial_knol = self.create_knol(user_input, iteration, previous_knol, previous_review)

                print(info("Improving and expanding the self knol..."))
                improved_knol = self.improve_knol(initial_knol, user_input, iteration, previous_review)

                print(info("Manager review in progress..."))
                manager_review, rating = self.manager_review(improved_knol, user_input, iteration)
                
                print(f"\n{success('Manager Review:')}")
                print(color_llm_response(manager_review))
                print(f"\n{success(f'Manager Rating: {rating}/10')}")

                if rating >= 8.0:
                    print(success(f"Self Knol creation process completed for {user_input} with a satisfactory rating."))
                    break
                else:
                    print(info(f"The knol did not meet the required standard (8.0). Current rating: {rating}/10"))
                    print(info("Proceeding with the next iteration..."))
                    iteration += 1
                    previous_knol = improved_knol
                    previous_review = manager_review

            print(info("You can find the results in files with '_initial', '_improved', and '_manager_review' suffixes."))

            print(f"\n{success('Final Improved Self Knol Content:')}")
            print(color_llm_response(improved_knol))

    

def main(api_type: str):
    worker_erag_api = create_erag_api(api_type)
    supervisor_erag_api = create_erag_api(api_type)
    manager_erag_api = create_erag_api(api_type)
    creator = SelfKnolCreator(worker_erag_api, supervisor_erag_api, manager_erag_api)
    creator.run_self_knol_creator()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        main(api_type)
    else:
        print(error("Error: API type not provided."))
        print(warning("Usage: python src/self_knol.py <api_type>"))
        print(info("Available API types: ollama, llama, groq"))
        sys.exit(1)
