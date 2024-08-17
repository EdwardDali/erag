# Standard library imports
import sqlite3
import os
from collections import deque

# Third-party imports
from sentence_transformers import SentenceTransformer

# Local imports
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight

class Talk2SD:
    def __init__(self, erag_api):
        self.erag_api = erag_api
        self.db_path = settings.structured_data_db
        self.schema = self.fetch_schema()
        self.conversation_history = []
        self.conversation_context = deque(maxlen=settings.conversation_context_size * 2)
        self.embedding_model = SentenceTransformer(settings.sentence_transformer_model)
        self.system_prompt = self.generate_system_prompt()

    def fetch_schema(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                schema[table_name] = [
                    {"name": column[1], "type": column[2]}
                    for column in columns
                ]
            
            conn.close()
            return schema
        except Exception as e:
            print(error(f"Error fetching schema: {str(e)}"))
            return {}

    def generate_system_prompt(self):
        example_queries = """
        1. SELECT * FROM your_table_name LIMIT 5;
        2. SELECT COUNT(*) FROM your_table_name;
        3. SELECT column1, column2 FROM your_table_name WHERE condition;
        4. SELECT DISTINCT column FROM your_table_name;
        5. SELECT column, COUNT(*) FROM your_table_name GROUP BY column;
        6. SELECT t1.column, t2.column FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id;
        7. SELECT column, AVG(numeric_column) FROM your_table_name GROUP BY column HAVING AVG(numeric_column) > value;
        """

        return f"""You are an AI assistant designed to help with SQL queries and data analysis. Follow these steps when interacting with the database:

        1. Create a syntactically correct SQLite query based on the user's request.
        2. Ensure that you generate only a single SQL statement. Multiple statements are not allowed.
        3. Limit your query results to at most 5 rows unless specified otherwise.
        4. Only query for relevant columns, not all columns from a table.
        5. Do not make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        6. When responding to queries, focus solely on the SQL results. Do not add any storytelling or unnecessary elaboration.
        7. Provide concise, factual responses based on the data returned by the SQL query.

        Available tables and their schemas:
        {self.format_schema_for_prompt()}

        Here are some example SQLite queries of increasing complexity:
        {example_queries}

        Remember to use these steps for every database interaction and always provide a single SQL statement."""

    def format_schema_for_prompt(self):
        formatted_schema = ""
        for table, columns in self.schema.items():
            formatted_schema += f"Table: {table}\n"
            for column in columns:
                formatted_schema += f"  - {column['name']} ({column['type']})\n"
            formatted_schema += "\n"
        return formatted_schema

    def run(self):
        print(info("Starting Talk2SD. Type 'exit' to end the conversation or 'clear' to clear conversation history."))
        self.print_schema_overview()
        while True:
            user_input = input(highlight("User: "))
            if user_input.lower() == 'exit':
                print(info("Exiting Talk2SD."))
                break
            elif user_input.lower() == 'clear':
                self.conversation_history.clear()
                self.conversation_context.clear()
                print(info("Conversation history and context cleared."))
                continue

            sql_query, result = self.generate_and_execute_query(user_input)

            if result is not None:
                print(info(f"Generated SQL query: {sql_query}"))
                print(info(f"Query result: {result}"))
                response = self.generate_response(user_input, sql_query, result)
                print(success(f"AI: {response}"))
                self.update_conversation_context(user_input, response)
            else:
                print(error("Failed to generate a valid SQL query after multiple attempts."))

    def print_schema_overview(self):
        print(info("Database Schema Overview:"))
        for table, columns in self.schema.items():
            print(highlight(f"Table: {table}"))
            for column in columns:
                print(info(f"  - {column['name']} ({column['type']})"))
            print()

    def generate_and_execute_query(self, user_input, max_attempts=3):
        for attempt in range(max_attempts):
            sql_query = self.generate_sql_query(user_input, attempt)
            if sql_query:
                result = self.execute_sql_query(sql_query)
                if result is not None:
                    return sql_query, result
            print(warning(f"Attempt {attempt + 1} failed. Trying again..."))
        return None, None

    def generate_sql_query(self, user_input, attempt):
        context = "\n".join(self.conversation_context)
        prompt = f"""
        {self.system_prompt}

        Conversation context:
        {context}

        User question: {user_input}
        Attempt: {attempt + 1}
        
        IMPORTANT: Provide ONLY the SQL query as your response, without any additional text, explanation, or markdown formatting.
        Ensure it is a single statement query.
        If this is not the first attempt, please try to improve upon the previous query.
        SQL query:
        """
        response = self.erag_api.chat([{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}])
        return response.strip() if response else None

    def execute_sql_query(self, sql_query):
        try:
            clean_query = sql_query.replace('```sql', '').replace('```', '').strip()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(clean_query)
            result = cursor.fetchall()
            conn.close()
            return result
        except Exception as e:
            print(error(f"Error executing SQL query: {str(e)}"))
            return None

    def generate_response(self, user_input, sql_query, result):
        prompt = f"""
        User question: {user_input}
        SQL query: {sql_query}
        Query result: {result}

        Based on the SQL query result, generate a concise and factual response that directly answers the user's question.
        Do not add any storytelling or unnecessary elaboration. Focus solely on the data returned by the SQL query.
        If the result is empty or None, state that no data was found.
        
        Response:
        """
        response = self.erag_api.chat([{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}])
        return response.strip() if response else None

    def update_conversation_context(self, user_input: str, assistant_response: str):
        self.conversation_context.append(f"User: {user_input}")
        self.conversation_context.append(f"Assistant: {assistant_response}")
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

        if len(self.conversation_history) > settings.max_history_length * 3:
            self.conversation_history = self.conversation_history[-settings.max_history_length * 3:]

    def encode_text(self, text):
        return self.embedding_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
