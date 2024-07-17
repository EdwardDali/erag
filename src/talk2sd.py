import sqlite3
import os
from collections import deque
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from sentence_transformers import SentenceTransformer

class Talk2SD:
    def __init__(self, erag_api):
        self.erag_api = erag_api
        self.db_path = settings.structured_data_db
        self.schema = self.fetch_schema()
        self.conversation_history = []
        self.conversation_context = deque(maxlen=settings.conversation_context_size * 2)
        self.embedding_model = SentenceTransformer(settings.sentence_transformer_model)

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
                schema[table_name] = [column[1] for column in columns]
            
            conn.close()
            return schema
        except Exception as e:
            print(error(f"Error fetching schema: {str(e)}"))
            return {}

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
                response = self.generate_response(user_input, sql_query, result)
                print(success(f"AI: {response}"))
                self.update_conversation_context(user_input, response)
            else:
                print(error("Failed to generate a valid SQL query."))

    def print_schema_overview(self):
        print(info("Database Schema Overview:"))
        for table, columns in self.schema.items():
            print(highlight(f"Table: {table}"))
            print(info(f"Columns: {', '.join(columns)}"))
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
        Based on the user's question and the conversation context, generate an appropriate SQL query for SQLite.
        The query should be safe and not allow any harmful operations.
        Database schema: {self.schema}
        
        Conversation context:
        {context}

        Some helpful SQLite-specific queries:
        - To get table names: SELECT name FROM sqlite_master WHERE type='table';
        - To get column info for a table: PRAGMA table_info(table_name);
        - To count rows in a table: SELECT COUNT(*) FROM table_name;

        User question: {user_input}
        Attempt: {attempt + 1}
        
        IMPORTANT: Provide ONLY the SQL query as your response, without any additional text, explanation, or markdown formatting.
        If this is not the first attempt, please try to improve upon the previous query.
        SQL query:
        """
        response = self.erag_api.chat([{"role": "user", "content": prompt}])
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
        context = "\n".join(self.conversation_context)
        prompt = f"""
        Conversation context:
        {context}

        User question: {user_input}
        SQL query: {sql_query}
        Query result: {result}

        Based on the conversation context, user's question, and the SQL query result, generate a natural language response.
        Response:
        """
        response = self.erag_api.chat([{"role": "user", "content": prompt}])
        return response.strip() if response else None

    def update_conversation_context(self, user_input: str, assistant_response: str):
        self.conversation_context.append(f"User: {user_input}")
        self.conversation_context.append(f"Assistant: {assistant_response}")
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

        if len(self.conversation_history) > settings.max_history_length * 2:
            self.conversation_history = self.conversation_history[-settings.max_history_length * 2:]

    def encode_text(self, text):
        return self.embedding_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
