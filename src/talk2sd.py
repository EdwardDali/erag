import sqlite3
import os
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight

class Talk2SD:
    def __init__(self, erag_api):
        self.erag_api = erag_api
        self.db_path = settings.structured_data_db
        self.schema = self.fetch_schema()
        self.create_query_history_table()

    def fetch_schema(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Fetch all table names
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

    def create_query_history_table(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT,
                    sql_query TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(error(f"Error creating query_history table: {str(e)}"))

    def run(self):
        print(info("Starting Talk2SD. Type 'exit' to end the conversation."))
        self.print_schema_overview()
        while True:
            user_input = input(highlight("User: "))
            if user_input.lower() == 'exit':
                print(info("Exiting Talk2SD."))
                break

            sql_query = self.search_query_history(user_input)
            if sql_query:
                print(info(f"Using previously successful query: {sql_query}"))
                result = self.execute_sql_query(sql_query)
            else:
                sql_query, result = self.generate_and_execute_query(user_input)

            if result is not None:
                response = self.generate_response(user_input, sql_query, result)
                print(success(f"AI: {response}"))
            else:
                print(error("Failed to generate a valid SQL query after multiple attempts."))

    def print_schema_overview(self):
        print(info("Database Schema Overview:"))
        for table, columns in self.schema.items():
            print(highlight(f"Table: {table}"))
            print(info(f"Columns: {', '.join(columns)}"))
        print()

    def search_query_history(self, user_input):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT sql_query FROM query_history WHERE question = ?", (user_input,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            print(error(f"Error searching query history: {str(e)}"))
            return None

    def generate_and_execute_query(self, user_input, max_attempts=3):
        for attempt in range(max_attempts):
            sql_query = self.generate_sql_query(user_input, attempt)
            if sql_query:
                result = self.execute_sql_query(sql_query)
                if result is not None:
                    self.store_successful_query(user_input, sql_query)
                    return sql_query, result
            print(warning(f"Attempt {attempt + 1} failed. Trying again..."))
        return None, None

    def generate_sql_query(self, user_input, attempt):
        prompt = f"""
        Based on the user's question, generate an appropriate SQL query for SQLite.
        The query should be safe and not allow any harmful operations.
        Database schema: {self.schema}

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
        prompt = f"""
        User question: {user_input}
        SQL query: {sql_query}
        Query result: {result}

        Based on the user's question and the SQL query result, generate a natural language response.
        Response:
        """
        response = self.erag_api.chat([{"role": "user", "content": prompt}])
        return response.strip() if response else None

    def store_successful_query(self, question, sql_query):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO query_history (question, sql_query) VALUES (?, ?)", (question, sql_query))
            conn.commit()
            conn.close()
            print(info("Successful query stored in history."))
        except Exception as e:
            print(error(f"Error storing successful query: {str(e)}"))
