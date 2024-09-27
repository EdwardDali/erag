import sqlite3
import csv
import os
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight

class DataQualityChecker:
    def __init__(self, erag_api, db_path):
        self.erag_api = erag_api
        self.db_path = db_path
        self.schema = self.fetch_schema()
        self.output_folder = os.path.join(os.path.dirname(db_path), "data_quality_output")
        os.makedirs(self.output_folder, exist_ok=True)

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

    def run(self):
        print(info("Starting Data Quality Check..."))
        for table_name, columns in self.schema.items():
            print(highlight(f"Checking table: {table_name}"))
            self.check_table(table_name, columns)
        print(success("Data Quality Check completed. Results saved in CSV files."))

    def check_table(self, table_name, columns):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        errors = []

        # Check for missing values
        for column in columns:
            query = f"SELECT COUNT(*) FROM {table_name} WHERE {column['name']} IS NULL OR {column['name']} = '';"
            cursor.execute(query)
            null_count = cursor.fetchone()[0]
            if null_count > 0:
                errors.append({
                    "table": table_name,
                    "column": column['name'],
                    "error_type": "Missing Values",
                    "count": null_count
                })

        # Check for data type mismatches (for numeric columns)
        for column in columns:
            if column['type'].lower() in ['integer', 'real', 'float', 'double']:
                query = f"SELECT COUNT(*) FROM {table_name} WHERE typeof({column['name']}) != '{column['type'].lower()}';"
                cursor.execute(query)
                mismatch_count = cursor.fetchone()[0]
                if mismatch_count > 0:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Data Type Mismatch",
                        "count": mismatch_count
                    })

        conn.close()

        # Save errors to CSV
        if errors:
            output_file = os.path.join(self.output_folder, f"{table_name}_data_quality_errors.csv")
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["table", "column", "error_type", "count"])
                writer.writeheader()
                writer.writerows(errors)
            print(warning(f"Data quality issues found in {table_name}. Results saved to {output_file}"))
        else:
            print(success(f"No data quality issues found in {table_name}."))

        # Generate AI interpretation
        self.generate_ai_interpretation(table_name, errors)

    def generate_ai_interpretation(self, table_name, errors):
        prompt = f"""
        Analyze the following data quality issues found in the table '{table_name}':

        {errors}

        Provide a concise interpretation of these issues, their potential impact on data analysis, 
        and suggestions for addressing them. Focus on the most critical issues if there are many.
        """

        response = self.erag_api.chat([{"role": "system", "content": "You are a data quality expert."},
                                       {"role": "user", "content": prompt}])

        interpretation_file = os.path.join(self.output_folder, f"{table_name}_ai_interpretation.txt")
        with open(interpretation_file, 'w') as f:
            f.write(response)
        
        print(info(f"AI interpretation for {table_name} saved to {interpretation_file}"))