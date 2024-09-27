import sqlite3
import csv
import os
import shutil
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight

class DataQualityChecker:
    def __init__(self, erag_api, db_path):
        self.erag_api = erag_api
        self.db_path = db_path
        self.schema = self.fetch_schema()
        self.output_folder = os.path.join(os.path.dirname(db_path), "data_quality_output")
        os.makedirs(self.output_folder, exist_ok=True)
        self.marked_db_path = None

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

    def create_marked_database(self):
        marked_db_path = os.path.join(self.output_folder, "marked_" + os.path.basename(self.db_path))
        shutil.copy(self.db_path, marked_db_path)
        
        conn = sqlite3.connect(marked_db_path)
        cursor = conn.cursor()

        for table_name, columns in self.schema.items():
            # Add error marking columns
            for column in columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column['name']}_error TEXT;")

        conn.commit()
        conn.close()

        return marked_db_path

    def run(self):
        print(info("Starting Data Quality Check..."))
        self.marked_db_path = self.create_marked_database()
        for table_name, columns in self.schema.items():
            print(highlight(f"Checking table: {table_name}"))
            self.check_table(table_name, columns)
        self.export_marked_data_to_csv()
        print(success(f"Data Quality Check completed. Results saved in CSV files, marked database: {self.marked_db_path}, and exported to CSV files."))

    def check_table(self, table_name, columns):
        original_conn = sqlite3.connect(self.db_path)
        marked_conn = sqlite3.connect(self.marked_db_path)
        original_cursor = original_conn.cursor()
        marked_cursor = marked_conn.cursor()

        errors = []

        # Check for missing values
        for column in columns:
            query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NULL OR {column['name']} = '';"
            original_cursor.execute(query)
            null_rows = original_cursor.fetchall()
            null_count = len(null_rows)
            if null_count > 0:
                errors.append({
                    "table": table_name,
                    "column": column['name'],
                    "error_type": "Missing Values",
                    "count": null_count
                })
                for row in null_rows:
                    marked_cursor.execute(f"UPDATE {table_name} SET {column['name']}_error = 'Missing Value' WHERE rowid = ?", (row[0],))

        # Check for data type mismatches (for numeric columns)
        for column in columns:
            if column['type'].lower() in ['integer', 'real', 'float', 'double']:
                query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE typeof({column['name']}) != '{column['type'].lower()}';"
                original_cursor.execute(query)
                mismatch_rows = original_cursor.fetchall()
                mismatch_count = len(mismatch_rows)
                if mismatch_count > 0:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Data Type Mismatch",
                        "count": mismatch_count
                    })
                    for row in mismatch_rows:
                        marked_cursor.execute(f"UPDATE {table_name} SET {column['name']}_error = 'Type Mismatch' WHERE rowid = ?", (row[0],))

        marked_conn.commit()
        original_conn.close()
        marked_conn.close()

        # Save errors summary to CSV
        if errors:
            output_file = os.path.join(self.output_folder, f"{table_name}_data_quality_errors.csv")
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
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
        with open(interpretation_file, 'w', encoding='utf-8') as f:
            f.write(response)
        
        print(info(f"AI interpretation for {table_name} saved to {interpretation_file}"))

    def export_marked_data_to_csv(self):
        conn = sqlite3.connect(self.marked_db_path)
        cursor = conn.cursor()

        for table_name in self.schema.keys():
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            if rows:
                headers = [description[0] for description in cursor.description]
                output_file = os.path.join(self.output_folder, f"{table_name}_marked_data.csv")
                
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)
                    for row in rows:
                        writer.writerow([str(cell) if cell is not None else '' for cell in row])
                
                print(info(f"Marked data for {table_name} exported to {output_file}"))

        conn.close()