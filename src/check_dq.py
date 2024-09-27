import sqlite3
import csv
import os
import shutil
import re
from datetime import datetime
from statistics import mean, stdev
import numpy as np
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
        self.total_checks = 10
        self.current_check = 0

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
            for column in columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column['name']}_error TEXT;")

        conn.commit()
        conn.close()

        return marked_db_path

    def run(self):
        print(info("Starting Data Quality Check..."))
        self.marked_db_path = self.create_marked_database()
        for table_name, columns in self.schema.items():
            print(highlight(f"\nAnalyzing table: {table_name}"))
            self.check_table(table_name, columns)
        self.export_marked_data_to_csv()
        print(success(f"Data Quality Check completed. Results saved in {self.output_folder}"))

    def check_table(self, table_name, columns):
        original_conn = sqlite3.connect(self.db_path)
        marked_conn = sqlite3.connect(self.marked_db_path)
        original_cursor = original_conn.cursor()
        marked_cursor = marked_conn.cursor()

        errors = []

        check_methods = [
            self.check_missing_values,
            self.check_data_type_mismatches,
            self.check_duplicate_records,
            self.check_inconsistent_formatting,
            self.check_outliers,
            self.check_whitespace,
            self.check_special_characters,
            self.check_inconsistent_capitalization,
            self.check_data_truncation,
            self.check_value_frequency
        ]

        for method in check_methods:
            self.current_check += 1
            progress = (self.current_check / self.total_checks) * 100
            print(info(f"Progress: {progress:.2f}% - Running {method.__name__}"))
            method(table_name, columns, original_cursor, marked_cursor, errors)

        marked_conn.commit()
        original_conn.close()
        marked_conn.close()

        self.save_errors_to_csv(table_name, errors)
        self.generate_ai_interpretation(table_name, errors)

    def check_missing_values(self, table_name, columns, original_cursor, marked_cursor, errors):
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

    def check_data_type_mismatches(self, table_name, columns, original_cursor, marked_cursor, errors):
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

    def check_duplicate_records(self, table_name, columns, original_cursor, marked_cursor, errors):
        column_names = ', '.join([c['name'] for c in columns])
        query = f"""
        SELECT {column_names}, COUNT(*) as cnt
        FROM {table_name}
        GROUP BY {column_names}
        HAVING cnt > 1
        """
        original_cursor.execute(query)
        duplicates = original_cursor.fetchall()
        if duplicates:
            errors.append({
                "table": table_name,
                "column": "All",
                "error_type": "Duplicate Records",
                "count": len(duplicates)
            })
            for dup in duplicates:
                conditions = ' AND '.join([f"{c['name']} = ?" for c in columns])
                marked_cursor.execute(f"UPDATE {table_name} SET error = 'Duplicate Record' WHERE {conditions}", dup[:-1])

    def check_inconsistent_formatting(self, table_name, columns, original_cursor, marked_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT DISTINCT {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL AND {column['name']} != '';"
                original_cursor.execute(query)
                distinct_values = original_cursor.fetchall()
                
                # Check for inconsistent date formats
                date_formats = set()
                for value in distinct_values:
                    try:
                        datetime.strptime(value[0], '%Y-%m-%d')
                        date_formats.add('%Y-%m-%d')
                    except ValueError:
                        try:
                            datetime.strptime(value[0], '%d/%m/%Y')
                            date_formats.add('%d/%m/%Y')
                        except ValueError:
                            pass
                
                if len(date_formats) > 1:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Inconsistent Date Formatting",
                        "count": len(distinct_values)
                    })
                    marked_cursor.execute(f"UPDATE {table_name} SET {column['name']}_error = 'Inconsistent Date Format' WHERE {column['name']} IS NOT NULL AND {column['name']} != ''")

    def check_outliers(self, table_name, columns, original_cursor, marked_cursor, errors):
        for column in columns:
            if column['type'].lower() in ['integer', 'real', 'float', 'double']:
                query = f"SELECT {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL;"
                original_cursor.execute(query)
                values = [row[0] for row in original_cursor.fetchall()]
                if values:
                    Q1 = np.percentile(values, 25)
                    Q3 = np.percentile(values, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = [v for v in values if v < lower_bound or v > upper_bound]
                    if outliers:
                        errors.append({
                            "table": table_name,
                            "column": column['name'],
                            "error_type": "Outliers",
                            "count": len(outliers)
                        })
                        marked_cursor.execute(f"UPDATE {table_name} SET {column['name']}_error = 'Outlier' WHERE {column['name']} < ? OR {column['name']} > ?", (lower_bound, upper_bound))

    def check_whitespace(self, table_name, columns, original_cursor, marked_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL AND ({column['name']} LIKE ' %' OR {column['name']} LIKE '% ' OR {column['name']} != TRIM({column['name']}));"
                original_cursor.execute(query)
                whitespace_rows = original_cursor.fetchall()
                if whitespace_rows:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Leading/Trailing Whitespace",
                        "count": len(whitespace_rows)
                    })
                    for row in whitespace_rows:
                        marked_cursor.execute(f"UPDATE {table_name} SET {column['name']}_error = 'Whitespace Issue' WHERE rowid = ?", (row[0],))

    def check_special_characters(self, table_name, columns, original_cursor, marked_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL;"
                original_cursor.execute(query)
                rows = original_cursor.fetchall()
                special_char_rows = [row for row in rows if any(not c.isalnum() and not c.isspace() for c in row[1])]
                if special_char_rows:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Special Characters",
                        "count": len(special_char_rows)
                    })
                    for row in special_char_rows:
                        marked_cursor.execute(f"UPDATE {table_name} SET {column['name']}_error = 'Special Characters' WHERE rowid = ?", (row[0],))

    def check_inconsistent_capitalization(self, table_name, columns, original_cursor, marked_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT DISTINCT {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL;"
                original_cursor.execute(query)
                distinct_values = original_cursor.fetchall()
                
                case_types = set()
                for value in distinct_values:
                    if value[0].islower():
                        case_types.add('lowercase')
                    elif value[0].isupper():
                        case_types.add('uppercase')
                    elif value[0].istitle():
                        case_types.add('titlecase')
                    else:
                        case_types.add('mixedcase')
                
                if len(case_types) > 1:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Inconsistent Capitalization",
                        "count": len(distinct_values)
                    })
                    marked_cursor.execute(f"UPDATE {table_name} SET {column['name']}_error = 'Inconsistent Capitalization' WHERE {column['name']} IS NOT NULL")

    def check_data_truncation(self, table_name, columns, original_cursor, marked_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT MAX(LENGTH({column['name']})) FROM {table_name};"
                original_cursor.execute(query)
                max_length = original_cursor.fetchone()[0]
                
                if max_length is not None and max_length > 0:
                    threshold = max_length * 0.9  # Consider as potentially truncated if length is 90% or more of max length
                    query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE LENGTH({column['name']}) >= ?;"
                    original_cursor.execute(query, (threshold,))
                    truncated_rows = original_cursor.fetchall()
                    if truncated_rows:
                        errors.append({
                            "table": table_name,
                            "column": column['name'],
                            "error_type": "Possible Data Truncation",
                            "count": len(truncated_rows)
                        })
                        for row in truncated_rows:
                            marked_cursor.execute(f"UPDATE {table_name} SET {column['name']}_error = 'Possible Truncation' WHERE rowid = ?", (row[0],))

    def check_value_frequency(self, table_name, columns, original_cursor, marked_cursor, errors):
        for column in columns:
            query = f"SELECT {column['name']}, COUNT(*) as frequency FROM {table_name} GROUP BY {column['name']} ORDER BY frequency DESC LIMIT 1;"
            original_cursor.execute(query)
            most_frequent = original_cursor.fetchone()
            
            if most_frequent:
                value, frequency = most_frequent
                total_rows = original_cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                frequency_percentage = (frequency / total_rows) * 100
                
                if frequency_percentage > 90:  # If a single value appears in more than 90% of rows
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "High Frequency Value",
                        "count": frequency,
                        "details": f"Value '{value}' appears in {frequency_percentage:.2f}% of rows"
                    })
                    marked_cursor.execute(f"UPDATE {table_name} SET {column['name']}_error = 'High Frequency Value' WHERE {column['name']} = ?", (value,))

    def save_errors_to_csv(self, table_name, errors):
        if errors:
            output_file = os.path.join(self.output_folder, f"{table_name}_data_quality_errors.csv")
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["table", "column", "error_type", "count", "details"])
                writer.writeheader()
                writer.writerows(errors)
            print(warning(f"Data quality issues found in {table_name}. Results saved to {output_file}"))
        else:
            print(success(f"No data quality issues found in {table_name}."))

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

if __name__ == "__main__":
    # Example usage
    from src.api_model import EragAPI  # Ensure this import is correct for your project structure

    db_path = "path/to/your/database.sqlite"
    erag_api = EragAPI()  # Initialize your API object here
    checker = DataQualityChecker(erag_api, db_path)
    checker.run()