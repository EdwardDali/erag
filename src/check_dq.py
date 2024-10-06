import sqlite3
import csv
import os
import shutil
import re
from datetime import datetime
import numpy as np
import pandas as pd
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.api_model import EragAPI

class DataQualityChecker:
    def __init__(self, erag_api, db_path, enable_ai_interpretation=False):
        self.erag_api = erag_api
        self.db_path = db_path
        self.output_folder = os.path.join(os.path.dirname(db_path), "data_quality_output")
        os.makedirs(self.output_folder, exist_ok=True)
        self.marked_db_path = None
        self.marked_conn = None
        self.marked_cursor = None
        self.total_checks = 20
        self.current_check = 0
        self.column_name_changes = {}
        self.schema = self.fetch_schema()
        self.error_types = [
            'Missing Value', 'Type Mismatch', 'Duplicate Record', 'Inconsistent Date Format',
            'Outlier', 'Whitespace Issue', 'Special Characters', 'Inconsistent Capitalization',
            'Possible Data Truncation', 'High Frequency Value', 'Suspicious Date Range',
            'Large Numeric Range', 'Very Short String', 'Very Long String',
            'Invalid Email Format', 'Non-unique Value', 'Invalid Foreign Key',
            'Date Inconsistency', 'Logical Relationship Violation', 'Pattern Mismatch'
        ]
        self.enable_ai_interpretation = enable_ai_interpretation

    def sanitize_column_name(self, column_name):
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', column_name)
        if sanitized_name[0].isdigit():
            sanitized_name = 'col_' + sanitized_name
        return sanitized_name

    def fetch_schema(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema = {}
            for table in tables:
                table_name = table[0]
                try:
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    schema[table_name] = []
                    for column in columns:
                        original_name = column[1]
                        sanitized_name = self.sanitize_column_name(original_name)
                        if original_name != sanitized_name:
                            self.column_name_changes[original_name] = sanitized_name
                            print(warning(f"Column name '{original_name}' has been changed to '{sanitized_name}'"))
                        schema[table_name].append({"name": sanitized_name, "type": column[2], "original_name": original_name})
                except sqlite3.OperationalError as e:
                    print(warning(f"Error fetching schema for table {table_name}: {str(e)}"))
                    continue
            
            conn.close()
            return schema
        except Exception as e:
            print(error(f"Error fetching schema: {str(e)}"))
            return {}

    def create_marked_database(self):
        marked_db_path = os.path.join(self.output_folder, "marked_" + os.path.basename(self.db_path))
        shutil.copy(self.db_path, marked_db_path)
        
        self.marked_conn = sqlite3.connect(marked_db_path)
        self.marked_cursor = self.marked_conn.cursor()

        for table_name, columns in self.schema.items():
            for column in columns:
                try:
                    self.marked_cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column['name']}_errors TEXT;")
                except sqlite3.OperationalError as e:
                    print(warning(f"Error adding error column to {table_name}.{column['name']}: {str(e)}"))
                    continue

        self.marked_conn.commit()
        return marked_db_path

    def mark_error(self, table_name, column_name, row_id, error_type):
        current_errors = self.marked_cursor.execute(
            f"SELECT {column_name}_errors FROM {table_name} WHERE rowid = ?",
            (row_id,)
        ).fetchone()[0]
        
        if current_errors and current_errors.strip():
            errors_set = set([e.strip().lower() for e in current_errors.split(',')])
        else:
            errors_set = set()

        errors_set.add(error_type.lower())

        new_errors = ', '.join(sorted(errors_set))
        self.marked_cursor.execute(
            f"UPDATE {table_name} SET {column_name}_errors = ? WHERE rowid = ?",
            (new_errors, row_id)
        )


    def check_missing_values(self, table_name, columns, original_cursor, errors):
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
                    self.mark_error(table_name, column['name'], row[0], 'Missing Value')

    def check_data_type_mismatches(self, table_name, columns, original_cursor, errors):
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
                        self.mark_error(table_name, column['name'], row[0], 'Type Mismatch')


    def check_duplicate_records(self, table_name, columns, original_cursor, errors):
        column_names = ', '.join([c['name'] for c in columns])
        query = f"""
        SELECT {column_names}, COUNT(*) as cnt, MIN(rowid) as min_rowid
        FROM {table_name}
        GROUP BY {column_names}
        HAVING cnt > 1
        """
        original_cursor.execute(query)
        duplicates = original_cursor.fetchall()
        if duplicates:
            duplicate_count = sum(dup[-2] for dup in duplicates)  # Sum of all duplicate counts
            errors.append({
                "table": table_name,
                "column": "All",
                "error_type": "Duplicate Records",
                "count": duplicate_count
            })
            for dup in duplicates:
                conditions = ' AND '.join([f"{c['name']} = ?" for c in columns])
                query = f"SELECT rowid FROM {table_name} WHERE {conditions}"
                original_cursor.execute(query, dup[:-2])
                duplicate_rows = original_cursor.fetchall()
                for row in duplicate_rows:
                    self.mark_error(table_name, "All", row[0], 'Duplicate Record')

    def check_inconsistent_formatting(self, table_name, columns, original_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL AND {column['name']} != '';"
                original_cursor.execute(query)
                rows = original_cursor.fetchall()
                
                date_formats = set()
                inconsistent_rows = []
                for row in rows:
                    try:
                        datetime.strptime(row[1], '%Y-%m-%d')
                        date_formats.add('%Y-%m-%d')
                    except ValueError:
                        try:
                            datetime.strptime(row[1], '%d/%m/%Y')
                            date_formats.add('%d/%m/%Y')
                        except ValueError:
                            pass
                    
                    if len(date_formats) > 1:
                        inconsistent_rows.append(row)
                
                if inconsistent_rows:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Inconsistent Date Formatting",
                        "count": len(inconsistent_rows)
                    })
                    for row in inconsistent_rows:
                        self.mark_error(table_name, column['name'], row[0], 'Inconsistent Date Format')

    def check_outliers(self, table_name, columns, original_cursor, errors):
        for column in columns:
            if column['type'].lower() in ['integer', 'real', 'float', 'double']:
                query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL;"
                original_cursor.execute(query)
                values = original_cursor.fetchall()

                if values:
                    data = [row[1] for row in values]
                    Q1 = np.percentile(data, 25)
                    Q3 = np.percentile(data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = [row for row in values if row[1] < lower_bound or row[1] > upper_bound]

                    if outliers:
                        errors.append({
                            "table": table_name,
                            "column": column['name'],
                            "error_type": "Outliers",
                            "count": len(outliers)
                        })
                        for row in outliers:
                            self.mark_error(table_name, column['name'], row[0], 'Outlier')

    def check_whitespace(self, table_name, columns, original_cursor, errors):
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
                        self.mark_error(table_name, column['name'], row[0], 'Whitespace Issue')

    def check_special_characters(self, table_name, columns, original_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL;"
                original_cursor.execute(query)
                rows = original_cursor.fetchall()
                special_char_rows = [row for row in rows if any(not c.isalnum() and not c.isspace() for c in str(row[1]))]
                if special_char_rows:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Special Characters",
                        "count": len(special_char_rows)
                    })
                    for row in special_char_rows:
                        self.mark_error(table_name, column['name'], row[0], 'Special Characters')

    def check_inconsistent_capitalization(self, table_name, columns, original_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL;"
                original_cursor.execute(query)
                rows = original_cursor.fetchall()
                
                case_types = set()
                inconsistent_rows = []
                for row in rows:
                    if row[1].islower():
                        case_types.add('lowercase')
                    elif row[1].isupper():
                        case_types.add('uppercase')
                    elif row[1].istitle():
                        case_types.add('titlecase')
                    else:
                        case_types.add('mixedcase')
                    
                    if len(case_types) > 1:
                        inconsistent_rows.append(row)
                
                if inconsistent_rows:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Inconsistent Capitalization",
                        "count": len(inconsistent_rows)
                    })
                    for row in inconsistent_rows:
                        self.mark_error(table_name, column['name'], row[0], 'Inconsistent Capitalization')

    def check_data_truncation(self, table_name, columns, original_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT MAX(LENGTH({column['name']})) FROM {table_name};"
                original_cursor.execute(query)
                max_length = original_cursor.fetchone()[0]
                
                if max_length is not None and max_length > 0:
                    threshold = max_length * 0.9
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
                            self.mark_error(table_name, column['name'], row[0], 'Possible Data Truncation')

    def check_value_frequency(self, table_name, columns, original_cursor, errors):
        for column in columns:
            query = f"SELECT {column['name']}, COUNT(*) as frequency FROM {table_name} GROUP BY {column['name']} ORDER BY frequency DESC LIMIT 1;"
            original_cursor.execute(query)
            most_frequent = original_cursor.fetchone()
            
            if most_frequent:
                value, frequency = most_frequent
                total_rows = original_cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                frequency_percentage = (frequency / total_rows) * 100
                
                if frequency_percentage > 90:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "High Frequency Value",
                        "count": frequency,
                        "details": f"Value '{value}' appears in {frequency_percentage:.2f}% of rows"
                    })
                    query = f"SELECT rowid FROM {table_name} WHERE {column['name']} = ?;"
                    original_cursor.execute(query, (value,))
                    high_frequency_rows = original_cursor.fetchall()
                    for row in high_frequency_rows:
                        self.mark_error(table_name, column['name'], row[0], 'High Frequency Value')

    def check_date_range(self, table_name, columns, original_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL AND {column['name']} != '';"
                original_cursor.execute(query)
                rows = original_cursor.fetchall()
                
                suspicious_rows = []
                for row in rows:
                    try:
                        date = datetime.strptime(row[1], '%Y-%m-%d')
                        if date.year < 1900 or date.year > datetime.now().year + 1:
                            suspicious_rows.append(row)
                    except ValueError:
                        pass
                
                if suspicious_rows:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Suspicious Date Range",
                        "count": len(suspicious_rows)
                    })
                    for row in suspicious_rows:
                        self.mark_error(table_name, column['name'], row[0], 'Suspicious Date Range')

    def check_numeric_range(self, table_name, columns, original_cursor, errors):
        for column in columns:
            if column['type'].lower() in ['integer', 'real', 'float', 'double']:
                query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL;"
                original_cursor.execute(query)
                rows = original_cursor.fetchall()
                
                if rows:
                    values = [row[1] for row in rows]
                    min_val, max_val = min(values), max(values)
                    range_size = max_val - min_val
                    if range_size > 1e9:
                        large_range_rows = [row for row in rows if row[1] in (min_val, max_val)]
                        errors.append({
                            "table": table_name,
                            "column": column['name'],
                            "error_type": "Large Numeric Range",
                            "count": len(large_range_rows),
                            "details": f"Range from {min_val} to {max_val}"
                        })
                        for row in large_range_rows:
                            self.mark_error(table_name, column['name'], row[0], 'Large Numeric Range')

    def check_string_length(self, table_name, columns, original_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                query = f"SELECT rowid, {column['name']}, LENGTH({column['name']}) as length FROM {table_name} WHERE {column['name']} IS NOT NULL;"
                original_cursor.execute(query)
                rows = original_cursor.fetchall()
                
                short_strings = [row for row in rows if row[2] < 2]
                long_strings = [row for row in rows if row[2] > 255]
                
                if short_strings:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Very Short Strings",
                        "count": len(short_strings)
                    })
                    for row in short_strings:
                        self.mark_error(table_name, column['name'], row[0], 'Very Short String')
                
                if long_strings:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Very Long Strings",
                        "count": len(long_strings)
                    })
                    for row in long_strings:
                        self.mark_error(table_name, column['name'], row[0], 'Very Long String')

    def check_email_format(self, table_name, columns, original_cursor, errors):
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        for column in columns:
            if column['type'].lower() == 'text' and 'email' in column['name'].lower():
                query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL;"
                original_cursor.execute(query)
                rows = original_cursor.fetchall()
                
                invalid_emails = [row for row in rows if not re.match(email_regex, row[1])]
                
                if invalid_emails:
                    errors.append({
                        "table": table_name,
                        "column": column['name'],
                        "error_type": "Invalid Email Format",
                        "count": len(invalid_emails)
                    })
                    for row in invalid_emails:
                        self.mark_error(table_name, column['name'], row[0], 'Invalid Email Format')

    def check_unique_constraints(self, table_name, columns, original_cursor, errors):
        for column in columns:
            query = f"SELECT {column['name']}, COUNT(*) as count FROM {table_name} GROUP BY {column['name']} HAVING count > 1;"
            original_cursor.execute(query)
            non_unique_values = original_cursor.fetchall()
            
            if non_unique_values:
                total_non_unique = sum(row[1] for row in non_unique_values)
                errors.append({
                    "table": table_name,
                    "column": column['name'],
                    "error_type": "Non-unique Values",
                    "count": total_non_unique
                })
                for value, count in non_unique_values:
                    query = f"SELECT rowid FROM {table_name} WHERE {column['name']} = ?;"
                    original_cursor.execute(query, (value,))
                    non_unique_rows = original_cursor.fetchall()
                    for row in non_unique_rows:
                        self.mark_error(table_name, column['name'], row[0], 'Non-unique Value')

    def check_foreign_key_integrity(self, table_name, columns, original_cursor, errors):
        query = f"PRAGMA foreign_key_list({table_name});"
        original_cursor.execute(query)
        foreign_keys = original_cursor.fetchall()
        
        for fk in foreign_keys:
            fk_column = fk[3]
            ref_table = fk[2]
            ref_column = fk[4]
            
            query = f"""
            SELECT {table_name}.rowid, {table_name}.{fk_column}
            FROM {table_name}
            LEFT JOIN {ref_table} ON {table_name}.{fk_column} = {ref_table}.{ref_column}
            WHERE {ref_table}.{ref_column} IS NULL AND {table_name}.{fk_column} IS NOT NULL;
            """
            original_cursor.execute(query)
            invalid_fks = original_cursor.fetchall()
            
            if invalid_fks:
                errors.append({
                    "table": table_name,
                    "column": fk_column,
                    "error_type": "Invalid Foreign Key",
                    "count": len(invalid_fks),
                    "details": f"References {ref_table}.{ref_column}"
                })
                for row in invalid_fks:
                    self.mark_error(table_name, fk_column, row[0], 'Invalid Foreign Key')

    def check_data_consistency(self, table_name, columns, original_cursor, errors):
        date_columns = [col for col in columns if 'date' in col['name'].lower()]
        for i in range(len(date_columns)):
            for j in range(i+1, len(date_columns)):
                col1 = date_columns[i]
                col2 = date_columns[j]
                query = f"""
                SELECT rowid, {col1['name']}, {col2['name']}
                FROM {table_name}
                WHERE {col1['name']} > {col2['name']}
                AND {col1['name']} IS NOT NULL
                AND {col2['name']} IS NOT NULL;
                """
                original_cursor.execute(query)
                inconsistent_dates = original_cursor.fetchall()
                
                if inconsistent_dates:
                    errors.append({
                        "table": table_name,
                        "column": f"{col1['name']} and {col2['name']}",
                        "error_type": "Date Inconsistency",
                        "count": len(inconsistent_dates),
                        "details": f"{col1['name']} is after {col2['name']}"
                    })
                    for row in inconsistent_dates:
                        self.mark_error(table_name, col1['name'], row[0], 'Date Inconsistency')
                        self.mark_error(table_name, col2['name'], row[0], 'Date Inconsistency')

    def detect_column_types(self, df):
        """Detect column types and categorize them."""
        column_info = {
            "dates": [],
            "numerics": [],
            "categoricals": []
        }
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                column_info["dates"].append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                column_info["numerics"].append(col)
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                column_info["categoricals"].append(col)
        
        return column_info

    def check_logical_relationships(self, table_name, columns, original_cursor, errors):
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", original_cursor.connection)
        column_types = self.detect_column_types(df)
        
        # Check date relationships
        for i in range(len(column_types["dates"])):
            for j in range(i+1, len(column_types["dates"])):
                col1, col2 = column_types["dates"][i], column_types["dates"][j]
                invalid = df[df[col1] > df[col2]]
                if not invalid.empty:
                    errors.append({
                        "table": table_name,
                        "column": f"{col1}, {col2}",
                        "error_type": "Logical Relationship Violation",
                        "count": len(invalid),
                        "details": f"{col1} is after {col2}"
                    })
                    for _, row in invalid.iterrows():
                        self.mark_error(table_name, col1, row.name, 'Logical Relationship Violation')
                        self.mark_error(table_name, col2, row.name, 'Logical Relationship Violation')

        # Check numeric relationships (example: total columns)
        for col in column_types["numerics"]:
            if 'total' in col.lower():
                potential_parts = [c for c in column_types["numerics"] if c != col and c in col.lower()]
                if potential_parts:
                    calculated_total = df[potential_parts].sum(axis=1)
                    invalid = df[df[col] != calculated_total]
                    if not invalid.empty:
                        errors.append({
                            "table": table_name,
                            "column": f"{col}, {', '.join(potential_parts)}",
                            "error_type": "Logical Relationship Violation",
                            "count": len(invalid),
                            "details": f"{col} does not equal sum of {', '.join(potential_parts)}"
                        })
                        for _, row in invalid.iterrows():
                            self.mark_error(table_name, col, row.name, 'Logical Relationship Violation')

    def check_pattern_matching(self, table_name, columns, original_cursor, errors):
        for column in columns:
            if column['type'].lower() == 'text':
                # Example patterns (can be extended)
                patterns = {
                    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    'phone': r'^\+?1?\d{9,15}$',
                    'zipcode': r'^\d{5}(-\d{4})?$',
                    'url': r'^(http|https)://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$'
                }
                
                for pattern_name, regex in patterns.items():
                    if pattern_name in column['name'].lower():
                        query = f"SELECT rowid, {column['name']} FROM {table_name} WHERE {column['name']} IS NOT NULL AND {column['name']} NOT REGEXP ?;"
                        original_cursor.execute(query, (regex,))
                        invalid_rows = original_cursor.fetchall()
                        
                        if invalid_rows:
                            errors.append({
                                "table": table_name,
                                "column": column['name'],
                                "error_type": "Pattern Mismatch",
                                "count": len(invalid_rows),
                                "details": f"Does not match {pattern_name} pattern"
                            })
                            for row in invalid_rows:
                                self.mark_error(table_name, column['name'], row[0], 'Pattern Mismatch')

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
        if not self.enable_ai_interpretation:
            print(info(f"AI interpretation skipped for {table_name} as it is disabled in settings."))
            return

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
        for table_name in self.schema.keys():
            try:
                self.marked_cursor.execute(f"SELECT * FROM {table_name}")
                rows = self.marked_cursor.fetchall()
                
                if rows:
                    headers = [description[0] for description in self.marked_cursor.description]
                    output_file = os.path.join(self.output_folder, f"{table_name}_marked_data.csv")
                    
                    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        
                        # Create new headers for each error type
                        new_headers = []
                        error_columns = []
                        for header in headers:
                            if header.endswith('_errors'):
                                error_columns.append(header)
                                # Add a column for each error type
                                new_headers.extend([f"{header}_{error_type}" for error_type in self.error_types])
                            else:
                                new_headers.append(header)
                        
                        writer.writerow(new_headers)
                        
                        for row in rows:
                            new_row = []
                            for i, cell in enumerate(row):
                                if headers[i] in error_columns:
                                    errors = [e.strip().lower() for e in cell.split(',')] if cell else []  # Standardized error list
                                    for error_type in self.error_types:
                                        new_row.append('1' if error_type.lower() in errors else '0')
                                else:
                                    new_row.append(str(cell) if cell is not None else '')
                            writer.writerow(new_row)
                    
                    print(info(f"Marked data for {table_name} exported to {output_file}"))
            except sqlite3.OperationalError as e:
                print(warning(f"Error exporting marked data for table {table_name}: {str(e)}"))
                print(info("Skipping this table and continuing with others..."))
                continue


    def save_column_name_changes(self):
        if self.column_name_changes:
            changes_file = os.path.join(self.output_folder, "column_name_changes.csv")
            with open(changes_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Original Name', 'New Name'])
                for original, new in self.column_name_changes.items():
                    writer.writerow([original, new])
            print(info(f"Column name changes saved to {changes_file}"))

    def check_table(self, table_name, columns):
        original_conn = sqlite3.connect(self.db_path)
        original_cursor = original_conn.cursor()

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
            self.check_value_frequency,
            self.check_date_range,
            self.check_numeric_range,
            self.check_string_length,
            self.check_email_format,
            self.check_unique_constraints,
            self.check_foreign_key_integrity,
            self.check_data_consistency,
            self.check_logical_relationships,
            self.check_pattern_matching
        ]

        for method in check_methods:
            self.current_check += 1
            progress = (self.current_check / self.total_checks) * 100
            print(info(f"Progress: {progress:.2f}% - Running {method.__name__}"))
            try:
                method(table_name, columns, original_cursor, errors)
            except sqlite3.OperationalError as e:
                print(warning(f"Error in {method.__name__} for table {table_name}: {str(e)}"))
                print(info("Skipping this check and continuing with others..."))
                continue

        self.marked_conn.commit()
        original_conn.close()

        self.save_errors_to_csv(table_name, errors)
        self.generate_ai_interpretation(table_name, errors)

    def run(self):
        print(info("Starting Data Quality Check..."))
        self.marked_db_path = self.create_marked_database()
        for table_name, columns in self.schema.items():
            print(highlight(f"\nAnalyzing table: {table_name}"))
            try:
                self.check_table(table_name, columns)
            except sqlite3.OperationalError as e:
                print(error(f"Error checking table {table_name}: {str(e)}"))
                print(info("Skipping this table and continuing with others..."))
                continue
        self.export_marked_data_to_csv()
        self.marked_conn.close()
        self.save_column_name_changes()
        print(success(f"Data Quality Check completed. Results saved in {self.output_folder}"))

if __name__ == "__main__":
    try:
        db_path = "path/to/your/database.sqlite"  # Update this with your actual database path
        erag_api = EragAPI()  # Initialize your API object here
        
        if erag_api is None or not hasattr(erag_api, 'chat'):
            print(warning("EragAPI not properly initialized. Proceeding without AI interpretation."))
            erag_api = None
        
        checker = DataQualityChecker(erag_api, db_path)
        checker.run()
    except Exception as e:
        print(error(f"An error occurred: {str(e)}"))
        import traceback
        print(traceback.format_exc())