# Standard library imports
import os
import re

# Third-party imports
import pandas as pd
import sqlite3

# Local imports
from src.settings import settings

def sanitize_table_name(name):
    # Remove or replace forbidden characters
    name = re.sub(r'[^\w]', '_', name)
    # Ensure the name doesn't start with a number
    if name[0].isdigit():
        name = '_' + name
    return name

def process_structured_data(file_path):
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            combined_df = df
            print("Processing CSV file.")
        elif file_path.endswith('.xlsx'):
            xlsx = pd.ExcelFile(file_path)
            first_sheet = True
            combined_df = pd.DataFrame()
            excluded_sheets = []

            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name)
                if first_sheet:
                    combined_df = df
                    first_sheet_header = list(df.columns)
                    first_sheet = False
                    print(f"Processing sheet: {sheet_name}")
                elif list(df.columns) == first_sheet_header:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                    print(f"Processing sheet: {sheet_name}")
                else:
                    excluded_sheets.append(sheet_name)
                    print(f"Warning: Sheet '{sheet_name}' has a different header. Excluding from import.")
            
            if excluded_sheets:
                print(f"The following sheets were excluded due to different headers: {', '.join(excluded_sheets)}")
        else:
            raise ValueError("Unsupported file format. Please use CSV or XLSX.")

        if combined_df.empty:
            raise ValueError("No valid data found in the file.")

        # Create SQLite database
        db_path = os.path.join(settings.output_folder, 'structured_data.db')
        conn = sqlite3.connect(db_path)

        # Sanitize table name
        original_table_name = os.path.splitext(os.path.basename(file_path))[0]
        sanitized_table_name = sanitize_table_name(original_table_name)

        # Save data to SQLite
        combined_df.to_sql(sanitized_table_name, conn, if_exists='replace', index=False)

        conn.close()
        
        print(f"Data has been processed and saved as '{sanitized_table_name}' in the database.")
        return True
    except Exception as e:
        print(f"Error processing structured data: {str(e)}")
        return False