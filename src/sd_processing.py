# Standard library imports
import os
import re
import sys

# Third-party imports
import pandas as pd
import numpy as np
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

def sanitize_column_name(name):
    return re.sub(r'\W+', '_', str(name).strip().lower())

def convert_datatypes(df):
    df.columns = [sanitize_column_name(col) for col in df.columns]
    for column in df.columns:
        # Convert timestamps to strings
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = df[column].astype(str)
        # Convert any remaining objects to strings
        elif df[column].dtype == 'object':
            df[column] = df[column].astype(str)
        # Convert float64 to float32 to avoid precision issues
        elif df[column].dtype == 'float64':
            df[column] = df[column].astype('float32')
        # Convert int64 to int32 if possible, otherwise to float32
        elif df[column].dtype == 'int64':
            if df[column].min() > np.iinfo(np.int32).min and df[column].max() < np.iinfo(np.int32).max:
                df[column] = df[column].astype('int32')
            else:
                df[column] = df[column].astype('float32')
    return df

def process_structured_data(file_path):
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            combined_df = convert_datatypes(df)  # Apply conversion immediately
            print("Processing CSV file.")
        elif file_path.endswith('.xlsx'):
            xlsx = pd.ExcelFile(file_path)
            first_sheet = True
            combined_df = pd.DataFrame()
            excluded_sheets = []

            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name)
                df = convert_datatypes(df)  # Apply conversion to each sheet
                if first_sheet:
                    combined_df = df
                    first_sheet_header = [sanitize_column_name(col) for col in df.columns]
                    first_sheet = False
                    print(f"Processing sheet: {sheet_name}")
                elif [sanitize_column_name(col) for col in df.columns] == first_sheet_header:
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

        # Print column types for debugging
        print("Column types after conversion:")
        for column, dtype in combined_df.dtypes.items():
            print(f"{column}: {dtype}")

        # Create SQLite database
        db_path = os.path.join(settings.output_folder, 'structured_data.db')
        conn = sqlite3.connect(db_path)

        # Sanitize table name
        original_table_name = os.path.splitext(os.path.basename(file_path))[0]
        sanitized_table_name = sanitize_table_name(original_table_name)

        # Ensure column names are sanitized before saving to SQL
        combined_df.columns = [sanitize_column_name(col) for col in combined_df.columns]

        # Save data to SQLite
        try:
            combined_df.to_sql(sanitized_table_name, conn, if_exists='replace', index=False)
        except sqlite3.OperationalError as e:
            if "syntax error" in str(e).lower():
                print("Error: Invalid column names. Please check for special characters in your headers.")
            else:
                print(f"SQLite error: {e}")
            return False
        finally:
            conn.close()
        
        print(f"Data has been processed and saved as '{sanitized_table_name}' in the database.")
        return True
    except Exception as e:
        print(f"Error processing structured data: {str(e)}")
        print("Exception details:")
        print(sys.exc_info())
        return False