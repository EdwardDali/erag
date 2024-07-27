import pandas as pd
import sqlite3
import os
import re
from src.settings import settings

def sanitize_table_name(name):
    # Remove or replace forbidden characters
    name = re.sub(r'[^\w]', '_', name)
    # Ensure the name doesn't start with a number
    if name[0].isdigit():
        name = '_' + name
    return name

def create_information_schema(conn, table_name, column_info):
    cursor = conn.cursor()
    
    # Create the information_schema.columns table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS information_schema_columns (
        table_catalog TEXT,
        table_schema TEXT,
        table_name TEXT,
        column_name TEXT,
        ordinal_position INTEGER,
        data_type TEXT,
        PRIMARY KEY (table_name, column_name)
    )
    ''')
    
    # Insert the column information into the information_schema.columns table
    for i, column in enumerate(column_info, start=1):
        cursor.execute('''
        INSERT OR REPLACE INTO information_schema_columns 
        (table_catalog, table_schema, table_name, column_name, ordinal_position, data_type)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', ('main', 'main', table_name, column['name'], i, column['type']))
    
    conn.commit()

def process_structured_data(file_path):
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or XLSX.")

        # Create SQLite database
        db_path = os.path.join(settings.output_folder, 'structured_data.db')
        conn = sqlite3.connect(db_path)

        # Sanitize table name
        original_table_name = os.path.splitext(os.path.basename(file_path))[0]
        sanitized_table_name = sanitize_table_name(original_table_name)

        # Save data to SQLite
        df.to_sql(sanitized_table_name, conn, if_exists='replace', index=False)

        # Get column information
        column_info = [
            {'name': col, 'type': str(df[col].dtype)} 
            for col in df.columns
        ]

        # Create or update the information schema
        create_information_schema(conn, sanitized_table_name, column_info)

        conn.close()
        
        print(f"Table '{original_table_name}' has been processed and saved as '{sanitized_table_name}' in the database.")
        print(f"Information schema has been updated for table '{sanitized_table_name}'.")
        return True
    except Exception as e:
        print(f"Error processing structured data: {str(e)}")
        return False
