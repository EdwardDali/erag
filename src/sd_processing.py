import pandas as pd
import sqlite3
import os
from src.settings import settings

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

        # Save data to SQLite
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        conn.close()
        return True
    except Exception as e:
        print(f"Error processing structured data: {str(e)}")
        return False
