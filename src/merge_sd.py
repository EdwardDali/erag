# Standard library imports
import os
import time
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dateutil.parser import parse
import jellyfish


def load_file(file_path):
    """Load a CSV or XLSX file into a pandas DataFrame."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or XLSX.")

def find_common_columns(df1, df2):
    """Find common and similar columns between two DataFrames."""
    common_columns = set(df1.columns) & set(df2.columns)
    similar_columns = []
    for col1 in df1.columns:
        for col2 in df2.columns:
            if col1 not in common_columns and col2 not in common_columns:
                if fuzz.ratio(col1.lower(), col2.lower()) > 80:
                    similar_columns.append((col1, col2))
    return common_columns, similar_columns

def analyze_data_types(df1, df2):
    """Analyze and compare data types of columns in both DataFrames."""
    data_types = {}
    for col in set(df1.columns) | set(df2.columns):
        if col in df1.columns and col in df2.columns:
            data_types[col] = f"DF1: {df1[col].dtype}, DF2: {df2[col].dtype}"
        elif col in df1.columns:
            data_types[col] = f"DF1: {df1[col].dtype}, DF2: Not present"
        else:
            data_types[col] = f"DF1: Not present, DF2: {df2[col].dtype}"
    return data_types

def fuzzy_match(df1, df2, col1, col2, threshold=80):
    """Performs fuzzy matching on columns between two dataframes."""
    matches = []
    for idx1, val1 in df1[col1].items():
        best_match = process.extractOne(str(val1), df2[col2].astype(str), scorer=fuzz.token_sort_ratio)
        if best_match[1] >= threshold:
            matches.append((idx1, df2[df2[col2].astype(str) == best_match[0]].index[0], best_match[1]))
    return pd.DataFrame(matches, columns=['df1_index', 'df2_index', 'similarity'])

def cosine_similarity_match(df1, df2, col1, col2, threshold=0.8):
    """Performs cosine similarity matching between two text columns."""
    tfidf = TfidfVectorizer().fit_transform(df1[col1].fillna('').astype(str) + ' ' + df2[col2].fillna('').astype(str))
    cosine_sim = cosine_similarity(tfidf)
    matches = []
    for i in range(len(df1)):
        for j in range(len(df2)):
            if cosine_sim[i, j + len(df1)] > threshold:
                matches.append((df1.index[i], df2.index[j], cosine_sim[i, j + len(df1)]))
    return pd.DataFrame(matches, columns=['df1_index', 'df2_index', 'similarity'])

def phonetic_match(df1, df2, col1, col2, method='soundex'):
    """Performs phonetic matching using various algorithms."""
    phonetic_funcs = {
        'soundex': jellyfish.soundex,
        'metaphone': jellyfish.metaphone,
        'nysiis': jellyfish.nysiis
    }
    func = phonetic_funcs.get(method, jellyfish.soundex)
    
    df1['phonetic'] = df1[col1].astype(str).apply(func)
    df2['phonetic'] = df2[col2].astype(str).apply(func)
    
    merged = pd.merge(df1, df2, on='phonetic', suffixes=('_1', '_2'))
    return merged[[col1, col2, 'phonetic']]

def date_fuzzy_match(df1, df2, date_col1, date_col2, max_days_diff=3):
    """Performs fuzzy matching on date columns within a specified range."""
    def parse_date(date_str):
        try:
            return parse(str(date_str))
        except:
            return pd.NaT

    df1['parsed_date'] = df1[date_col1].apply(parse_date)
    df2['parsed_date'] = df2[date_col2].apply(parse_date)
    
    matches = []
    for idx1, row1 in df1.iterrows():
        for idx2, row2 in df2.iterrows():
            if pd.notna(row1['parsed_date']) and pd.notna(row2['parsed_date']):
                days_diff = abs((row1['parsed_date'] - row2['parsed_date']).days)
                if days_diff <= max_days_diff:
                    matches.append((idx1, idx2, days_diff))
    
    return pd.DataFrame(matches, columns=['df1_index', 'df2_index', 'days_difference'])

def numeric_range_match(df1, df2, num_col1, num_col2, tolerance=0.1):
    """Performs matching on numeric columns within a specified tolerance range."""
    matches = []
    for idx1, val1 in df1[num_col1].items():
        for idx2, val2 in df2[num_col2].items():
            if pd.notna(val1) and pd.notna(val2):
                if abs(val1 - val2) / max(abs(val1), abs(val2)) <= tolerance:
                    matches.append((idx1, idx2, abs(val1 - val2)))
    
    return pd.DataFrame(matches, columns=['df1_index', 'df2_index', 'difference'])

def determine_best_method(df1, df2, cols1, cols2):
    """Determine the best matching method based on column types and data."""
    if len(cols1) != len(cols2):
        raise ValueError("The number of columns to match must be the same for both dataframes.")
    
    methods = []
    for col1, col2 in zip(cols1, cols2):
        # Check if columns are exactly the same
        if col1 == col2 and df1[col1].dtype == df2[col2].dtype:
            methods.append('exact')
        # Check for dates
        elif 'datetime' in str(df1[col1].dtype) or 'datetime' in str(df2[col2].dtype):
            methods.append('date')
        # Check for numeric data
        elif np.issubdtype(df1[col1].dtype, np.number) and np.issubdtype(df2[col2].dtype, np.number):
            methods.append('numeric')
        # For string data, use a combination of methods
        elif df1[col1].dtype == 'object' and df2[col2].dtype == 'object':
            # Check for potential name columns
            if 'name' in col1.lower() or 'name' in col2.lower():
                methods.append('combined_text')
            # For longer text, use cosine similarity
            elif df1[col1].str.len().mean() > 10 or df2[col2].str.len().mean() > 10:
                methods.append('cosine')
            # For shorter text, use fuzzy matching
            else:
                methods.append('fuzzy')
        # Default to fuzzy matching
        else:
            methods.append('fuzzy')
    
    return methods

def perform_multi_column_matching(df1, df2, cols1, cols2, methods):
    """Perform matching based on multiple columns."""
    df1['__temp_key__'] = df1[cols1].astype(str).agg('-'.join, axis=1)
    df2['__temp_key__'] = df2[cols2].astype(str).agg('-'.join, axis=1)
    
    if all(method == 'exact' for method in methods):
        return pd.merge(df1, df2, left_on='__temp_key__', right_on='__temp_key__', how='inner')
    
    matches = []
    for idx1, key1 in df1['__temp_key__'].items():
        best_match = None
        best_score = 0
        for idx2, key2 in df2['__temp_key__'].items():
            score = 0
            for i, (method, col1, col2) in enumerate(zip(methods, cols1, cols2)):
                val1, val2 = str(df1.loc[idx1, col1]), str(df2.loc[idx2, col2])
                if method == 'fuzzy' or method == 'combined_text':
                    score += fuzz.token_sort_ratio(val1, val2)
                elif method == 'cosine':
                    tfidf = TfidfVectorizer().fit_transform([val1, val2])
                    score += cosine_similarity(tfidf)[0][1] * 100
                elif method == 'date':
                    try:
                        date1, date2 = parse(val1), parse(val2)
                        score += 100 if abs((date1 - date2).days) <= 3 else 0
                    except:
                        score += 0
                elif method == 'numeric':
                    try:
                        num1, num2 = float(val1), float(val2)
                        score += 100 if abs(num1 - num2) / max(abs(num1), abs(num2)) <= 0.1 else 0
                    except:
                        score += 0
            score /= len(methods)  # Normalize score
            if score > best_score:
                best_score = score
                best_match = idx2
        
        if best_match is not None and best_score >= 80:  # Threshold for considering a match
            matches.append((idx1, best_match, best_score))
    
    return pd.DataFrame(matches, columns=['df1_index', 'df2_index', 'similarity'])

def create_merged_table(df1, df2, matches, output_path):
    """Create a merged CSV file."""
    if isinstance(matches, pd.DataFrame) and 'df1_index' in matches.columns and 'df2_index' in matches.columns:
        # Merge all columns from both dataframes
        merged_df = pd.merge(
            df1.loc[matches['df1_index']].reset_index(drop=True),
            df2.loc[matches['df2_index']].reset_index(drop=True),
            left_index=True, 
            right_index=True, 
            suffixes=('_df1', '_df2')
        )
    else:
        merged_df = matches
    
    merged_df.to_csv(output_path, index=False)
    print(f"Merged table created in {output_path}")
    return merged_df

def merge_structured_data(file1, file2, output_folder):
    # Load files
    try:
        df1 = load_file(file1)
        df2 = load_file(file2)
    except Exception as e:
        print(f"Error loading files: {e}")
        return False

    print("\nTrying to find common matches using the following methods:")
    print("- Fuzzy matching: For comparing strings with minor differences")
    print("- Cosine similarity: For comparing longer text fields")
    print("- Phonetic matching: For comparing names or words that sound similar")
    print("- Date fuzzy matching: For comparing dates within a specified range")
    print("- Numeric range matching: For comparing numeric values within a tolerance")

    # Add a 3-second delay
    time.sleep(3)

    # Find common and similar columns
    common_columns, similar_columns = find_common_columns(df1, df2)

    # Analyze data types
    data_types = analyze_data_types(df1, df2)

    # Display findings in the console
    print("\nAnalysis Results:")
    print("Data types:")
    for col, dtype in data_types.items():
        print(f"{col}: {dtype}")
    print("\nSimilar columns:", ", ".join([f"{a} - {b}" for a, b in similar_columns]))
    print("Common columns:", ", ".join(common_columns))

    # Ask user if they want to use common columns or enter manually
    while True:
        use_common = input("\nDo you want to perform merging based on common columns identified? (y/n): ").strip().lower()
        if use_common in ['y', 'n']:
            break
        print("Invalid input. Please enter 'y' or 'n'.")

    if use_common == 'y':
        columns = list(common_columns)
    else:
        # Ask user to select columns to merge on
        print("\nSelect columns to match (enter column names separated by commas):")
        columns_input = input().strip()
        columns = [col.strip().strip("'\"") for col in columns_input.split(',')]

    # Validate columns
    invalid_columns = [col for col in columns if col not in df1.columns or col not in df2.columns]
    if invalid_columns:
        print(f"Error: The following columns are not present in both dataframes: {invalid_columns}")
        return False

    # Use the same columns for both dataframes
    cols1 = cols2 = columns

    # Determine best matching methods
    try:
        methods = determine_best_method(df1, df2, cols1, cols2)
        print("\nBest matching methods determined for each column:")
        for col, method in zip(columns, methods):
            print(f"- {col}: {method}")
    except Exception as e:
        print(f"Error determining matching methods: {e}")
        return False

    # Perform matching
    try:
        matches = perform_multi_column_matching(df1, df2, cols1, cols2, methods)
        print(f"\nMatching results: {len(matches)} matches found")
    except Exception as e:
        print(f"Error during matching: {e}")
        return False

    # Create merged table
    output_path = os.path.join(output_folder, "merged_data.csv")
    merged_df = create_merged_table(df1, df2, matches, output_path)
    
    # Print information about the merge
    print("\nMerge Information:")
    print(f"Number of rows in original df1: {len(df1)}")
    print(f"Number of rows in original df2: {len(df2)}")
    print(f"Number of rows in merged dataframe: {len(merged_df)}")
    print("\nColumns in merged dataframe:")
    for col in merged_df.columns:
        print(f"- {col}")
    
    messagebox.showinfo("Merge Complete", f"Merge completed successfully. Output saved to {output_path}")
    
    return True

if __name__ == "__main__":
    # This script is now intended to be used as a module, so we don't need a standalone main function
    pass
