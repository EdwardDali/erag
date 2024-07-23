import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, anderson, pearsonr, probplot
import os
from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import threading
import time
from functools import wraps

class TimeoutException(Exception):
    pass

class AdvancedExploratoryDataAnalysis:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 5
        self.table_name = None
        self.output_folder = None
        self.text_output = ""
        self.pdf_content = []
        self.findings = []
        self.llm_name = f"Worker: {self.worker_erag_api.model}, Supervisor: {self.supervisor_erag_api.model}"
        self.toc_entries = []
        self.executive_summary = ""
        self.image_paths = []
        self.max_pixels = 400000
        self.timeout_seconds = 10
        self.image_data = []
        self.pdf_generator = None

    def calculate_figure_size(self, aspect_ratio=16/9):
        max_width = int(np.sqrt(self.max_pixels * aspect_ratio))
        max_height = int(max_width / aspect_ratio)
        return (max_width / 100, max_height / 100)

    def timeout(timeout_duration):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                result = [TimeoutException("Function call timed out")]

                def target():
                    try:
                        result[0] = func(self, *args, **kwargs)
                    except Exception as e:
                        result[0] = e

                thread = threading.Thread(target=target)
                thread.start()
                thread.join(timeout_duration)

                if thread.is_alive():
                    print(f"Warning: {func.__name__} timed out after {timeout_duration} seconds. Skipping this graphic.")
                    return None
                else:
                    if isinstance(result[0], Exception):
                        raise result[0]
                    return result[0]
            return wrapper
        return decorator

    @timeout(10)
    def generate_plot(self, plot_function, *args, **kwargs):
        return plot_function(*args, **kwargs)

    def run(self):
        print(info(f"Starting Advanced Exploratory Data Analysis on {self.db_path}"))
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis completed. Results saved in {self.output_folder}"))

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize PDFReportGenerator with table name
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        analysis_methods = [
            self.value_counts_analysis,
            self.grouped_summary_statistics,
            self.frequency_distribution_analysis,
            self.kde_plot_analysis,
            self.violin_plot_analysis
        ]

        for method in analysis_methods:
            method(df, table_name)

    def value_counts_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Value Counts Analysis"))
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        results = {}
        
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            results[col] = value_counts.to_dict()
        
        self.interpret_results("Value Counts Analysis", results, table_name)

    def grouped_summary_statistics(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Grouped Summary Statistics"))
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        results = {}
        
        for cat_col in categorical_columns:
            for num_col in numerical_columns:
                grouped_stats = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'min', 'max'])
                results[f"{cat_col} - {num_col}"] = grouped_stats.to_dict()
        
        self.interpret_results("Grouped Summary Statistics", results, table_name)

    def frequency_distribution_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Frequency Distribution Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        
        for col in numerical_columns:
            def plot_frequency_distribution():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f'Frequency Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                return fig, ax

            result = self.generate_plot(plot_frequency_distribution)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_frequency_distribution.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping frequency distribution plot for {col} due to timeout.")
        
        self.interpret_results("Frequency Distribution Analysis", image_paths, table_name)

    def kde_plot_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - KDE Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        
        for col in numerical_columns:
            def plot_kde():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.kdeplot(df[col], shade=True, ax=ax)
                ax.set_title(f'KDE Plot for {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                return fig, ax

            result = self.generate_plot(plot_kde)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_kde_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping KDE plot for {col} due to timeout.")
        
        self.interpret_results("KDE Plot Analysis", image_paths, table_name)

    def violin_plot_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Violin Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        image_paths = []
        
        if len(categorical_columns) > 0:
            for num_col in numerical_columns:
                def plot_violin():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.violinplot(x=categorical_columns[0], y=num_col, data=df, ax=ax)
                    ax.set_title(f'Violin Plot of {num_col} grouped by {categorical_columns[0]}')
                    ax.set_xlabel(categorical_columns[0])
                    ax.set_ylabel(num_col)
                    plt.xticks(rotation=45)
                    return fig, ax

                result = self.generate_plot(plot_violin)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{num_col}_violin_plot.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping violin plot for {num_col} due to timeout.")
        else:
            print("No categorical columns found for violin plot analysis.")
        
        self.interpret_results("Violin Plot Analysis", image_paths, table_name)

    def interpret_results(self, analysis_type, results, table_name):
        if isinstance(results, dict):
            results_str = "\n".join([f"{k}: {v}" for k, v in results.items()])
        elif isinstance(results, list):
            results_str = "\n".join([str(item) for item in results])
        else:
            results_str = str(results)

        prompt = f"""
        Analysis type: {analysis_type}
        Table name: {table_name}
        Results:
        {results_str}

        Please provide a detailed interpretation of these results, highlighting any noteworthy patterns, anomalies, or insights. Focus on the most important aspects that would be valuable for data analysis.

        Structure your response in the following format:

        1. Analysis:
        [Provide a detailed description of the analysis performed]

        2. SQLite Statements:
        [Provide the SQLite statements that would be used to perform this analysis]

        3. Positive Findings:
        [List any positive findings, or state "No significant positive findings" if none]

        4. Negative Findings:
        [List any negative findings, or state "No significant negative findings" if none]

        5. Conclusion:
        [Summarize the key takeaways and implications of this analysis]

        If there are no significant findings, state "No significant findings" in the appropriate sections and briefly explain why.

        Interpretation:
        """
        interpretation = self.worker_erag_api.chat([{"role": "system", "content": "You are a data analyst providing insights on advanced exploratory data analysis results. Respond in the requested format."}, 
                                                    {"role": "user", "content": prompt}])
        
        # Second LLM call to review and enhance the interpretation
        check_prompt = f"""
        Original data and analysis type:
        {prompt}

        Previous interpretation:
        {interpretation}

        Please review and improve the above interpretation. Ensure it accurately reflects the original data and analysis type. Enhance the text by:
        1. Verifying the accuracy of the interpretation against the original data.
        2. Ensuring the structure (Analysis, SQLite Statements, Positive Findings, Negative Findings, Conclusion) is maintained.
        3. Making the interpretation more narrative and detailed by adding context and explanations.
        4. Addressing any important aspects of the data that weren't covered.

        Provide your response in the same format, maintaining the original structure. 
        Do not add comments, questions, or explanations about the changes - simply provide the improved version.
        """

        enhanced_interpretation = self.supervisor_erag_api.chat([
            {"role": "system", "content": "You are a data analyst improving interpretations of advanced exploratory data analysis results. Provide direct enhancements without adding meta-comments or detailing the changes done."},
            {"role": "user", "content": check_prompt}
        ])

        print(success(f"AI Interpretation for {analysis_type}:"))
        print(enhanced_interpretation.strip())
        
        self.text_output += f"\n{enhanced_interpretation.strip()}\n\n"
        
        # Handle images
        image_data = []
        if isinstance(results, list):
            for i, item in enumerate(results):
                if isinstance(item, str) and item.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_data.append((f"{analysis_type} - Image {i+1}", item))
        
        self.pdf_content.append((analysis_type, image_data, enhanced_interpretation.strip()))
        
        # Extract important findings
        lines = enhanced_interpretation.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith("3. Positive Findings:") or line.startswith("4. Negative Findings:"):
                for finding in lines[i+1:]:
                    if finding.strip() and not finding.startswith(("3.", "4.", "5.")):
                        self.findings.append(f"{analysis_type}: {finding.strip()}")
                    elif finding.startswith(("3.", "4.", "5.")):
                        break

        # Update self.image_data
        self.image_data.extend(image_data)

    def generate_executive_summary(self):
        if not self.findings:
            self.executive_summary = "No significant findings were identified during the advanced analysis. This could be due to a lack of data, uniform data distribution, or absence of notable patterns or anomalies in the dataset."
            return

        summary_prompt = f"""
        Based on the following findings from the Advanced Exploratory Data Analysis:
        
        {self.findings}
        
        Please provide an executive summary of the analysis. The summary should:
        1. Briefly introduce the purpose of the advanced analysis.
        2. Highlight the most significant insights and patterns discovered.
        3. Mention any potential issues or areas that require further investigation.
        4. Conclude with recommendations for next steps or areas to focus on.

        Structure the summary in multiple paragraphs for readability.
        Please provide your response in plain text format, without any special formatting or markup.
        """
        
        try:
            interpretation = self.worker_erag_api.chat([
                {"role": "system", "content": "You are a data analyst providing an executive summary of an advanced exploratory data analysis. Respond in plain text format."},
                {"role": "user", "content": summary_prompt}
            ])
            
            if interpretation is not None:
                # Updated second LLM call to focus on direct improvements
                check_prompt = f"""
                Please review and improve the following executive summary:

                {interpretation}

                Enhance the summary by:
                1. Making it more comprehensive and narrative by adding context and explanations.
                2. Addressing any important aspects of the analysis that weren't covered.
                3. Ensuring it includes a clear introduction, highlights of significant insights, mention of potential issues, and recommendations for next steps.

                Provide your response in plain text format, without any special formatting or markup.
                Do not add comments, questions, or explanations about the changes - simply provide the improved version.
                """

                enhanced_summary = self.supervisor_erag_api.chat([
                    {"role": "system", "content": "You are a data analyst improving an executive summary of an advanced exploratory data analysis. Provide direct enhancements without adding meta-comments."},
                    {"role": "user", "content": check_prompt}
                ])

                self.executive_summary = enhanced_summary.strip()
            else:
                self.executive_summary = "Error: Unable to generate executive summary."
        except Exception as e:
            print(error(f"An error occurred while generating the executive summary: {str(e)}"))
            self.executive_summary = "Error: Unable to generate executive summary due to an exception."

        print(success("Enhanced Executive Summary generated successfully."))
        print(self.executive_summary)

    def save_text_output(self):
        output_file = os.path.join(self.output_folder, "axda_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis Report for {self.table_name}"
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.executive_summary,
            self.findings,
            self.pdf_content,
            self.image_data,
            filename=f"axda_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
        else:
            print(error("Failed to generate PDF report"))
