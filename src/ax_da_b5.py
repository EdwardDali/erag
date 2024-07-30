import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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

class AdvancedExploratoryDataAnalysisB5:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 3
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

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_b5_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        analysis_methods = [
            self.cooks_distance_analysis,
            self.stl_decomposition_analysis,
            self.hampel_filter_analysis
        ]

        for method in analysis_methods:
            try:
                method(df, table_name)
            except Exception as e:
                error_message = f"An error occurred during {method.__name__}: {str(e)}"
                print(error(error_message))
                self.text_output += f"\n{error_message}\n"
                self.pdf_content.append((method.__name__, [], error_message))
            finally:
                self.technique_counter += 1

    def cooks_distance_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Cook's Distance Analysis"))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            print(warning("Not enough numeric columns for Cook's Distance analysis."))
            return
        
        X = df[numeric_cols].drop(numeric_cols[-1], axis=1)
        y = df[numeric_cols[-1]]
        
        model = LinearRegression()
        model.fit(X, y)
        
        influence = OLSInfluence(model.fit(X, y))
        cooks_d = influence.cooks_distance[0]
        
        def plot_cooks_distance():
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ax.stem(range(len(cooks_d)), cooks_d, markerfmt=",")
            ax.set_title("Cook's Distance")
            ax.set_xlabel("Observation")
            ax.set_ylabel("Cook's Distance")
            return fig, ax

        result = self.generate_plot(plot_cooks_distance)
        if result is not None:
            fig, ax = result
            img_path = os.path.join(self.output_folder, f"{table_name}_cooks_distance.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            threshold = 4 / len(df)
            influential_points = np.where(cooks_d > threshold)[0]
            
            results = {
                'image_path': img_path,
                'influential_points': influential_points.tolist(),
                'threshold': threshold
            }
            
            self.interpret_results("Cook's Distance Analysis", results, table_name)
        else:
            print("Skipping Cook's Distance plot due to timeout.")

    def stl_decomposition_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - STL Decomposition Analysis"))
        
        date_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(date_cols) == 0 or len(numeric_cols) == 0:
            print(warning("No suitable columns for STL decomposition analysis."))
            return
        
        date_col = date_cols[0]
        numeric_col = numeric_cols[0]
        
        df = df.sort_values(by=date_col)
        df = df.set_index(date_col)
        
        stl = STL(df[numeric_col], period=12)  # Assuming monthly data, adjust as needed
        result = stl.fit()
        
        def plot_stl_decomposition():
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(self.calculate_figure_size()[0], self.calculate_figure_size()[1]*2))
            ax1.plot(df.index, result.observed)
            ax1.set_ylabel('Observed')
            ax2.plot(df.index, result.trend)
            ax2.set_ylabel('Trend')
            ax3.plot(df.index, result.seasonal)
            ax3.set_ylabel('Seasonal')
            ax4.plot(df.index, result.resid)
            ax4.set_ylabel('Residual')
            plt.tight_layout()
            return fig, (ax1, ax2, ax3, ax4)

        result_plot = self.generate_plot(plot_stl_decomposition)
        if result_plot is not None:
            fig, _ = result_plot
            img_path = os.path.join(self.output_folder, f"{table_name}_stl_decomposition.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            results = {
                'image_path': img_path,
                'trend_strength': 1 - np.var(result.resid) / np.var(result.trend + result.resid),
                'seasonal_strength': 1 - np.var(result.resid) / np.var(result.seasonal + result.resid)
            }
            
            self.interpret_results("STL Decomposition Analysis", results, table_name)
        else:
            print("Skipping STL Decomposition plot due to timeout.")

    def hampel_filter_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hampel Filter Analysis"))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            print(warning("No numeric columns for Hampel Filter analysis."))
            return
        
        results = {}
        for col in numeric_cols:
            series = df[col]
            rolling_median = series.rolling(window=5, center=True).median()
            mad = np.abs(series - rolling_median).rolling(window=5, center=True).median()
            threshold = 3 * 1.4826 * mad
            outliers = np.abs(series - rolling_median) > threshold
            
            def plot_hampel():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(series.index, series, label='Original')
                ax.scatter(series.index[outliers], series[outliers], color='red', label='Outliers')
                ax.set_title(f'Hampel Filter Outliers - {col}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_hampel)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_hampel_filter.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                results[col] = {
                    'image_path': img_path,
                    'outliers_count': outliers.sum(),
                    'outliers_percentage': (outliers.sum() / len(series)) * 100
                }
            else:
                print(f"Skipping Hampel Filter plot for {col} due to timeout.")
        
        self.interpret_results("Hampel Filter Analysis", results, table_name)

    def interpret_results(self, analysis_type, results, table_name):
        prompt = f"""
        Analysis type: {analysis_type}
        Table name: {table_name}
        Results: {results}

        Please provide a detailed interpretation of these results, highlighting any noteworthy patterns, anomalies, or insights. Focus on the most important aspects that would be valuable for data analysis.

        Structure your response in the following format:

        1. Analysis:
        [Provide a detailed description of the analysis performed]

        2. Findings:
        [List the main findings, both positive and negative]

        3. Implications:
        [Discuss the implications of these findings for the dataset and potential further analyses]

        4. Recommendations:
        [Provide recommendations based on the findings]

        Interpretation:
        """
        interpretation = self.worker_erag_api.chat([{"role": "system", "content": "You are a data analyst providing insights on advanced exploratory data analysis results. Respond in the requested format."}, 
                                                    {"role": "user", "content": prompt}])
        
        check_prompt = f"""
        Original data and analysis type:
        {prompt}

        Previous interpretation:
        {interpretation}

        Please review and improve the above interpretation. Ensure it accurately reflects the original data and analysis type. Enhance the text by:
        1. Verifying the accuracy of the interpretation against the original data.
        2. Ensuring the structure (Analysis, Findings, Implications, Recommendations) is maintained.
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
        
        image_data = []
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, dict) and 'image_path' in value:
                    image_data.append((f"{analysis_type} - {key}", value['image_path']))
        elif isinstance(results, str) and results.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_data.append((analysis_type, results))
        
        self.pdf_content.append((analysis_type, image_data, enhanced_interpretation.strip()))
        
        lines = enhanced_interpretation.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith("2. Findings:"):
                for finding in lines[i+1:]:
                    if finding.strip() and not finding.startswith(("3.", "4.")):
                        self.findings.append(f"{analysis_type}: {finding.strip()}")
                    elif finding.startswith(("3.", "4.")):
                        break

        self.image_data.extend(image_data)

    def generate_executive_summary(self):
        if not self.findings:
            self.executive_summary = "No significant findings were identified during the analysis. This could be due to a lack of data, uniform data distribution, or absence of notable patterns or anomalies in the dataset."
            return

        successful_techniques = sum(1 for item in self.pdf_content if len(item[1]) > 0 or not item[2].startswith("An error occurred"))
        failed_techniques = self.total_techniques - successful_techniques

        summary_prompt = f"""
        Based on the following findings from the Advanced Exploratory Data Analysis (Batch 5):
        
        {self.findings}
        
        Additional context:
        - {successful_techniques} out of {self.total_techniques} analysis techniques were successfully completed.
        - {failed_techniques} techniques encountered errors and were skipped.
        
        Please provide an executive summary of the analysis. The summary should:
        1. Briefly introduce the purpose of the advanced analysis (Batch 5).
        2. Mention the number of successful and failed techniques.
        3. Highlight the most significant insights and patterns discovered.
        4. Mention any potential issues or areas that require further investigation.
        5. Discuss any limitations of the analysis due to failed techniques.
        6. Conclude with recommendations for next steps or areas to focus on.

        Structure the summary in multiple paragraphs for readability.
        Please provide your response in plain text format, without any special formatting or markup.
        """
        
        try:
            interpretation = self.worker_erag_api.chat([
                {"role": "system", "content": "You are a data analyst providing an executive summary of an advanced exploratory data analysis (Batch 5). Respond in plain text format."},
                {"role": "user", "content": summary_prompt}
            ])
            
            if interpretation is not None:
                check_prompt = f"""
                Please review and improve the following executive summary:

                {interpretation}

                Enhance the summary by:
                1. Making it more comprehensive and narrative by adding context and explanations.
                2. Addressing any important aspects of the analysis that weren't covered.
                3. Ensuring it includes a clear introduction, highlights of significant insights, mention of potential issues, and recommendations for next steps.
                4. Discussing the implications of any failed techniques on the overall analysis.

                Provide your response in plain text format, without any special formatting or markup.
                Do not add comments, questions, or explanations about the changes - simply provide the improved version.
                """

                enhanced_summary = self.supervisor_erag_api.chat([
                    {"role": "system", "content": "You are a data analyst improving an executive summary of an advanced exploratory data analysis (Batch 5). Provide direct enhancements without adding meta-comments."},
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
        output_file = os.path.join(self.output_folder, "axda_b5_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 5) Report for {self.table_name}"
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.executive_summary,
            self.findings,
            self.pdf_content,
            self.image_data,
            filename=f"axda_b5_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None

    def run(self):
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch 5) on {self.db_path}"))
        
        # Get all tables
        all_tables = self.get_tables()
        
        if not all_tables:
            print(error("No tables found in the database. Exiting."))
            return
        
        # Present table choices to the user
        print(info("Available tables:"))
        for i, table in enumerate(all_tables, 1):
            print(f"{i}. {table}")
        
        # Ask user to choose a table
        while True:
            try:
                choice = int(input("Enter the number of the table you want to analyze: "))
                if 1 <= choice <= len(all_tables):
                    selected_table = all_tables[choice - 1]
                    break
                else:
                    print(error("Invalid choice. Please enter a number from the list."))
            except ValueError:
                print(error("Invalid input. Please enter a number."))
        
        print(info(f"You've selected to analyze the '{selected_table}' table."))
        
        # Analyze the selected table
        self.analyze_table(selected_table)
            
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        
        self.save_text_output()
        pdf_path = self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch 5) completed. Results saved in {self.output_folder}"))
        return pdf_path
      
