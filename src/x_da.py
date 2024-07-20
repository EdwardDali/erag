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

class ExploratoryDataAnalysis:
    def __init__(self, erag_api, db_path):
        self.erag_api = erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 5
        self.output_folder = os.path.join(settings.output_folder, "xda_output")
        os.makedirs(self.output_folder, exist_ok=True)
        self.text_output = ""
        self.pdf_content = []
        self.findings = []
        self.llm_name = self.erag_api.model
        self.toc_entries = []
        self.executive_summary = ""
        self.image_paths = []
        self.max_pixels = 400000
        self.timeout_seconds = 10
        self.image_data = []

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
        print(info(f"Starting Exploratory Data Analysis on {self.db_path}"))
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Exploratory Data Analysis completed. Results saved in {self.output_folder}"))

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        analysis_methods = [
            self.basic_statistics,
            self.data_types_and_missing_values,
            self.numerical_features_distribution,
            self.correlation_analysis,
            self.categorical_features_analysis
            # Target vs Features Analysis removed
        ]

        for method in analysis_methods:
            method(df, table_name)

    def basic_statistics(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Basic Statistics"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        for col in numerical_columns:
            data = df[col].dropna()
            
            # Calculate statistics
            mean = np.mean(data)
            std_dev = np.std(data, ddof=1)
            median = np.median(data)
            variance = np.var(data, ddof=1)
            skewness = data.skew()
            kurtosis = data.kurt()
            n = len(data)
            ci_mean = [mean - 1.96 * (std_dev / np.sqrt(n)), mean + 1.96 * (std_dev / np.sqrt(n))]
            ci_median = [np.percentile(data, 25), np.percentile(data, 75)]
            ci_std_dev = [std_dev - 1.96 * (std_dev / np.sqrt(2 * (n - 1))), std_dev + 1.96 * (std_dev / np.sqrt(2 * (n - 1)))]
            ad_stat, _, ad_significance = anderson(data)

            # Histogram with normal curve
            def plot_histogram():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.histplot(data, kde=False, bins=30, ax=ax, color='skyblue', edgecolor='black')
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mean, std_dev)
                ax.plot(x, p * len(data) * (xmax - xmin) / 30, 'r--', linewidth=2)
                ax.set_title(f'{col}: Histogram with Normal Curve')
                ax.set_xlabel('Data')
                ax.set_ylabel('Frequency')
                return fig, ax

            result = self.generate_plot(plot_histogram)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_histogram.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping histogram for {col} due to timeout.")

            # Boxplot
            def plot_boxplot():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.boxplot(x=data, ax=ax)
                ax.set_title(f'{col}: Boxplot')
                ax.set_xlabel('Data')
                return fig, ax

            result = self.generate_plot(plot_boxplot)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_boxplot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping boxplot for {col} due to timeout.")

            # Confidence Intervals
            def plot_confidence_intervals():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.errorbar(mean, 0, xerr=[[mean - ci_mean[0]], [ci_mean[1] - mean]], fmt='o', color='blue', label='Mean')
                ax.errorbar(median, 0, xerr=[[median - ci_median[0]], [ci_median[1] - median]], fmt='o', color='green', label='Median')
                ax.errorbar(std_dev, 0, xerr=[[std_dev - ci_std_dev[0]], [ci_std_dev[1] - std_dev]], fmt='o', color='red', label='Std Dev')
                ax.legend()
                ax.set_title(f'{col}: 95% Confidence Intervals')
                ax.set_yticks([])
                return fig, ax

            result = self.generate_plot(plot_confidence_intervals)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_confidence_intervals.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping confidence intervals for {col} due to timeout.")

            # Anderson-Darling Normality Test
            def plot_anderson_darling():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                summary_text = f"""
                Anderson-Darling Normality Test
                A-Squared: {ad_stat:.2f}
                P-Value: {ad_significance[2] if ad_stat < ad_significance[2] else ad_significance[-1]:.3f}

                Mean: {mean:.2f}
                StDev: {std_dev:.4f}
                Variance: {variance:.4f}
                Skewness: {skewness:.6f}
                Kurtosis: {kurtosis:.6f}
                N: {n}

                Minimum: {np.min(data):.2f}
                1st Quartile: {np.percentile(data, 25):.2f}
                Median: {median:.2f}
                3rd Quartile: {np.percentile(data, 75):.2f}
                Maximum: {np.max(data):.2f}

                95% CI for Mean: {ci_mean[0]:.4f}, {ci_mean[1]:.4f}
                95% CI for Median: {ci_median[0]:.4f}, {ci_median[1]:.4f}
                95% CI for Std Dev: {ci_std_dev[0]:.4f}, {ci_std_dev[1]:.4f}
                """
                ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
                ax.set_title(f'{col}: Anderson-Darling Normality Test')
                ax.axis('off')
                return fig, ax

            result = self.generate_plot(plot_anderson_darling)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_anderson_darling.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Anderson-Darling test for {col} due to timeout.")

        self.interpret_results("Basic Statistics", image_paths, table_name)

    def data_types_and_missing_values(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Data Types and Missing Values"))
        data_types = df.dtypes.to_frame(name='Data Type')
        missing_values = df.isnull().sum().to_frame(name='Missing Values')
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_frame(name='Missing Percentage')
        results = pd.concat([data_types, missing_values, missing_percentage], axis=1)
        self.interpret_results(f"{self.technique_counter}. Data Types and Missing Values", results, table_name)

    def numerical_features_distribution(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Numerical Features Distribution"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        if len(numerical_columns) > 0:
            for col in numerical_columns:
                data = df[col].dropna()
                
                # Histogram with KDE
                def plot_histogram_kde():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.histplot(data, kde=True, ax=ax, color='skyblue', edgecolor='black')
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    return fig, ax

                result = self.generate_plot(plot_histogram_kde)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{col}_distribution.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping histogram with KDE for {col} due to timeout.")
                
                # Q-Q plot
                def plot_qq():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    probplot(data, dist="norm", plot=ax)
                    ax.set_title(f'Q-Q Plot of {col}')
                    return fig, ax

                result = self.generate_plot(plot_qq)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{col}_qq_plot.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping Q-Q plot for {col} due to timeout.")
            
            self.interpret_results("Numerical Features Distribution", image_paths, table_name)
        else:
            self.interpret_results("Numerical Features Distribution", "N/A - No numerical features found", table_name)

    def correlation_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Correlation Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            correlation_matrix = df[numerical_columns].corr()
            
            def plot_correlation_heatmap():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
                ax.set_title('Correlation Matrix Heatmap')
                return fig, ax

            result = self.generate_plot(plot_correlation_heatmap)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_correlation_matrix.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Correlation Analysis", img_path, table_name)
            else:
                print("Skipping correlation heatmap due to timeout.")
                self.interpret_results("Correlation Analysis", "N/A - Correlation heatmap generation timed out", table_name)
        else:
            self.interpret_results("Correlation Analysis", "N/A - Not enough numerical features for correlation analysis", table_name)

    def categorical_features_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Categorical Features Analysis"))
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        image_paths = []
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                def plot_categorical_distribution():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    value_counts = df[col].value_counts().nlargest(10)  # Get top 10 categories
                    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette='Set3')
                    ax.set_title(f'Top 10 Categories in {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Count')
                    ax.tick_params(axis='x', rotation=45)
                    for i, v in enumerate(value_counts.values):
                        ax.text(i, v, str(v), ha='center', va='bottom')
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_categorical_distribution)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{col}_top10_categorical_distribution.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping categorical distribution for {col} due to timeout.")
            
            self.interpret_results("Categorical Features Analysis (Top 10 Categories)", image_paths, table_name)
        else:
            self.interpret_results("Categorical Features Analysis", "N/A - No categorical features found", table_name)

    
    def interpret_results(self, analysis_type, results, table_name):
        if isinstance(results, pd.DataFrame):
            results_str = f"DataFrame with shape {results.shape}:\n{results.to_string()}"
        elif isinstance(results, tuple) and isinstance(results[0], pd.DataFrame):
            results_str = f"DataFrame with shape {results[0].shape}:\n{results[0].to_string()}\nImage path: {results[1]}"
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
        interpretation = self.erag_api.chat([{"role": "system", "content": "You are a data analyst providing insights on exploratory data analysis results. Respond in the requested format."}, 
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
        5. Verifying and improving the SQLite statements if necessary.

        Provide your response in the same format, maintaining the original structure. 
        Do not add comments, questions, or explanations about the changes - simply provide the improved version.
        """

        enhanced_interpretation = self.erag_api.chat([
            {"role": "system", "content": "You are a data analyst improving interpretations of exploratory data analysis results. Provide direct enhancements without adding meta-comments."},
            {"role": "user", "content": check_prompt}
        ])

        print(success(f"AI Interpretation for {analysis_type}:"))
        print(enhanced_interpretation.strip())
        
        self.text_output += f"\n{enhanced_interpretation.strip()}\n\n"
        
        # Handle images
        image_data = []
        if isinstance(results, tuple) and len(results) == 2 and isinstance(results[1], str) and results[1].endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_data.append((f"{analysis_type} - Image", results[1]))
        elif isinstance(results, list):
            for i, item in enumerate(results):
                if isinstance(item, str) and item.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_data.append((f"{analysis_type} - Image {i+1}", item))
                elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str) and item[1].endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_data.append((f"{analysis_type} - {item[0]}", item[1]))
        
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
            self.executive_summary = "No significant findings were identified during the analysis. This could be due to a lack of data, uniform data distribution, or absence of notable patterns or anomalies in the dataset."
            return

        summary_prompt = f"""
        Based on the following findings from the Exploratory Data Analysis:
        
        {self.findings}
        
        Please provide an executive summary of the analysis. The summary should:
        1. Briefly introduce the purpose of the analysis.
        2. Highlight the most significant insights and patterns discovered.
        3. Mention any potential issues or areas that require further investigation.
        4. Conclude with recommendations for next steps or areas to focus on.

        Structure the summary in multiple paragraphs for readability.
        Please provide your response in plain text format, without any special formatting or markup.
        """
        
        try:
            interpretation = self.erag_api.chat([
                {"role": "system", "content": "You are a data analyst providing an executive summary of an exploratory data analysis. Respond in plain text format."},
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

                enhanced_summary = self.erag_api.chat([
                    {"role": "system", "content": "You are a data analyst improving an executive summary of an exploratory data analysis. Provide direct enhancements without adding meta-comments."},
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
        output_file = os.path.join(self.output_folder, "xda_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name)
        pdf_file = pdf_generator.create_enhanced_pdf_report(
            self.executive_summary,
            self.findings,
            self.pdf_content,
            self.image_data  # Pass self.image_data to the PDF generator
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
        else:
            print(error("Failed to generate PDF report"))
