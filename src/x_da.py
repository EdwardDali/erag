import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, anderson, pearsonr, probplot, iqr, zscore
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
from src.helper_da import get_technique_info
import inspect

class TimeoutException(Exception):
    pass

class ExploratoryDataAnalysis:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 12  # Updated to reflect new techniques
        self.table_name = None
        self.output_folder = None
        self.text_output = ""
        self.pdf_content = []
        self.findings = []
        self.llm_name = f"Worker: {self.worker_erag_api.model}, Supervisor: {self.supervisor_erag_api.model}"
        self.toc_entries = []
        self.image_paths = []
        self.max_pixels = 400000
        self.timeout_seconds = 10
        self.image_data = []
        self.pdf_generator = None
        self.selected_table = None 

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
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Exploratory Data Analysis completed. Results saved in {self.output_folder}"))

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"xda_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        analysis_methods = [
            self.data_summary,
            self.detailed_statistics_summary,
            self.null_missing_unique_analysis,
            self.column_importance_analysis,
            self.basic_statistics,
            self.data_types_and_missing_values,
            self.numerical_features_distribution,
            self.correlation_analysis,
            self.categorical_features_analysis,
            self.outlier_detection,
            self.data_quality_report,
            self.hypothesis_testing_suggestions
        ]

        for method in analysis_methods:
            try:
                method(df, table_name)
            except Exception as e:
                error_message = f"An error occurred during {method.__name__}: {str(e)}"
                print(error(error_message))
                self.text_output += f"\n{error_message}\n"
                # Optionally, add this error to the PDF report
                self.pdf_content.append((method.__name__, [], error_message))
            finally:
                # Ensure we always increment the technique counter, even if the method fails
                self.technique_counter += 0.5


    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]


 

    def basic_statistics(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Basic Statistics"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        results = {}
        
        for col in numerical_columns:
            data = df[col].dropna()
            
            # Calculate statistics
            mean = np.mean(data)
            std_dev = np.std(data, ddof=1)
            median = np.median(data)
            
            results[col] = {
                'mean': mean,
                'std_dev': std_dev,
                'median': median
            }
            
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
                image_paths.append((f"{col} Histogram", img_path))

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
                image_paths.append((f"{col} Boxplot", img_path))

        results['image_paths'] = image_paths
        self.interpret_results("Basic Statistics", results, table_name)

    def data_types_and_missing_values(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Data Types and Missing Values"))
        data_types = df.dtypes.to_frame(name='Data Type')
        missing_values = df.isnull().sum().to_frame(name='Missing Values')
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_frame(name='Missing Percentage')
        results = pd.concat([data_types, missing_values, missing_percentage], axis=1)
        self.interpret_results(f"{self.technique_counter}. Data Types and Missing Values", results, table_name)

    def numerical_features_distribution(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Numerical Features Distribution"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        results = {}
        
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
                    image_paths.append((f"{col} Distribution", img_path))
                
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
                    image_paths.append((f"{col} Q-Q Plot", img_path))
            
            results['image_paths'] = image_paths
            self.interpret_results("Numerical Features Distribution", results, table_name)
        else:
            self.interpret_results("Numerical Features Distribution", "N/A - No numerical features found", table_name)

    def correlation_analysis(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Correlation Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        
        if len(numerical_columns) > 1:
            correlation_matrix = df[numerical_columns].corr()
            results['correlation_matrix'] = correlation_matrix.to_dict()
            
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
                results['image_paths'] = [("Correlation Matrix Heatmap", img_path)]
            
            self.interpret_results("Correlation Analysis", results, table_name)
        else:
            self.interpret_results("Correlation Analysis", "N/A - Not enough numerical features for correlation analysis", table_name)

    def categorical_features_analysis(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Categorical Features Analysis"))
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        image_paths = []
        results = {}
        
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                results[col] = df[col].value_counts().to_dict()
                
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
                    image_paths.append((f"{col} Top 10 Categories", img_path))
            
            results['image_paths'] = image_paths
            self.interpret_results("Categorical Features Analysis (Top 10 Categories)", results, table_name)
        else:
            self.interpret_results("Categorical Features Analysis", "N/A - No categorical features found", table_name)


    def data_summary(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Data Summary"))
        
        summary = {
            "Number of rows": len(df),
            "Number of columns": len(df.columns),
            "Column names and data types": df.dtypes.to_dict(),
            "Preview": df.head().to_string()
        }
        
        # Create a bar plot of column types
        def plot_column_types():
            dtype_counts = df.dtypes.value_counts()
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            dtype_counts.plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Column Types')
            ax.set_xlabel('Data Type')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            return fig, ax

        result = self.generate_plot(plot_column_types)
        if result is not None:
            fig, ax = result
            img_path = os.path.join(self.output_folder, f"{table_name}_column_types.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            summary['image_paths'] = [("Data Summary - Column Types", img_path)]
        
        self.interpret_results("Data Summary", summary, table_name)

    def detailed_statistics_summary(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Detailed Statistics Summary"))
        
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        numeric_stats = df[numeric_columns].agg(['mean', 'median', 'std', 'min', 'max'])
        percentiles = df[numeric_columns].quantile([0.25, 0.5, 0.75])
        percentiles.index = ['25th Percentile', '50th Percentile', '75th Percentile']
        numeric_stats = pd.concat([numeric_stats, percentiles])
        
        categorical_stats = {col: df[col].value_counts().to_dict() for col in categorical_columns}
        
        stats_summary = {
            "Numeric Statistics": numeric_stats.to_dict(),
            "Categorical Statistics": categorical_stats
        }
        
        # Create a box plot for numeric columns
        def plot_numeric_boxplot():
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            df[numeric_columns].boxplot(ax=ax)
            ax.set_title('Box Plot of Numeric Columns')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            return fig, ax

        result = self.generate_plot(plot_numeric_boxplot)
        if result is not None:
            fig, ax = result
            img_path = os.path.join(self.output_folder, f"{table_name}_numeric_boxplot.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            stats_summary['image_paths'] = [("Detailed Statistics Summary - Numeric Box Plot", img_path)]
        
        self.interpret_results("Detailed Statistics Summary", stats_summary, table_name)

    def null_missing_unique_analysis(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Null, Missing, and Unique Value Analysis"))
        
        null_counts = df.isnull().sum()
        null_percentages = (df.isnull().sum() / len(df)) * 100
        unique_counts = df.nunique()
        
        analysis = {
            "Null Counts": null_counts.to_dict(),
            "Null Percentages": null_percentages.to_dict(),
            "Unique Value Counts": unique_counts.to_dict()
        }
        
        # Create a heatmap of missing values
        def plot_missing_values_heatmap():
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
            ax.set_title('Missing Value Heatmap')
            return fig, ax

        result = self.generate_plot(plot_missing_values_heatmap)
        if result is not None:
            fig, ax = result
            img_path = os.path.join(self.output_folder, f"{table_name}_missing_values_heatmap.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            analysis['image_paths'] = [("Null, Missing, and Unique Value Analysis - Missing Value Heatmap", img_path)]
        
        self.interpret_results("Null, Missing, and Unique Value Analysis", analysis, table_name)

    def column_importance_analysis(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Column Importance Analysis"))
        
        # This is a simplified version. In practice, you might want to use more sophisticated
        # methods to determine column importance, such as correlation with a target variable
        # or feature importance from a machine learning model.
        
        analysis = {col: {
            "dtype": str(df[col].dtype),
            "unique_count": df[col].nunique(),
            "null_percentage": (df[col].isnull().sum() / len(df)) * 100
        } for col in df.columns}
        
        self.interpret_results("Column Importance Analysis", analysis, table_name)

    def outlier_detection(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Outlier Detection"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        results = {}
        
        for col in numerical_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            results[col] = {
                'outliers_count': outliers,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            def plot_boxplot_with_outliers():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f'Boxplot with Outliers for {col}')
                ax.set_xlabel(col)
                return fig, ax

            result = self.generate_plot(plot_boxplot_with_outliers)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_boxplot_outliers.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append((f"{col} Boxplot with Outliers", img_path))
        
        results['image_paths'] = image_paths
        self.interpret_results("Outlier Detection", results, table_name)

    def data_quality_report(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Data Quality Report"))
        
        quality_report = {
            "Missing Values": df.isnull().sum().to_dict(),
            "Missing Percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
            "Duplicate Rows": df.duplicated().sum(),
            "Data Types": df.dtypes.to_dict(),
            "Unique Values": df.nunique().to_dict()
        }
        
        # Create a bar plot of missing value percentages
        def plot_missing_percentages():
            missing_percentages = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            missing_percentages.plot(kind='bar', ax=ax)
            ax.set_title('Percentage of Missing Values by Column')
            ax.set_xlabel('Columns')
            ax.set_ylabel('Percentage Missing')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig, ax

        result = self.generate_plot(plot_missing_percentages)
        if result is not None:
            fig, ax = result
            img_path = os.path.join(self.output_folder, f"{table_name}_missing_percentages.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            quality_report['image_paths'] = [("Missing Values Percentage", img_path)]
        else:
            quality_report['image_paths'] = []
        
        self.interpret_results("Data Quality Report", quality_report, table_name)

    def hypothesis_testing_suggestions(self, df, table_name):
        self.technique_counter += 0.5
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hypothesis Testing Suggestions"))
        
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        suggestions = []
        
        if len(numeric_columns) >= 2:
            suggestions.append({
                "test": "Correlation test",
                "variables": f"{numeric_columns[0]} and {numeric_columns[1]}",
                "description": f"Test the correlation between {numeric_columns[0]} and {numeric_columns[1]} to determine if there's a significant relationship."
            })
        
        if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            suggestions.append({
                "test": "ANOVA",
                "variables": f"{numeric_columns[0]} across categories of {categorical_columns[0]}",
                "description": f"Compare the means of {numeric_columns[0]} across different categories of {categorical_columns[0]} to see if there are significant differences."
            })
        
        if len(numeric_columns) >= 1:
            suggestions.append({
                "test": "One-sample t-test",
                "variables": numeric_columns[0],
                "description": f"Compare the mean of {numeric_columns[0]} to a hypothesized value to determine if it's significantly different."
            })
        
        results = {
            "suggestions": suggestions,
            "image_paths": []  # No images for this analysis
        }
        
        self.interpret_results("Hypothesis Testing Suggestions", results, table_name)

  

    
    

    def interpret_results(self, analysis_type, results, table_name):
        technique_info = get_technique_info(analysis_type)

        if isinstance(results, dict) and "Numeric Statistics" in results:
            numeric_stats = results["Numeric Statistics"]
            categorical_stats = results["Categorical Statistics"]
            
            numeric_table = "| Statistic | " + " | ".join(numeric_stats.keys()) + " |\n"
            numeric_table += "| --- | " + " | ".join(["---" for _ in numeric_stats.keys()]) + " |\n"
            for stat in numeric_stats[list(numeric_stats.keys())[0]].keys():
                numeric_table += f"| {stat} | " + " | ".join([f"{numeric_stats[col][stat]:.2f}" for col in numeric_stats.keys()]) + " |\n"
            
            categorical_summary = "\n".join([f"{col}:\n" + "\n".join([f"  - {value}: {count}" for value, count in stats.items()]) for col, stats in categorical_stats.items()])
            
            results_str = f"Numeric Statistics:\n{numeric_table}\n\nCategorical Statistics:\n{categorical_summary}"
        elif isinstance(results, pd.DataFrame):
            results_str = f"DataFrame with shape {results.shape}:\n{results.to_string()}"
        elif isinstance(results, dict):
            results_str = "\n".join([f"{k}: {v}" for k, v in results.items() if k != 'image_paths'])
        else:
            results_str = str(results)

        # Add information about number of visualizations
        num_visualizations = len(results.get('image_paths', []))
        results_str += f"\n\nNumber of visualizations created: {num_visualizations}"

        common_prompt = f"""
        Analysis type: {analysis_type}
        Table name: {table_name}

        Technique Context:
        {technique_info['context']}

        Results:
        {results_str}

        Interpretation Guidelines:
        {technique_info['guidelines']}
        """

        worker_prompt = f"""
        You are an expert data analyst providing insights on exploratory data analysis results. Your task is to interpret the following analysis results and provide a detailed, data-driven interpretation, focusing on discovering patterns and hidden insights. Avoid jargon.

        {common_prompt}

        Please provide a thorough interpretation of these results, highlighting noteworthy patterns, anomalies, or insights. Focus on aspects that would be valuable for business decisions and operational improvements. Always provide specific numbers and percentages.

        Structure your response in the following format:

        1. Analysis performed and Key Insights:
        [Briefly describe the analysis performed. List at least 2-3 important insights discovered, with relevant numbers and percentages. Provide detailed explanations for each insight.]

        2. Patterns and Trends:
        [Describe at least 2-3 significant patterns or trends observed in the data. Explain their potential significance.]

        3. Potential Issues:
        [Highlight any anomalies, unusual trends, or areas of concern. Mention at least 2-3 potential problems, red flags, audit findings, fraud cases always including relevant numbers and percentages.]

        Ensure your interpretation is comprehensive and focused on actionable insights. While you can be detailed, strive for clarity in your explanations. Use technical terms when necessary, but provide brief explanations for complex concepts.

        Interpretation:
        """

        worker_interpretation = self.worker_erag_api.chat([{"role": "system", "content": "You are an expert data analyst providing insights for business leaders and analysts. Respond in the requested format."}, 
                                                    {"role": "user", "content": worker_prompt}])

        supervisor_prompt = f"""
        You are an expert data analyst providing insights on exploratory data analysis results. Your task is to interpret the following analysis results and provide a detailed, data-driven interpretation.

        {common_prompt}

        Please provide a thorough interpretation of these results, highlighting noteworthy patterns, anomalies, or insights. Focus on the most important aspects that would be valuable for business operations and decision-making. Always provide specific numbers and percentages when discussing findings.
        If some data appears to be missing or incomplete, work with the available information without mentioning the limitations. Your goal is to extract as much insight as possible from the given data.
        Structure your response in the following format:
        1. Analysis:
        [Provide a detailed description of the analysis performed, including specific metrics and their values]
        2. Key Findings:
        [List the most important discoveries, always including relevant numbers and percentages]
        3. Implications:
        [Discuss the potential impact of these findings on business operations and decision-making]
        4. Operational Recommendations:
        [Suggest concrete operational steps or changes based on these results. Focus on actionable recommendations that can improve business processes, efficiency, or outcomes. Avoid recommending further data analysis.]
        Ensure your interpretation is concise yet comprehensive, focusing on actionable insights derived from the data that can be directly applied to business operations.

        Business Analysis:
        """

        supervisor_analysis = self.supervisor_erag_api.chat([
            {"role": "system", "content": "You are a senior business analyst providing insights based on data analysis results. Provide a concise yet comprehensive business analysis."},
            {"role": "user", "content": supervisor_prompt}
        ])

        combined_interpretation = f"""
        Data Analysis:
        {worker_interpretation.strip()}

        Business Analysis:
        {supervisor_analysis.strip()}
        """

        print(success(f"Combined Interpretation for {analysis_type}:"))
        print(combined_interpretation.strip())

        self.text_output += f"\n{combined_interpretation.strip()}\n\n"

        # Handle images for the PDF report
        image_data = []
        if isinstance(results, dict) and 'image_paths' in results:
            for img in results['image_paths']:
                if isinstance(img, tuple) and len(img) == 2:
                    image_data.append(img)
                elif isinstance(img, str):
                    image_data.append((analysis_type, img))

        # Prepare content for PDF report
        pdf_content = f"""
        # {analysis_type}

        ## Data Analysis
        {worker_interpretation.strip()}

        <br><br>

        ## Business Analysis
        {supervisor_analysis.strip()}
        """

        self.pdf_content.append((analysis_type, image_data, pdf_content))

        # Extract important findings
        self.findings.append(f"{analysis_type}:")
        lines = combined_interpretation.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith("1. Analysis performed and Key Insights:") or line.startswith("2. Key Findings:"):
                for finding in lines[i+1:]:
                    if finding.strip() and not finding.startswith(("2.", "3.", "4.")):
                        self.findings.append(finding.strip())
                    elif finding.startswith(("2.", "3.", "4.")):
                        break

        # Update self.image_data for the PDF report
        self.image_data.extend(image_data)



    def save_text_output(self):
        output_file = os.path.join(self.output_folder, "xda_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Exploratory Data Analysis Report for {self.table_name}"
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.findings,
            self.pdf_content,
            self.image_data,
            filename=f"xda_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None
