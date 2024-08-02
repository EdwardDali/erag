import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy.stats import rankdata
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from src.helper_da import get_technique_info

class TimeoutException(Exception):
    pass

class InnovativeDataAnalysis:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 2  # AMPR and ETSF
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
        

    def calculate_figure_size(self, data=None, aspect_ratio=16/9):
        if data is not None:
            rows, cols = data.shape
            base_size = 8
            width = base_size * min(cols, 10) / 5
            height = base_size * min(rows, 20) / 10
        else:
            max_width = int(np.sqrt(self.max_pixels * aspect_ratio))
            max_height = int(max_width / aspect_ratio)
            width, height = max_width / 100, max_height / 100
        return (width, height)

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
        print(info(f"Starting Innovative Data Analysis on {self.db_path}"))
        
        all_tables = self.get_tables()
        user_tables = [table for table in all_tables if table.lower() not in ['information_schema', 'sqlite_master', 'sqlite_sequence', 'sqlite_stat1']]
        
        if not user_tables:
            print(error("No user tables found in the database. Exiting."))
            return
        
        selected_table = user_tables[0]  # Automatically select the first user table
        
        print(info(f"Analyzing table: '{selected_table}'"))
        
        self.analyze_table(selected_table)
        
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        
        self.save_text_output()
        self.save_results_as_txt()
        self.generate_pdf_report()
        print(success(f"Innovative Data Analysis completed. Results saved in {self.output_folder}"))


    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"ida_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        analysis_methods = [
            self.ampr_analysis,
            self.etsf_analysis
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

    def save_unsuccessful_techniques(self):
        # Get the parent folder of the current output folder
        parent_folder = os.path.dirname(self.output_folder)
        filename = os.path.join(parent_folder, "unsuccessful_techniques.txt")
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a') as f:
            if not file_exists:
                f.write("Unsuccessful techniques log\n")
                f.write("="*50 + "\n\n")
            
            # Write a header for this batch
            f.write(f"Innovative Data Analysis - {self.table_name}\n")
            f.write("-"*30 + "\n")
            
            for technique in self.unsuccessful_techniques:
                f.write(f"{technique}\n")
            
            f.write("\n")  # Add a blank line for separation between different runs
        
        print(info(f"Unsuccessful techniques appended to {filename}"))

    def ampr_analysis(self, df, table_name):
        self.technique_counter += 1
        print(f"Performing test {self.technique_counter}/{self.total_techniques} - Adaptive Multi-dimensional Pattern Recognition (AMPR)")

        try:
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            X = df[numeric_columns].dropna()

            if X.empty:
                raise ValueError("No numeric data available for AMPR analysis.")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA()
            pca.fit(X_scaled)
            cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
            X_pca = pca.transform(X_scaled)[:, :n_components]

            best_silhouette = -1
            best_eps = 0.1
            for eps in np.arange(0.1, 1.1, 0.1):
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=5)
                    labels = dbscan.fit_predict(X_pca)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X_pca, labels)
                        if score > best_silhouette:
                            best_silhouette = score
                            best_eps = eps
                except Exception as e:
                    print(f"Error during clustering with eps={eps}: {str(e)}")

            if best_eps > 0:
                dbscan = DBSCAN(eps=best_eps, min_samples=5)
                cluster_labels = dbscan.fit_predict(X_pca)
            else:
                print("Could not find suitable clustering parameters. Using default KMeans clustering.")
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=3)
                cluster_labels = kmeans.fit_predict(X_pca)

            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = isolation_forest.fit_predict(X_pca)

            # Cluster Analysis Plot
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
            ax.set_title('Cluster Analysis')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            cluster_img_path = os.path.join(self.output_folder, f"{table_name}_ampr_cluster_analysis.png")
            plt.savefig(cluster_img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Feature Correlation Plot
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            sns.heatmap(pd.DataFrame(X_pca).corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Feature Correlation')
            correlation_img_path = os.path.join(self.output_folder, f"{table_name}_ampr_feature_correlation.png")
            plt.savefig(correlation_img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Anomaly Detection Plot
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_labels, cmap='RdYlGn')
            ax.set_title('Anomaly Detection')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            anomaly_img_path = os.path.join(self.output_folder, f"{table_name}_ampr_anomaly_detection.png")
            plt.savefig(anomaly_img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # PCA Explained Variance Plot
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            explained_variance_ratio = pca.explained_variance_ratio_[:n_components]
            ax.bar(range(1, n_components + 1), explained_variance_ratio)
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA Explained Variance')
            pca_img_path = os.path.join(self.output_folder, f"{table_name}_ampr_pca_explained_variance.png")
            plt.savefig(pca_img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            results = {
                "n_components": n_components,
                "best_eps": best_eps,
                "best_silhouette_score": best_silhouette,
                "n_clusters": len(np.unique(cluster_labels)),
                "n_anomalies": sum(anomaly_labels == -1),
                "image_paths": [
                    ("AMPR Analysis - Cluster Analysis", cluster_img_path),
                    ("AMPR Analysis - Feature Correlation", correlation_img_path),
                    ("AMPR Analysis - Anomaly Detection", anomaly_img_path),
                    ("AMPR Analysis - PCA Explained Variance", pca_img_path)
                ]
            }

            self.interpret_results("AMPR Analysis", results, table_name)

        except Exception as e:
            error_message = f"An error occurred during AMPR Analysis: {str(e)}"
            print(error(error_message))
            self.unsuccessful_techniques.append("AMPR Analysis")
            self.interpret_results("AMPR Analysis", error_message, table_name)

    def etsf_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Enhanced Time Series Forecasting (ETSF)"))

        try:
            date_columns = df.select_dtypes(include=['datetime64']).columns
            if len(date_columns) == 0:
                raise ValueError("No date column found for ETSF analysis.")

            date_column = date_columns[0]
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found for ETSF analysis.")

            value_column = numeric_columns[0]

            df = df[[date_column, value_column]].dropna().sort_values(date_column)
            df = df.set_index(date_column)

            if len(df) < 30:
                raise ValueError("Not enough data for meaningful ETSF analysis.")

            df['lag_1'] = df[value_column].shift(1)
            df['lag_7'] = df[value_column].shift(7)
            df['fourier_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
            df['fourier_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
            df = df.dropna()

            result = adfuller(df[value_column].dropna())
            adf_result = {
                'ADF Statistic': result[0],
                'p-value': result[1],
                'Critical Values': result[4]
            }

            decomposition = seasonal_decompose(df[value_column], model='additive', period=min(len(df), 365))
            
            # Observed Data Plot
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ax.plot(decomposition.observed)
            ax.set_title('Observed Data')
            observed_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_observed_data.png")
            plt.savefig(observed_img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Trend Plot
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ax.plot(decomposition.trend)
            ax.set_title('Trend')
            trend_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_trend.png")
            plt.savefig(trend_img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Seasonal Plot
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ax.plot(decomposition.seasonal)
            ax.set_title('Seasonal')
            seasonal_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_seasonal.png")
            plt.savefig(seasonal_img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Residual Plot
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ax.plot(decomposition.resid)
            ax.set_title('Residual')
            residual_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_residual.png")
            plt.savefig(residual_img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Split the data into train and test sets
            train_size = int(len(df) * 0.8)
            train, test = df[:train_size], df[train_size:]

            model = ARIMA(train[value_column], order=(1,1,1), 
                          exog=train[['lag_1', 'lag_7', 'fourier_sin', 'fourier_cos']])
            results = model.fit()

            predictions = results.forecast(steps=len(test), 
                                           exog=test[['lag_1', 'lag_7', 'fourier_sin', 'fourier_cos']])

            # Forecast Plot
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ax.plot(df.index, df[value_column], label='Actual')
            ax.plot(test.index, predictions, label='Forecast', color='red')
            ax.set_title('ETSF Forecast vs Actual')
            ax.legend()
            forecast_img_path = os.path.join(self.output_folder, f"{table_name}_etsf_forecast.png")
            plt.savefig(forecast_img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            mse = mean_squared_error(test[value_column], predictions)
            rmse = np.sqrt(mse)

            results = {
                "adf_result": adf_result,
                "mse": mse,
                "rmse": rmse,
                "image_paths": [
                    ("ETSF Analysis - Observed Data", observed_img_path),
                    ("ETSF Analysis - Trend", trend_img_path),
                    ("ETSF Analysis - Seasonal", seasonal_img_path),
                    ("ETSF Analysis - Residual", residual_img_path),
                    ("ETSF Analysis - Forecast", forecast_img_path)
                ]
            }

            self.interpret_results("ETSF Analysis", results, table_name)

        except Exception as e:
            error_message = f"An error occurred during ETSF Analysis: {str(e)}"
            print(error(error_message))
            self.unsuccessful_techniques.append("ETSF Analysis")
            self.interpret_results("ETSF Analysis", error_message, table_name)

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


        prompt = f"""
        You are an expert data analyst providing insights on exploratory data analysis results. Your task is to interpret the following analysis results and provide a detailed, data-driven interpretation. focusing on discovering patterns and hidden insights. Avoid jargon.

        Analysis type: {analysis_type}
        Table name: {table_name}

        Technique Context:
        {technique_info['context']}

        Results:
        {results_str}

        Interpretation Guidelines:
        {technique_info['guidelines']}

        Please provide a thorough interpretation of these results, highlighting noteworthy patterns, anomalies, or insights. Focus on aspects that would be valuable for business decisions and operational improvements. Always provide specific numbers and percentages.

        Structure your response in the following format:

        1. Analysis performed and Key Insights:
        [Briefly describe the analysis performed. List at least 2-3 important insights discovered, with relevant numbers and percentages. Provide detailed explanations for each insight.]

        2. Patterns and Trends:
        [Describe at least 2-3 significant patterns or trends observed in the data. Explain their potential significance.]

        3. Potential Issues:
        [Highlight any anomalies, unusual trends, or areas of concern. Mention at least 2-3 potential problems, red flags, audit findings, fraud cases always including relevant numbers and percentages.]

        4. Business Impact:
        [Discuss how these findings might impact business operations or strategic decisions. Consider both short-term and long-term implications.]

        5. Suggested Actions:
        [Provide at least 2-3 specific, actionable steps for auditors, management or analysts. Explain the rationale behind each suggestion and its expected outcome. Do not focus on suggesting more data analysis techniques but on operational controls.]

        Ensure your interpretation is comprehensive and focused on actionable insights. While you can be detailed, strive for clarity in your explanations. Use technical terms when necessary, but provide brief explanations for complex concepts.

        Interpretation:
        """

        interpretation = self.worker_erag_api.chat([{"role": "system", "content": "You are an expert data analyst providing insights for business leaders and analysts. Respond in the requested format."}, 
                                                    {"role": "user", "content": prompt}])
        
        check_prompt = f"""
        As a senior business analyst enhance the following interpretation of data analysis results. The original data and analysis type are:

        {prompt}

        Previous interpretation:
        {interpretation}

        Improve this interpretation by adding or modifying the interpretation but without adding questions or review statements:
        1. Ensuring all statements are backed by specific data points from the original results.
        2. Expanding on any points that could benefit from more detailed explanation.
        3. Verifying that the interpretation covers all significant aspects of the data, adding any important points that may have been missed.
        4. Strengthening the connections between insights, patterns, and suggested actions.
        5. Ensuring the language is accessible to business leaders while still conveying complex ideas accurately.

        Maintain the same format but aim for a comprehensive yet clear analysis in each section. Focus on providing actionable insights that are well-explained and justified by the data.

        Enhanced Interpretation:
        """

        enhanced_interpretation = self.supervisor_erag_api.chat([
            {"role": "system", "content": "You are a senior business analyst improving interpretations of data analysis results. Provide direct enhancements without meta-comments."},
            {"role": "user", "content": check_prompt}
        ])

        print(success(f"AI Interpretation for {analysis_type}:"))
        print(enhanced_interpretation.strip())
        
        self.text_output += f"\n{enhanced_interpretation.strip()}\n\n"
        
        # Handle images for the PDF report (not for LLM interpretation)
        image_data = []
        if isinstance(results, dict) and 'image_paths' in results:
            for img in results['image_paths']:
                if isinstance(img, tuple) and len(img) == 2:
                    image_data.append(img)
                elif isinstance(img, str):
                    image_data.append((analysis_type, img))
        
        self.pdf_content.append((analysis_type, image_data, enhanced_interpretation.strip()))
        
        # Extract important findings
        lines = enhanced_interpretation.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith("1. Key Patterns and Insights:"):
                for finding in lines[i+1:]:
                    if finding.strip() and not finding.startswith(("2.", "3.", "4.")):
                        self.findings.append(f"{analysis_type}: {finding.strip()}")
                    elif finding.startswith(("2.", "3.", "4.")):
                        break

        # Update self.image_data for the PDF report
        self.image_data.extend(image_data)

    def generate_executive_summary(self):
        if not self.findings:
            self.executive_summary = "No significant findings were identified during the analysis. This could be due to a lack of data, uniform data distribution, or absence of notable patterns or anomalies in the dataset."
            return

        successful_techniques = sum(1 for item in self.pdf_content if len(item[1]) > 0 or not item[2].startswith("An error occurred"))
        failed_techniques = self.total_techniques - successful_techniques

        summary_prompt = f"""
        Based on the following findings from the Data Analysis:
        
        {self.findings}
        
        Additional context:
        - {successful_techniques} out of {self.total_techniques} analysis techniques were successfully completed.
        - {failed_techniques} techniques encountered errors and were skipped.
        
        Please provide an executive summary of the analysis tailored for auditors and business managers. The summary should:
        1. Briefly introduce the purpose of the analysis in the context of auditing or business improvement.
        2. Highlight the most significant patterns, trends, and potential issues discovered.
        3. Identify key areas of concern or opportunity that require immediate attention.
        4. Discuss the potential impact of these findings on business operations, compliance, or strategic decisions.
        5. Provide high-level, actionable recommendations for next steps or areas for deeper investigation.
        6. Briefly mention any limitations of the analysis that might affect decision-making.

        Focus on insights that are most relevant for audit findings or business decisions. Avoid technical jargon and prioritize clear, actionable information. Structure the summary in multiple paragraphs for readability.

        Please provide your response in plain text format, without any special formatting or markup.
        """
        
        try:
            interpretation = self.worker_erag_api.chat([
                {"role": "system", "content": "You are a data analyst providing an executive summary of a data analysis for auditors and business managers. Respond in plain text format."},
                {"role": "user", "content": summary_prompt}
            ])
            
            if interpretation is not None:
                check_prompt = f"""
                Please review and improve the following executive summary:

                {interpretation}

                Enhance the summary by:
                1. Ensuring it clearly communicates the most critical findings and their implications for auditing or business management.
                2. Strengthening the actionable recommendations, making them more specific and tied to the key findings.
                3. Improving the overall narrative flow, ensuring it tells a coherent story about the state of the business or areas requiring audit attention.
                4. Balancing the discussion of risks and opportunities identified in the analysis.

                Provide your response in plain text format, without any special formatting or markup.
                Do not add comments about the changes - simply provide the improved version.
                """

                enhanced_summary = self.supervisor_erag_api.chat([
                    {"role": "system", "content": "You are a senior auditor or business analyst improving an executive summary of a data analysis. Provide direct enhancements without adding meta-comments."},
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
        output_file = os.path.join(self.output_folder, "ida_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def save_results_as_txt(self):
        output_file = os.path.join(self.output_folder, f"ida_{self.table_name}_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(f"Innovative Data Analysis Results for {self.table_name}\n\n")
            f.write("Executive Summary:\n")
            f.write(self.executive_summary)
            f.write("\n\nKey Findings:\n")
            for finding in self.findings:
                f.write(f"- {finding}\n")
            f.write("\nDetailed Analysis Results:\n")
            f.write(self.text_output)
        print(success(f"Results saved as txt file: {output_file}"))

    def generate_pdf_report(self):
        report_title = f"Innovative Data Analysis Report for {self.table_name}"
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.executive_summary,
            self.findings,
            self.pdf_content,
            self.image_data,  # This now contains all image paths
            filename=f"ida_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
        else:
            print(error("Failed to generate PDF report"))

# Example usage
if __name__ == "__main__":
    from src.api_model import EragAPI
    
    worker_api = EragAPI("worker_model_name")
    supervisor_api = EragAPI("supervisor_model_name")
    
    db_path = "path/to/your/database.sqlite"
    
    ida = InnovativeDataAnalysis(worker_api, supervisor_api, db_path)
    
    ida.run()
