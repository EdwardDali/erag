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
            self.interpret_results("ETSF Analysis", error_message, table_name)

    def save_results(self, analysis_type, results):
        results_file = os.path.join(self.output_folder, f"{analysis_type.lower().replace(' ', '_')}_results.txt")
        with open(results_file, "w", encoding='utf-8') as f:
            f.write(f"Results for {analysis_type}:\n")
            if isinstance(results, dict):
                for key, value in results.items():
                    if key != 'image_paths':
                        f.write(f"{key}: {value}\n")
            else:
                f.write(str(results))
        print(success(f"Results saved as txt file: {results_file}"))

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

        # Save the results
        self.save_results(analysis_type, results)

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
        output_file = os.path.join(self.output_folder, "ida_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def save_results_as_txt(self):
        output_file = os.path.join(self.output_folder, f"ida_{self.table_name}_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(f"Innovative Data Analysis Results for {self.table_name}\n\n")
            f.write("Key Findings:\n")
            for finding in self.findings:
                f.write(f"- {finding}\n")
            f.write("\nDetailed Analysis Results:\n")
            f.write(self.text_output)
        print(success(f"Results saved as txt file: {output_file}"))

    def generate_pdf_report(self):
        report_title = f"Innovative Data Analysis Report for {self.table_name}"
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.findings,
            self.pdf_content,
            self.image_data,
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
