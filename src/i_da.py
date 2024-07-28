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
        
        print(info("Available tables:"))
        for i, table in enumerate(user_tables, 1):
            print(f"{i}. {table}")
        
        while True:
            try:
                choice = int(input("Enter the number of the table you want to analyze: "))
                if 1 <= choice <= len(user_tables):
                    selected_table = user_tables[choice - 1]
                    break
                else:
                    print(error("Invalid choice. Please enter a number from the list."))
            except ValueError:
                print(error("Invalid input. Please enter a number."))
        
        print(info(f"You've selected to analyze the '{selected_table}' table."))
        
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
            method(df, table_name)

    def ampr_analysis(self, df, table_name):
        self.technique_counter += 1
        print(f"Performing test {self.technique_counter}/{self.total_techniques} - Adaptive Multi-dimensional Pattern Recognition (AMPR)")

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        X = df[numeric_columns].dropna()

        if X.empty:
            print("No numeric data available for AMPR analysis.")
            return

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

        fig = plt.figure(figsize=self.calculate_figure_size(X_pca))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
        ax1.set_title('Cluster Analysis')
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')

        ax2 = fig.add_subplot(gs[0, 1])
        sns.heatmap(pd.DataFrame(X_pca).corr(), annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Feature Correlation')

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_labels, cmap='RdYlGn')
        ax3.set_title('Anomaly Detection')
        ax3.set_xlabel('Principal Component 1')
        ax3.set_ylabel('Principal Component 2')

        ax4 = fig.add_subplot(gs[1, 1])
        explained_variance_ratio = pca.explained_variance_ratio_[:n_components]
        ax4.bar(range(1, n_components + 1), explained_variance_ratio)
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Explained Variance Ratio')
        ax4.set_title('PCA Explained Variance')

        plt.tight_layout()
        img_path = os.path.join(self.output_folder, f"{table_name}_ampr_analysis.png")
        plt.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close()

        results = {
            "n_components": n_components,
            "best_eps": best_eps,
            "best_silhouette_score": best_silhouette,
            "n_clusters": len(np.unique(cluster_labels)),
            "n_anomalies": sum(anomaly_labels == -1),
            "image_path": img_path
        }

        self.interpret_results("AMPR Analysis", results, table_name)

    def etsf_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Enhanced Time Series Forecasting (ETSF)"))

        date_columns = df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) == 0:
            print(warning("No date column found for ETSF analysis."))
            self.interpret_results("ETSF Analysis", "No date column found", table_name)
            return

        date_column = date_columns[0]
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

        if len(numeric_columns) == 0:
            print(warning("No numeric columns found for ETSF analysis."))
            self.interpret_results("ETSF Analysis", "No numeric columns found", table_name)
            return

        value_column = numeric_columns[0]

        df = df[[date_column, value_column]].dropna().sort_values(date_column)
        df = df.set_index(date_column)

        if len(df) < 30:
            print(warning("Not enough data for meaningful ETSF analysis."))
            self.interpret_results("ETSF Analysis", "Not enough data", table_name)
            return

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
        
        fig = plt.figure(figsize=self.calculate_figure_size(df))
        gs = fig.add_gridspec(4, 1)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(decomposition.observed)
        ax1.set_title('Observed')

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(decomposition.trend)
        ax2.set_title('Trend')

        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(decomposition.seasonal)
        ax3.set_title('Seasonal')

        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(decomposition.resid)
        ax4.set_title('Residual')

        plt.tight_layout()
        decomposition_path = os.path.join(self.output_folder, f"{table_name}_etsf_decomposition.png")
        plt.savefig(decomposition_path, dpi=100, bbox_inches='tight')
        plt.close()

        train_size = int(len(df) * 0.8)
        train, test = df[:train_size], df[train_size:]

        model = ARIMA(train[value_column], order=(1,1,1), 
                      exog=train[['lag_1', 'lag_7', 'fourier_sin', 'fourier_cos']])
        results = model.fit()

        predictions = results.forecast(steps=len(test), 
                                       exog=test[['lag_1', 'lag_7', 'fourier_sin', 'fourier_cos']])

        plt.figure(figsize=self.calculate_figure_size(df))
        plt.plot(df.index, df[value_column], label='Actual')
        plt.plot(test.index, predictions, label='Forecast', color='red')
        plt.title('ETSF Forecast vs Actual')
        plt.legend()
        forecast_path = os.path.join(self.output_folder, f"{table_name}_etsf_forecast.png")
        plt.savefig(forecast_path, dpi=100, bbox_inches='tight')
        plt.close()

        mse = mean_squared_error(test[value_column], predictions)
        rmse = np.sqrt(mse)

        results = {
            "adf_result": adf_result,
            "mse": mse,
            "rmse": rmse,
            "decomposition_path": decomposition_path,
            "forecast_path": forecast_path
        }

        self.interpret_results("ETSF Analysis", results, table_name)

    def interpret_results(self, analysis_type, results, table_name):
        if isinstance(results, dict):
            results_str = "\n".join([f"{k}: {v}" for k, v in results.items()])
        elif isinstance(results, pd.DataFrame):
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

        2. Key Findings:
        [List the most important findings, including both positive and negative aspects]

        3. Implications:
        [Discuss the potential implications of these findings for the dataset or business context]

        4. Recommendations:
        [Provide actionable recommendations based on the analysis]

        If there are no significant findings, state "No significant findings" in the appropriate sections and briefly explain why.

        Interpretation:
        """
        interpretation = self.worker_erag_api.chat([{"role": "system", "content": "You are a data analyst providing insights on innovative data analysis results. Respond in the requested format."}, 
                                                    {"role": "user", "content": prompt}])
        
        check_prompt = f"""
        Original data and analysis type:
        {prompt}

        Previous interpretation:
        {interpretation}

        Please review and improve the above interpretation. Ensure it accurately reflects the original data and analysis type. Enhance the text by:
        1. Verifying the accuracy of the interpretation against the original data.
        2. Ensuring the structure (Analysis, Key Findings, Implications, Recommendations) is maintained.
        3. Making the interpretation more narrative and detailed by adding context and explanations.
        4. Addressing any important aspects of the data that weren't covered.

        Provide your response in the same format, maintaining the original structure. 
        Do not add comments, questions, or explanations about the changes - simply provide the improved version.
        """

        enhanced_interpretation = self.supervisor_erag_api.chat([
            {"role": "system", "content": "You are a data analyst improving interpretations of innovative data analysis results. Provide direct enhancements without adding meta-comments or detailing the changes done."},
            {"role": "user", "content": check_prompt}
        ])

        print(success(f"AI Interpretation for {analysis_type}:"))
        print(enhanced_interpretation.strip())
        
        self.text_output += f"\n{enhanced_interpretation.strip()}\n\n"
        
        image_data = []
        if isinstance(results, dict):
            if 'image_path' in results:
                image_data.append((f"{analysis_type} - Image", results['image_path']))
            if 'decomposition_path' in results and 'forecast_path' in results:
                image_data.append((f"{analysis_type} - Decomposition", results['decomposition_path']))
                image_data.append((f"{analysis_type} - Forecast", results['forecast_path']))
        
        self.pdf_content.append((analysis_type, image_data, enhanced_interpretation.strip()))
        
        lines = enhanced_interpretation.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith("2. Key Findings:"):
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

        summary_prompt = f"""
        Based on the following findings from the Innovative Data Analysis:
        
        {self.findings}
        
        Please provide an executive summary of the analysis. The summary should:
        1. Briefly introduce the purpose of the analysis and the innovative techniques used (AMPR and ETSF).
        2. Highlight the most significant insights and patterns discovered.
        3. Mention any potential issues or areas that require further investigation.
        4. Conclude with recommendations for next steps or areas to focus on.

        Structure the summary in multiple paragraphs for readability.
        Please provide your response in plain text format, without any special formatting or markup.
        """
        
        try:
            interpretation = self.worker_erag_api.chat([
                {"role": "system", "content": "You are a data analyst providing an executive summary of an innovative data analysis. Respond in plain text format."},
                {"role": "user", "content": summary_prompt}
            ])
            
            if interpretation is not None:
                check_prompt = f"""
                Please review and improve the following executive summary:

                {interpretation}

                Enhance the summary by:
                1. Making it more comprehensive and narrative by adding context and explanations.
                2. Addressing any important aspects of the analysis that weren't covered.
                3. Ensuring it includes a clear introduction of the innovative techniques (AMPR and ETSF), highlights of significant insights, mention of potential issues, and recommendations for next steps.

                Provide your response in plain text format, without any special formatting or markup.
                Do not add comments, questions, or explanations about the changes - simply provide the improved version.
                """

                enhanced_summary = self.supervisor_erag_api.chat([
                    {"role": "system", "content": "You are a data analyst improving an executive summary of an innovative data analysis. Provide direct enhancements without adding meta-comments."},
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
