import os
import time
import sqlite3
import threading
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm, zscore, chi2
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

import networkx as nx
from Bio import pairwise2

from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

class TimeoutException(Exception):
    pass

class AdvancedExploratoryDataAnalysisB4:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 15
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
        self.output_folder = os.path.join(settings.output_folder, f"axda_b4_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        analysis_methods = [
            self.matrix_profile,
            self.ensemble_anomaly_detection,
            self.gaussian_mixture_models,
            self.expectation_maximization,
            self.statistical_process_control,
            self.z_score_analysis,
            self.mahalanobis_distance,
            self.box_cox_transformation,
            self.grubbs_test,
            self.chauvenet_criterion,
            self.benfords_law_analysis,
            self.forensic_accounting,
            self.network_analysis_fraud_detection,
            self.sequence_alignment,
            self.conformal_anomaly_detection
        ]

        for method in analysis_methods:
            try:
                self.technique_counter += 1
                method(df, table_name)
            except Exception as e:
                error_message = f"An error occurred during {method.__name__}: {str(e)}"
                print(error(error_message))
                self.text_output += f"\n{error_message}\n"
                self.pdf_content.append((method.__name__, [], error_message))

    # Implement the new techniques here
    def matrix_profile(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Matrix Profile"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_matrix_profile():
                from stumpy import stump
                
                # Select the first numerical column for demonstration
                data = df[numerical_columns[0]].dropna().values
                window_size = min(len(data) // 4, 100)  # Adjust window size as needed
                
                mp = stump(data, m=window_size)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.calculate_figure_size())
                ax1.plot(data)
                ax1.set_title(f"Original Data: {numerical_columns[0]}")
                ax2.plot(mp[:, 0])
                ax2.set_title("Matrix Profile")
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_matrix_profile)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_matrix_profile.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Matrix Profile", img_path, table_name)
            else:
                print("Skipping Matrix Profile plot due to error in plot generation.")
        else:
            print("No numerical columns found for Matrix Profile analysis.")

    def ensemble_anomaly_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Ensemble Anomaly Detection"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_ensemble_anomaly_detection():
                X = df[numerical_columns].dropna()
                
                # Ensemble of anomaly detection methods
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                elliptic_env = EllipticEnvelope(contamination=0.1, random_state=42)
                
                # Fit and predict
                iso_forest_pred = iso_forest.fit_predict(X)
                elliptic_env_pred = elliptic_env.fit_predict(X)
                
                # Combine predictions (1 for inlier, -1 for outlier)
                ensemble_pred = np.mean([iso_forest_pred, elliptic_env_pred], axis=0)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=ensemble_pred, cmap='viridis')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title("Ensemble Anomaly Detection")
                plt.colorbar(scatter, label='Anomaly Score')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_ensemble_anomaly_detection)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_ensemble_anomaly_detection.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Ensemble Anomaly Detection", img_path, table_name)
            else:
                print("Skipping Ensemble Anomaly Detection plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Ensemble Anomaly Detection analysis.")

    def gaussian_mixture_models(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Gaussian Mixture Models (GMM)"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_gmm():
                X = df[numerical_columns].dropna()
                
                # Fit GMM
                gmm = GaussianMixture(n_components=3, random_state=42)
                gmm.fit(X)
                
                # Predict
                labels = gmm.predict(X)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title("Gaussian Mixture Models (GMM)")
                plt.colorbar(scatter, label='Cluster')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_gmm)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_gmm.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Gaussian Mixture Models (GMM)", img_path, table_name)
            else:
                print("Skipping Gaussian Mixture Models plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Gaussian Mixture Models analysis.")

    def expectation_maximization(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Expectation-Maximization Algorithm"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_em():
                X = df[numerical_columns].dropna()
                
                # EM is essentially the same as GMM in sklearn
                em = GaussianMixture(n_components=3, random_state=42)
                em.fit(X)
                
                # Predict
                labels = em.predict(X)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title("Expectation-Maximization Algorithm")
                plt.colorbar(scatter, label='Cluster')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_em)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_em.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Expectation-Maximization Algorithm", img_path, table_name)
            else:
                print("Skipping Expectation-Maximization plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Expectation-Maximization analysis.")

    def statistical_process_control(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Statistical Process Control (SPC) Charts"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_spc():
                data = df[numerical_columns[0]].dropna()
                
                mean = data.mean()
                std = data.std()
                ucl = mean + 3 * std
                lcl = mean - 3 * std
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(data, marker='o', linestyle='-', linewidth=0.5, markersize=5)
                ax.axhline(mean, color='g', linestyle='--')
                ax.axhline(ucl, color='r', linestyle='--')
                ax.axhline(lcl, color='r', linestyle='--')
                ax.set_title(f"Control Chart for {numerical_columns[0]}")
                ax.set_ylabel("Value")
                ax.set_xlabel("Sample")
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_spc)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_spc.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Statistical Process Control (SPC) Charts", img_path, table_name)
            else:
                print("Skipping Statistical Process Control plot due to error in plot generation.")
        else:
            print("No numerical columns found for Statistical Process Control analysis.")

    def z_score_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Z-Score and Modified Z-Score"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_z_score():
                data = df[numerical_columns[0]].dropna()
                
                z_scores = zscore(data)
                modified_z_scores = 0.6745 * (data - np.median(data)) / np.median(np.abs(data - np.median(data)))
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.calculate_figure_size())
                ax1.scatter(range(len(data)), z_scores)
                ax1.axhline(y=3, color='r', linestyle='--')
                ax1.axhline(y=-3, color='r', linestyle='--')
                ax1.set_title(f"Z-Scores for {numerical_columns[0]}")
                ax1.set_ylabel("Z-Score")
                
                ax2.scatter(range(len(data)), modified_z_scores)
                ax2.axhline(y=3.5, color='r', linestyle='--')
                ax2.axhline(y=-3.5, color='r', linestyle='--')
                ax2.set_title(f"Modified Z-Scores for {numerical_columns[0]}")
                ax2.set_ylabel("Modified Z-Score")
                ax2.set_xlabel("Sample")
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_z_score)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_z_score.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Z-Score and Modified Z-Score", img_path, table_name)
            else:
                print("Skipping Z-Score plot due to error in plot generation.")
        else:
            print("No numerical columns found for Z-Score analysis.")

    def mahalanobis_distance(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Mahalanobis Distance"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_mahalanobis():
                X = df[numerical_columns].dropna()
                
                # Calculate mean and covariance
                mean = np.mean(X, axis=0)
                cov = np.cov(X, rowvar=False)
                
                # Calculate Mahalanobis distance
                inv_cov = np.linalg.inv(cov)
                diff = X - mean
                left = np.dot(diff, inv_cov)
                mahalanobis = np.sqrt(np.sum(left * diff, axis=1))
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=mahalanobis, cmap='viridis')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title("Mahalanobis Distance")
                plt.colorbar(scatter, label='Mahalanobis Distance')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_mahalanobis)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_mahalanobis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Mahalanobis Distance", img_path, table_name)
            else:
                print("Skipping Mahalanobis Distance plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Mahalanobis Distance analysis.")

    def box_cox_transformation(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Box-Cox Transformation"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_box_cox():
                from scipy.stats import boxcox
                
                data = df[numerical_columns[0]].dropna()
                
                # Ensure all values are positive
                if np.min(data) <= 0:
                    data = data - np.min(data) + 1
                
                transformed_data, lambda_param = boxcox(data)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.calculate_figure_size())
                ax1.hist(data, bins=30)
                ax1.set_title(f"Original Distribution of {numerical_columns[0]}")
                ax1.set_ylabel("Frequency")
                
                ax2.hist(transformed_data, bins=30)
                ax2.set_title(f"Box-Cox Transformed Distribution (λ = {lambda_param:.2f})")
                ax2.set_ylabel("Frequency")
                ax2.set_xlabel("Value")
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_box_cox)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_box_cox.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Box-Cox Transformation", img_path, table_name)
            else:
                print("Skipping Box-Cox Transformation plot due to error in plot generation.")
        else:
            print("No numerical columns found for Box-Cox Transformation analysis.")

    def grubbs_test(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Grubbs' Test"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_grubbs():
                from scipy import stats
                
                data = df[numerical_columns[0]].dropna()
                
                def grubbs_test(data, alpha=0.05):
                    n = len(data)
                    mean = np.mean(data)
                    std = np.std(data)
                    z = (data - mean) / std
                    G = np.max(np.abs(z))
                    t_value = stats.t.ppf(1 - alpha / (2 * n), n - 2)
                    G_critical = ((n - 1) * np.sqrt(t_value**2 / (n - 2 + t_value**2))) / np.sqrt(n)
                    return G > G_critical, G, G_critical
                
                is_outlier, G, G_critical = grubbs_test(data)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.boxplot(data)
                ax.set_title(f"Grubbs' Test for {numerical_columns[0]}")
                ax.set_ylabel("Value")
                plt.figtext(0.5, 0.01, f"Grubbs' statistic: {G:.2f}, Critical value: {G_critical:.2f}", ha="center")
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_grubbs)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_grubbs.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Grubbs' Test", img_path, table_name)
            else:
                print("Skipping Grubbs' Test plot due to error in plot generation.")
        else:
            print("No numerical columns found for Grubbs' Test analysis.")

    def chauvenet_criterion(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Chauvenet's Criterion"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_chauvenet():
                data = df[numerical_columns[0]].dropna()
                
                def chauvenet(data):
                    mean = np.mean(data)
                    std = np.std(data)
                    N = len(data)
                    criterion = 1.0 / (2 * N)
                    d = abs(data - mean) / std
                    prob = 2 * (1 - norm.cdf(d))
                    return prob < criterion
                
                is_outlier = chauvenet(data)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.scatter(range(len(data)), data, c=['r' if x else 'b' for x in is_outlier])
                ax.set_title(f"Chauvenet's Criterion for {numerical_columns[0]}")
                ax.set_ylabel("Value")
                ax.set_xlabel("Sample")
                plt.figtext(0.5, 0.01, f"Red points are potential outliers according to Chauvenet's Criterion", ha="center")
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_chauvenet)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_chauvenet.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Chauvenet's Criterion", img_path, table_name)
            else:
                print("Skipping Chauvenet's Criterion plot due to error in plot generation.")
        else:
            print("No numerical columns found for Chauvenet's Criterion analysis.")

    def benfords_law_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Benford's Law Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_benfords_law():
                data = df[numerical_columns[0]].dropna()
                
                def get_first_digit(n):
                    return int(str(abs(n)).strip('0.')[0])
                
                first_digits = data.apply(get_first_digit)
                observed_freq = first_digits.value_counts().sort_index() / len(first_digits)
                
                expected_freq = pd.Series([np.log10(1 + 1/d) for d in range(1, 10)], index=range(1, 10))
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                observed_freq.plot(kind='bar', ax=ax, alpha=0.5, label='Observed')
                expected_freq.plot(kind='line', ax=ax, color='r', label='Expected (Benford\'s Law)')
                ax.set_title(f"Benford's Law Analysis for {numerical_columns[0]}")
                ax.set_xlabel("First Digit")
                ax.set_ylabel("Frequency")
                ax.legend()
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_benfords_law)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_benfords_law.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Benford's Law Analysis", img_path, table_name)
            else:
                print("Skipping Benford's Law Analysis plot due to error in plot generation.")
        else:
            print("No numerical columns found for Benford's Law Analysis.")

    def forensic_accounting(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Forensic Accounting Techniques"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_forensic_accounting():
                # For this example, we'll use a scatter plot of two numerical columns
                # and highlight potential anomalies using Mahalanobis distance
                X = df[numerical_columns].dropna()
                
                # Calculate Mahalanobis distance
                mean = np.mean(X, axis=0)
                cov = np.cov(X, rowvar=False)
                inv_cov = np.linalg.inv(cov)
                diff = X - mean
                left = np.dot(diff, inv_cov)
                mahalanobis = np.sqrt(np.sum(left * diff, axis=1))
                
                # Define threshold for anomalies (e.g., top 5% of Mahalanobis distances)
                threshold = np.percentile(mahalanobis, 95)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=mahalanobis, cmap='viridis')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title("Forensic Accounting: Potential Anomalies")
                plt.colorbar(scatter, label='Mahalanobis Distance')
                
                # Highlight potential anomalies
                anomalies = X[mahalanobis > threshold]
                ax.scatter(anomalies.iloc[:, 0], anomalies.iloc[:, 1], color='red', s=100, facecolors='none', edgecolors='r', label='Potential Anomalies')
                
                ax.legend()
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_forensic_accounting)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_forensic_accounting.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Forensic Accounting Techniques", img_path, table_name)
            else:
                print("Skipping Forensic Accounting plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Forensic Accounting analysis.")

    def network_analysis_fraud_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Network Analysis for Fraud Detection"))
        
        # For this example, we'll assume we have columns for 'source', 'target', and 'amount'
        if all(col in df.columns for col in ['source', 'target', 'amount']):
            def plot_network_analysis():
                G = nx.from_pandas_edgelist(df, 'source', 'target', ['amount'])
                
                # Calculate degree centrality
                degree_centrality = nx.degree_centrality(G)
                
                # Identify potential fraudulent nodes (e.g., nodes with high degree centrality)
                threshold = np.percentile(list(degree_centrality.values()), 95)
                suspicious_nodes = [node for node, centrality in degree_centrality.items() if centrality > threshold]
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=suspicious_nodes, node_color='red', node_size=700)
                
                ax.set_title("Network Analysis for Fraud Detection")
                plt.figtext(0.5, 0.01, "Red nodes are potentially suspicious based on high degree centrality", ha="center")
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_network_analysis)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_network_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Network Analysis for Fraud Detection", img_path, table_name)
            else:
                print("Skipping Network Analysis plot due to error in plot generation.")
        else:
            print("Required columns (source, target, amount) not found for Network Analysis.")

    def sequence_alignment(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Sequence Alignment and Matching"))
        
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            def plot_sequence_alignment():
                from Bio import pairwise2
                from Bio.SubsMat import MatrixInfo as matlist
                
                # Select the first text column
                sequences = df[text_columns[0]].dropna().head(10).tolist()  # Limit to 10 sequences for simplicity
                
                # Perform pairwise alignments
                alignments = []
                for i in range(len(sequences)):
                    for j in range(i+1, len(sequences)):
                        alignment = pairwise2.align.globalds(sequences[i], sequences[j], matlist.blosum62, -10, -0.5)
                        alignments.append((i, j, alignment[0][2]))  # Store indices and alignment score
                
                # Create a similarity matrix
                similarity_matrix = np.zeros((len(sequences), len(sequences)))
                for i, j, score in alignments:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = score
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                im = ax.imshow(similarity_matrix, cmap='viridis')
                ax.set_title("Sequence Alignment Similarity Matrix")
                ax.set_xlabel("Sequence Index")
                ax.set_ylabel("Sequence Index")
                plt.colorbar(im, label='Alignment Score')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_sequence_alignment)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_sequence_alignment.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Sequence Alignment and Matching", img_path, table_name)
            else:
                print("Skipping Sequence Alignment plot due to error in plot generation.")
        else:
            print("No text columns found for Sequence Alignment analysis.")

    def conformal_anomaly_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Conformal Anomaly Detection"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_conformal_anomaly_detection():
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestRegressor
                from nonconformist.cp import IcpRegressor
                from nonconformist.nc import AbsErrorErrFunc
                
                X = df[numerical_columns].dropna()
                y = X.iloc[:, 0]  # Use the first column as the target
                X = X.iloc[:, 1:]  # Use the remaining columns as features
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Create and fit underlying model
                underlying_model = RandomForestRegressor(n_estimators=100, random_state=42)
                underlying_model.fit(X_train, y_train)
                
                # Create ICP
                icp = IcpRegressor(underlying_model, AbsErrorErrFunc())
                
                # Fit ICP
                icp.fit(X_train, y_train)
                
                # Perform conformal prediction
                predictions = icp.predict(X_test, significance=0.1)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(y_test, predictions[:, 0], alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("True Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Conformal Anomaly Detection")
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_conformal_anomaly_detection)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_conformal_anomaly_detection.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Conformal Anomaly Detection", img_path, table_name)
            else:
                print("Skipping Conformal Anomaly Detection plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Conformal Anomaly Detection analysis.")


    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 4) Report for {self.table_name}"
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.executive_summary,
            self.findings,
            self.pdf_content,
            self.image_data,
            filename=f"axda_b4_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
        else:
            print(error("Failed to generate PDF report"))
        return pdf_file

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

        2. Positive Findings:
        [List any positive findings, or state "No significant positive findings" if none]

        3. Negative Findings:
        [List any negative findings, or state "No significant negative findings" if none]

        4. Conclusion:
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
        2. Ensuring the structure (Analysis, Positive Findings, Negative Findings, Conclusion) is maintained.
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
        if isinstance(results, str) and results.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_data.append((f"{analysis_type}", results))
        
        self.pdf_content.append((analysis_type, image_data, enhanced_interpretation.strip()))
        
        # Extract important findings
        lines = enhanced_interpretation.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith("2. Positive Findings:") or line.startswith("3. Negative Findings:"):
                for finding in lines[i+1:]:
                    if finding.strip() and not finding.startswith(("2.", "3.", "4.")):
                        self.findings.append(f"{analysis_type}: {finding.strip()}")
                    elif finding.startswith(("2.", "3.", "4.")):
                        break

        # Update self.image_data
        self.image_data.extend(image_data)

    def generate_executive_summary(self):
        if not self.findings:
            self.executive_summary = "No significant findings were identified during the analysis. This could be due to a lack of data, uniform data distribution, or absence of notable patterns or anomalies in the dataset."
            return

        # Count the number of successful techniques
        successful_techniques = sum(1 for item in self.pdf_content if len(item[1]) > 0 or not item[2].startswith("An error occurred"))
        failed_techniques = self.total_techniques - successful_techniques

        summary_prompt = f"""
        Based on the following findings from the Advanced Exploratory Data Analysis (Batch 4):
        
        {self.findings}
        
        Additional context:
        - {successful_techniques} out of {self.total_techniques} analysis techniques were successfully completed.
        - {failed_techniques} techniques encountered errors and were skipped.
        
        Please provide an executive summary of the analysis. The summary should:
        1. Briefly introduce the purpose of the advanced analysis.
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
                4. Discussing the implications of any failed techniques on the overall analysis.

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
        output_file = os.path.join(self.output_folder, "axda_b4_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)
