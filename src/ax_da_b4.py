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

    def run(self):
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch4) on {self.db_path}"))
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch4) completed. Results saved in {self.output_folder}"))



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

    def matrix_profile(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Matrix Profile"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_matrix_profile():
                from stumpy import stump
                
                # Select the first numerical column for demonstration
                data = df[numerical_columns[0]].dropna().astype(float).values  # Convert to float
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
                image_paths.append(img_path)
                
            else:
                print("Skipping Matrix Profile plot due to error in plot generation.")
        else:
            print("No numerical columns found for Matrix Profile analysis.")
        self.interpret_results("Matrix Profile", {'image_paths': image_paths}, table_name)

    def ensemble_anomaly_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Ensemble Anomaly Detection"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
            else:
                print("Skipping Ensemble Anomaly Detection plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Ensemble Anomaly Detection analysis.")
        self.interpret_results("Ensemble Anomaly Detection", {'image_paths': image_paths}, table_name)

    def gaussian_mixture_models(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Gaussian Mixture Models (GMM)"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
            else:
                print("Skipping Gaussian Mixture Models plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Gaussian Mixture Models analysis.")
        self.interpret_results("Gaussian Mixture Models (GMM)", {'image_paths': image_paths}, table_name)

    def expectation_maximization(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Expectation-Maximization Algorithm"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
            else:
                print("Skipping Expectation-Maximization plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Expectation-Maximization analysis.")
        self.interpret_results("Expectation-Maximization Algorithm", {'image_paths': image_paths}, table_name)

    def statistical_process_control(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Statistical Process Control (SPC) Charts"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
            else:
                print("Skipping Statistical Process Control plot due to error in plot generation.")
        else:
            print("No numerical columns found for Statistical Process Control analysis.")
        self.interpret_results("Statistical Process Control (SPC) Charts", {'image_paths': image_paths}, table_name)

    def z_score_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Z-Score and Modified Z-Score"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
            else:
                print("Skipping Z-Score plot due to error in plot generation.")
        else:
            print("No numerical columns found for Z-Score analysis.")
        self.interpret_results("Z-Score and Modified Z-Score", {'image_paths': image_paths}, table_name)

    def mahalanobis_distance(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Mahalanobis Distance"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_mahalanobis():
                X = df[numerical_columns].dropna()
                
                # Remove constant columns
                X = X.loc[:, (X != X.iloc[0]).any()]
                
                if X.shape[1] < 2:
                    print("Not enough variable columns for Mahalanobis Distance analysis.")
                    return None

                # Calculate mean and covariance
                mean = np.mean(X, axis=0)
                cov = np.cov(X, rowvar=False)
                
                try:
                    # Calculate Mahalanobis distance
                    inv_cov = np.linalg.inv(cov)
                    diff = X - mean
                    left = np.dot(diff, inv_cov)
                    mahalanobis = np.sqrt(np.sum(left * diff, axis=1))
                    
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=mahalanobis, cmap='viridis')
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.set_title("Mahalanobis Distance")
                    plt.colorbar(scatter, label='Mahalanobis Distance')
                    plt.tight_layout()
                    return fig, ax
                except np.linalg.LinAlgError:
                    print("Singular matrix encountered. Skipping Mahalanobis Distance analysis.")
                    return None

            result = self.generate_plot(plot_mahalanobis)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_mahalanobis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Mahalanobis Distance plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Mahalanobis Distance analysis.")
        self.interpret_results("Mahalanobis Distance", {'image_paths': image_paths}, table_name)

    def box_cox_transformation(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Box-Cox Transformation"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
            else:
                print("Skipping Box-Cox Transformation plot due to error in plot generation.")
        else:
            print("No numerical columns found for Box-Cox Transformation analysis.")
        self.interpret_results("Box-Cox Transformation", {'image_paths': image_paths}, table_name)

    def grubbs_test(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Grubbs' Test"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
            else:
                print("Skipping Grubbs' Test plot due to error in plot generation.")
        else:
            print("No numerical columns found for Grubbs' Test analysis.")
        self.interpret_results("Grubbs' Test", {'image_paths': image_paths}, table_name)

    def chauvenet_criterion(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Chauvenet's Criterion"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
            else:
                print("Skipping Chauvenet's Criterion plot due to error in plot generation.")
        else:
            print("No numerical columns found for Chauvenet's Criterion analysis.")
        self.interpret_results("Chauvenet's Criterion", {'image_paths': image_paths}, table_name)

    def benfords_law_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Benford's Law Analysis"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
            else:
                print("Skipping Benford's Law Analysis plot due to error in plot generation.")
        else:
            print("No numerical columns found for Benford's Law Analysis.")
        self.interpret_results("Benford's Law Analysis", {'image_paths': image_paths}, table_name)

    def forensic_accounting(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Forensic Accounting Techniques"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_forensic_accounting():
                # For this example, we'll use a scatter plot of two numerical columns
                # and highlight potential anomalies using Mahalanobis distance
                X = df[numerical_columns].dropna()
                
                # Remove constant columns
                X = X.loc[:, (X != X.iloc[0]).any()]
                
                if X.shape[1] < 2:
                    print("Not enough variable columns for Forensic Accounting analysis.")
                    return None

                try:
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
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.set_title("Forensic Accounting: Potential Anomalies")
                    plt.colorbar(scatter, label='Mahalanobis Distance')
                    
                    # Highlight potential anomalies
                    anomalies = X[mahalanobis > threshold]
                    ax.scatter(anomalies.iloc[:, 0], anomalies.iloc[:, 1], color='red', s=100, facecolors='none', edgecolors='r', label='Potential Anomalies')
                    
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax
                except np.linalg.LinAlgError:
                    print("Singular matrix encountered. Performing alternative analysis.")
                    
                    # Alternative analysis: Simple outlier detection using Z-score
                    z_scores = np.abs(stats.zscore(X))
                    outliers = (z_scores > 3).any(axis=1)
                    
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c='blue', label='Normal')
                    ax.scatter(X[outliers].iloc[:, 0], X[outliers].iloc[:, 1], c='red', label='Potential Anomalies')
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.set_title("Forensic Accounting: Potential Anomalies (Z-score method)")
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

            result = self.generate_plot(plot_forensic_accounting)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_forensic_accounting.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Forensic Accounting plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Forensic Accounting analysis.")
        self.interpret_results("Forensic Accounting Techniques", {'image_paths': image_paths}, table_name)

    def network_analysis_fraud_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Network Analysis for Fraud Detection"))
        image_paths = []
        
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
                image_paths.append(img_path)
                self.interpret_results("Network Analysis for Fraud Detection", {'image_paths': image_paths}, table_name)
            else:
                print("Required columns (source, target, amount) not found. Performing alternative analysis.")
                numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
                if len(numerical_columns) >= 2:
                    def plot_alternative_analysis():
                        X = df[numerical_columns].dropna()
                        
                        # Perform a simple correlation analysis
                        corr = X.corr()
                        
                        fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                        ax.set_title("Correlation Heatmap (Alternative to Network Analysis)")
                        plt.tight_layout()
                        return fig, ax

                    result = self.generate_plot(plot_alternative_analysis)
                    if result is not None:
                        fig, _ = result
                        img_path = os.path.join(self.output_folder, f"{table_name}_alternative_network_analysis.png")
                        plt.savefig(img_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        image_paths.append(img_path)
                        
                    else:
                        print("Skipping alternative analysis plot due to error in plot generation.")
                else:
                    print("Not enough numerical columns for alternative analysis.")
                self.interpret_results("Alternative Network Analysis", {'image_paths': image_paths}, table_name)

    def sequence_alignment(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Sequence Alignment and Matching"))
        image_paths = []
        
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            def plot_sequence_alignment():
                try:
                    from Bio import pairwise2
                    from Bio.Align import substitution_matrices
                    
                    # Select the first text column
                    sequences = df[text_columns[0]].dropna().head(10).tolist()  # Limit to 10 sequences for simplicity
                    
                    # Perform pairwise alignments
                    alignments = []
                    for i in range(len(sequences)):
                        for j in range(i+1, len(sequences)):
                            alignment = pairwise2.align.globalxx(sequences[i], sequences[j])
                            alignments.append((i, j, alignment[0].score))  # Store indices and alignment score
                    
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
                
                except ImportError:
                    print("Bio module not found. Performing alternative text analysis.")
                    
                    # Select the first text column
                    text_data = df[text_columns[0]].dropna().head(10).tolist()  # Limit to 10 texts for simplicity
                    
                    # Calculate simple text similarity based on character count difference
                    similarity_matrix = np.zeros((len(text_data), len(text_data)))
                    for i in range(len(text_data)):
                        for j in range(i+1, len(text_data)):
                            similarity = 1 - abs(len(text_data[i]) - len(text_data[j])) / max(len(text_data[i]), len(text_data[j]))
                            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
                    
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    im = ax.imshow(similarity_matrix, cmap='viridis')
                    ax.set_title("Text Similarity Matrix (Based on Length)")
                    ax.set_xlabel("Text Index")
                    ax.set_ylabel("Text Index")
                    plt.colorbar(im, label='Similarity Score')
                    plt.tight_layout()
                    return fig, ax

            result = self.generate_plot(plot_sequence_alignment)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_sequence_alignment.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Sequence Alignment plot due to error in plot generation.")
        else:
            print("No text columns found for Sequence Alignment analysis.")
        self.interpret_results("Sequence Alignment and Matching", {'image_paths': image_paths}, table_name)

    def conformal_anomaly_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Conformal Anomaly Detection"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_conformal_anomaly_detection():
                try:
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
                except Exception as e:
                    print(f"Error in Conformal Anomaly Detection: {str(e)}")
                    return None

            result = self.generate_plot(plot_conformal_anomaly_detection)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_conformal_anomaly_detection.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Conformal Anomaly Detection plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Conformal Anomaly Detection analysis.")
        self.interpret_results("Conformal Anomaly Detection", {'image_paths': image_paths}, table_name)




    def interpret_results(self, analysis_type, results, table_name):
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

        prompt = f"""
        You are an expert data analyst providing insights on exploratory data analysis results. Your task is to interpret the following analysis results and provide a detailed, data-driven interpretation.

        Analysis type: {analysis_type}
        Table name: {table_name}
        Results:
        {results_str}

        Please provide a thorough interpretation of these results, highlighting noteworthy patterns, anomalies, or insights. Focus on the most important aspects that would be valuable for data analysis. Always provide specific numbers and percentages when discussing findings.

        If some data appears to be missing or incomplete, work with the available information without mentioning the limitations. Your goal is to extract as much insight as possible from the given data.

        Structure your response in the following format:

        1. Analysis:
        [Provide a detailed description of the analysis performed, including specific metrics and their values]

        2. Key Findings:
        [List the most important discoveries, always including relevant numbers and percentages]

        3. Implications:
        [Discuss the potential impact of these findings on business decisions or further analyses]

        4. Recommendations:
        [Suggest next steps or areas for deeper investigation based on these results]

        Ensure your interpretation is concise yet comprehensive, focusing on actionable insights derived from the data.

        Interpretation:
        """
        interpretation = self.worker_erag_api.chat([{"role": "system", "content": "You are an expert data analyst providing insights on exploratory data analysis results. Respond in the requested format."}, 
                                                    {"role": "user", "content": prompt}])
        
        # Updated supervisor prompt
        check_prompt = f"""
        As a senior data analyst, review and enhance the following interpretation of exploratory data analysis results. The original data and analysis type are:

        {prompt}

        Previous interpretation:
        {interpretation}

        Improve this interpretation by:
        1. Ensuring all statements are backed by specific data points, numbers, or percentages from the original results.
        2. Removing any vague statements and replacing them with precise, data-driven observations.
        3. Adding any critical insights that may have been overlooked, always referencing specific data.
        4. Strengthening the implications and recommendations sections with concrete, actionable suggestions based on the data.

        Provide your enhanced interpretation in the same format (Analysis, Key Findings, Implications, Recommendations). Do not list your changes or repeat the original interpretation. Simply provide the improved version, focusing on clarity, specificity, and actionable insights.

        Enhanced Interpretation:
        """

        enhanced_interpretation = self.supervisor_erag_api.chat([
            {"role": "system", "content": "You are a senior data analyst improving interpretations of exploratory data analysis results. Provide direct enhancements without meta-comments."},
            {"role": "user", "content": check_prompt}
        ])

        print(success(f"AI Interpretation for {analysis_type}:"))
        print(enhanced_interpretation.strip())
        
        self.text_output += f"\n{enhanced_interpretation.strip()}\n\n"
        
         # Handle images
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
            if line.startswith("2. Key Findings:"):
                for finding in lines[i+1:]:
                    if finding.strip() and not finding.startswith(("3.", "4.")):
                        self.findings.append(f"{analysis_type}: {finding.strip()}")
                    elif finding.startswith(("3.", "4.")):
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
        Based on the following findings from the Exploratory Data Analysis:
        
        {self.findings}
        
        Additional context:
        - {successful_techniques} out of {self.total_techniques} analysis techniques were successfully completed.
        - {failed_techniques} techniques encountered errors and were skipped.
        
        Please provide an executive summary of the analysis. The summary should:
        1. Briefly introduce the purpose of the analysis.
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
                4. Discussing the implications of any failed techniques on the overall analysis.

                Provide your response in plain text format, without any special formatting or markup.
                Do not add comments, questions, or explanations about the changes - simply provide the improved version.
                """

                enhanced_summary = self.supervisor_erag_api.chat([
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
        output_file = os.path.join(self.output_folder, "axda_b4_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 4) Report for {self.table_name}"
        
        # Ensure all image data is in the correct format
        formatted_image_data = []
        for item in self.pdf_content:
            analysis_type, images, interpretation = item
            if isinstance(images, list):
                for image in images:
                    if isinstance(image, tuple) and len(image) == 2:
                        formatted_image_data.append(image)
                    elif isinstance(image, str):
                        # If it's just a string (path), use the analysis type as the title
                        formatted_image_data.append((analysis_type, image))
            elif isinstance(images, str):
                # If it's just a string (path), use the analysis type as the title
                formatted_image_data.append((analysis_type, images))
        
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.executive_summary,
            self.findings,
            self.pdf_content,
            formatted_image_data,  # Use the formatted image data
            filename=f"axda_b4_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None
