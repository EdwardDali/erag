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
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.tools import diff
from scipy.stats import t, chi2, norm, jarque_bera
from statsmodels.stats.diagnostic import lilliefors
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

class TimeoutException(Exception):
    pass


class AdvancedExploratoryDataAnalysisB5:
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
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch5) on {self.db_path}"))
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch5) completed. Results saved in {self.output_folder}"))



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
            self.hampel_filter_analysis,
            self.gesd_test_analysis,
            self.dixons_q_test_analysis,
            self.peirce_criterion_analysis,
            self.thompson_tau_test_analysis,
            self.control_charts_analysis,
            self.kde_anomaly_detection_analysis,
            self.hotellings_t_squared_analysis,
            self.breakdown_point_analysis,
            self.chi_square_test_analysis,
            self.simple_thresholding_analysis,
            self.lilliefors_test_analysis,
            self.jarque_bera_test_analysis
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
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            print(warning("Not enough numeric columns for Cook's Distance analysis."))
            return
        
        X = df[numeric_cols].drop(numeric_cols[-1], axis=1)
        y = df[numeric_cols[-1]]
        
        model = LinearRegression().fit(X, y)
        
        influence = OLSInfluence(model)
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
            image_paths.append(img_path)
            
            threshold = 4 / len(df)
            influential_points = np.where(cooks_d > threshold)[0]
            
            self.interpret_results("Cook's Distance Analysis", {
                'image_paths': image_paths,
                'influential_points': influential_points.tolist(),
                'threshold': threshold
            }, table_name)
        else:
            print("Skipping Cook's Distance plot due to timeout.")

    def stl_decomposition_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - STL Decomposition Analysis"))
        image_paths = []
        
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
            image_paths.append(img_path)
            
            self.interpret_results("STL Decomposition Analysis", {
                'image_paths': image_paths,
                'trend_strength': 1 - np.var(result.resid) / np.var(result.trend + result.resid),
                'seasonal_strength': 1 - np.var(result.resid) / np.var(result.seasonal + result.resid)
            }, table_name)
        else:
            print("Skipping STL Decomposition plot due to timeout.")

    def hampel_filter_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hampel Filter Analysis"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
                results[col] = {
                    'outliers_count': outliers.sum(),
                    'outliers_percentage': (outliers.sum() / len(series)) * 100
                }
            else:
                print(f"Skipping Hampel Filter plot for {col} due to timeout.")
        
        self.interpret_results("Hampel Filter Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)


    def gesd_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - GESD Test Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 3:  # GESD test requires at least 3 data points
                continue
            
            def gesd_test(data, alpha=0.05, max_outliers=10):
                n = len(data)
                if n <= 2:
                    return []
                
                outliers = []
                for i in range(max_outliers):
                    if n <= 2:
                        break
                    mean = np.mean(data)
                    std = np.std(data, ddof=1)
                    R = np.max(np.abs(data - mean)) / std
                    idx = np.argmax(np.abs(data - mean))
                    
                    t_ppf = t.ppf(1 - alpha / (2 * n), n - 2)
                    lambda_crit = ((n - 1) * t_ppf) / np.sqrt((n - 2 + t_ppf**2) * n)
                    
                    if R > lambda_crit:
                        outliers.append((idx, data[idx]))
                        data = np.delete(data, idx)
                        n -= 1
                    else:
                        break
                
                return outliers
            
            outliers = gesd_test(data.values)
            results[col] = outliers

            def plot_gesd():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(df.index, df[col], label='Original Data')
                outlier_indices = [idx for idx, _ in outliers]
                outlier_values = [val for _, val in outliers]
                ax.scatter(df.index[outlier_indices], outlier_values, color='red', label='Outliers')
                ax.set_title(f'GESD Test Outliers - {col}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_gesd)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_gesd_test.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping GESD Test plot for {col} due to timeout.")
        
        self.interpret_results("GESD Test Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

    def dixons_q_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Dixon's Q Test Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = df[col].dropna().sort_values()
            if len(data) < 3 or len(data) > 30:  # Dixon's Q test is typically used for sample sizes between 3 and 30
                continue
            
            def dixon_q_test(data, alpha=0.05):
                n = len(data)
                if n < 3 or n > 30:
                    return None, None
                
                q_crit_table = {
                    3: 0.970, 4: 0.829, 5: 0.710, 6: 0.628, 7: 0.569, 8: 0.608, 9: 0.564, 10: 0.530,
                    11: 0.502, 12: 0.479, 13: 0.611, 14: 0.586, 15: 0.565, 16: 0.546, 17: 0.529,
                    18: 0.514, 19: 0.501, 20: 0.489, 21: 0.478, 22: 0.468, 23: 0.459, 24: 0.451,
                    25: 0.443, 26: 0.436, 27: 0.429, 28: 0.423, 29: 0.417, 30: 0.412
                }
                
                q_crit = q_crit_table[n]
                
                if n <= 7:
                    q_low = (data[1] - data[0]) / (data[-1] - data[0])
                    q_high = (data[-1] - data[-2]) / (data[-1] - data[0])
                else:
                    q_low = (data[1] - data[0]) / (data[-2] - data[0])
                    q_high = (data[-1] - data[-2]) / (data[-1] - data[1])
                
                outlier_low = q_low > q_crit
                outlier_high = q_high > q_crit
                
                return (data[0] if outlier_low else None, data[-1] if outlier_high else None)
            
            low_outlier, high_outlier = dixon_q_test(data.values)
            results[col] = {'low_outlier': low_outlier, 'high_outlier': high_outlier}

            def plot_dixon_q():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(range(len(data)), data, 'bo-')
                if low_outlier is not None:
                    ax.plot(0, low_outlier, 'ro', markersize=10, label='Low Outlier')
                if high_outlier is not None:
                    ax.plot(len(data)-1, high_outlier, 'go', markersize=10, label='High Outlier')
                ax.set_title(f"Dixon's Q Test - {col}")
                ax.set_xlabel('Sorted Data Index')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_dixon_q)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_dixon_q_test.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Dixon's Q Test plot for {col} due to timeout.")
        
        self.interpret_results("Dixon's Q Test Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

    def peirce_criterion_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Peirce's Criterion Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        def peirce_criterion(data):
            n = len(data)
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            
            # Peirce's criterion table (approximation)
            R_table = {1: 1.0, 2: 1.28, 3: 1.38, 4: 1.44, 5: 1.48, 6: 1.51, 7: 1.53, 8: 1.55, 9: 1.57, 10: 1.58}
            
            if n <= 10:
                R = R_table[n]
            else:
                R = 1.58 + 0.2 * np.log10(n / 10)
            
            threshold = R * std
            outliers = [x for x in data if abs(x - mean) > threshold]
            
            return outliers
        
        for col in numeric_cols:
            data = df[col].dropna()
            outliers = peirce_criterion(data.values)
            results[col] = outliers

            def plot_peirce():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(df.index, df[col], label='Original Data')
                ax.scatter(df.index[df[col].isin(outliers)], outliers, color='red', label='Outliers')
                ax.set_title(f"Peirce's Criterion Outliers - {col}")
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_peirce)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_peirce_criterion.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Peirce's Criterion plot for {col} due to timeout.")
        
        self.interpret_results("Peirce's Criterion Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

    def thompson_tau_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Thompson Tau Test Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        def thompson_tau_test(data, alpha=0.05):
            n = len(data)
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            
            t_value = t.ppf(1 - alpha / 2, n - 2)
            tau = (t_value * (n - 1)) / (np.sqrt(n) * np.sqrt(n - 2 + t_value**2))
            
            delta = tau * std
            outliers = [x for x in data if abs(x - mean) > delta]
            
            return outliers
        
        for col in numeric_cols:
            data = df[col].dropna()
            outliers = thompson_tau_test(data.values)
            results[col] = outliers

            def plot_thompson_tau():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(df.index, df[col], label='Original Data')
                ax.scatter(df.index[df[col].isin(outliers)], outliers, color='red', label='Outliers')
                ax.set_title(f'Thompson Tau Test Outliers - {col}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_thompson_tau)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_thompson_tau.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Thompson Tau Test plot for {col} due to timeout.")
        
        self.interpret_results("Thompson Tau Test Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

    def control_charts_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Control Charts Analysis (CUSUM, EWMA)"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        def cusum_chart(data, threshold=1, drift=0):
            cumsum = np.zeros(len(data))
            for i in range(1, len(data)):
                cumsum[i] = max(0, cumsum[i-1] + data[i] - (np.mean(data) + drift))
            
            upper_control_limit = threshold * np.std(data)
            return cumsum, upper_control_limit
        
        def ewma_chart(data, lambda_param=0.2, L=3):
            ewma = np.zeros(len(data))
            ewma[0] = data[0]
            for i in range(1, len(data)):
                ewma[i] = lambda_param * data[i] + (1 - lambda_param) * ewma[i-1]
            
            std_ewma = np.std(data) * np.sqrt(lambda_param / (2 - lambda_param))
            upper_control_limit = np.mean(data) + L * std_ewma
            lower_control_limit = np.mean(data) - L * std_ewma
            
            return ewma, upper_control_limit, lower_control_limit
        
        for col in numeric_cols:
            data = df[col].dropna().values
            
            cusum, cusum_ucl = cusum_chart(data)
            ewma, ewma_ucl, ewma_lcl = ewma_chart(data)
            
            def plot_control_charts():
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.calculate_figure_size()[0], self.calculate_figure_size()[1]*2))
                
                # CUSUM chart
                ax1.plot(cusum, label='CUSUM')
                ax1.axhline(y=cusum_ucl, color='r', linestyle='--', label='Upper Control Limit')
                ax1.set_title(f'CUSUM Control Chart for {col}')
                ax1.set_xlabel('Observation')
                ax1.set_ylabel('Cumulative Sum')
                ax1.legend()
                
                # EWMA chart
                ax2.plot(ewma, label='EWMA')
                ax2.axhline(y=ewma_ucl, color='r', linestyle='--', label='Upper Control Limit')
                ax2.axhline(y=ewma_lcl, color='r', linestyle='--', label='Lower Control Limit')
                ax2.set_title(f'EWMA Control Chart for {col}')
                ax2.set_xlabel('Observation')
                ax2.set_ylabel('EWMA')
                ax2.legend()
                
                plt.tight_layout()
                return fig, (ax1, ax2)
            
            result = self.generate_plot(plot_control_charts)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_control_charts.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
                results[col] = {
                    'cusum_out_of_control': np.sum(cusum > cusum_ucl),
                    'ewma_out_of_control': np.sum((ewma > ewma_ucl) | (ewma < ewma_lcl))
                }
            else:
                print(f"Skipping control charts for {col} due to timeout.")
        
        self.interpret_results("Control Charts Analysis (CUSUM, EWMA)", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

    def kde_anomaly_detection_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - KDE Anomaly Detection Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = df[col].dropna().values
            
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(min(data), max(data), 1000)
            density = kde(x_range)
            
            threshold = np.percentile(density, 5)  # Use 5th percentile as anomaly threshold
            anomalies = data[kde(data) < threshold]
            
            def plot_kde_anomalies():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(x_range, density, label='KDE')
                ax.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
                ax.scatter(anomalies, kde(anomalies), color='r', label='Anomalies')
                ax.set_title(f'KDE Anomaly Detection for {col}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_kde_anomalies)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_kde_anomalies.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
                results[col] = {
                    'anomalies_count': len(anomalies),
                    'anomalies_percentage': (len(anomalies) / len(data)) * 100
                }
            else:
                print(f"Skipping KDE anomaly detection plot for {col} due to timeout.")
        
        self.interpret_results("KDE Anomaly Detection Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

    def hotellings_t_squared_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hotelling's T-squared Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            print(warning("Not enough numeric columns for Hotelling's T-squared analysis."))
            return
        
        X = df[numeric_cols].dropna()
        n, p = X.shape
        
        if n <= p:
            print(warning("Not enough samples for Hotelling's T-squared analysis."))
            return
        
        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print(warning("Singular matrix encountered. Using pseudo-inverse instead."))
            inv_cov = np.linalg.pinv(cov)
        
        def t_squared(x):
            diff = x - mean
            return np.dot(np.dot(diff, inv_cov), diff.T)
        
        t_sq = np.array([t_squared(x) for x in X.values])
        
        # Calculate critical value
        f_crit = stats.f.ppf(0.95, p, n-p)
        t_sq_crit = ((n-1)*p/(n-p)) * f_crit
        
        outliers = X[t_sq > t_sq_crit]
        
        def plot_t_squared():
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ax.plot(t_sq, label="T-squared")
            ax.axhline(y=t_sq_crit, color='r', linestyle='--', label='Critical Value')
            ax.set_title("Hotelling's T-squared Control Chart")
            ax.set_xlabel('Observation')
            ax.set_ylabel('T-squared')
            ax.legend()
            return fig, ax

        result = self.generate_plot(plot_t_squared)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_hotellings_t_squared.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
            
            results = {
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers) / n) * 100
            }
            
            self.interpret_results("Hotelling's T-squared Analysis", {
                'image_paths': image_paths,
                'results': results
            }, table_name)
        else:
            print("Skipping Hotelling's T-squared plot due to timeout.")

    def breakdown_point_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Breakdown Point Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = df[col].dropna().values
            n = len(data)
            
            # Calculate breakdown point for mean and median
            bp_mean = 1 / n
            bp_median = 0.5
            
            # Calculate trimmed mean with different trimming levels
            trim_levels = [0.1, 0.2, 0.3]
            trimmed_means = [stats.trim_mean(data, trim) for trim in trim_levels]
            
            results[col] = {
                'bp_mean': bp_mean,
                'bp_median': bp_median,
                'trimmed_means': dict(zip(trim_levels, trimmed_means))
            }

            def plot_breakdown_point():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                ax1.bar(['Mean', 'Median'], [bp_mean, bp_median])
                ax1.set_title(f'Breakdown Point - {col}')
                ax1.set_ylabel('Breakdown Point')
                
                ax2.plot(trim_levels, trimmed_means, marker='o')
                ax2.set_title(f'Trimmed Means - {col}')
                ax2.set_xlabel('Trimming Level')
                ax2.set_ylabel('Trimmed Mean')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_breakdown_point)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_breakdown_point.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Breakdown Point plot for {col} due to timeout.")
        
        self.interpret_results("Breakdown Point Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

    def chi_square_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Chi-Square Test Analysis"))
        image_paths = []
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        results = {}
        
        for col in categorical_cols:
            observed = df[col].value_counts()
            n = len(df)
            expected = pd.Series(n/len(observed), index=observed.index)
            
            chi2, p_value = stats.chisquare(observed, expected)
            
            results[col] = {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': len(observed) - 1
            }

            def plot_chi_square():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                x = np.arange(len(observed))
                width = 0.35
                ax.bar(x - width/2, observed, width, label='Observed')
                ax.bar(x + width/2, expected, width, label='Expected')
                ax.set_xlabel('Categories')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Chi-Square Test - {col}')
                ax.set_xticks(x)
                ax.set_xticklabels(observed.index, rotation=45, ha='right')
                ax.legend()
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_chi_square)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_chi_square.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Chi-Square Test plot for {col} due to timeout.")
        
        self.interpret_results("Chi-Square Test Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

    def simple_thresholding_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Simple Thresholding Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            results[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers) / len(data)) * 100
            }

            def plot_simple_thresholding():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.boxplot(data)
                ax.scatter(np.ones(len(outliers)), outliers, color='red', label='Outliers')
                ax.set_title(f'Simple Thresholding - {col}')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_simple_thresholding)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_simple_thresholding.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Simple Thresholding plot for {col} due to timeout.")
        
        self.interpret_results("Simple Thresholding Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

    def lilliefors_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Lilliefors Test Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = df[col].dropna().values
            
            statistic, p_value = lilliefors(data)
            
            results[col] = {
                'test_statistic': statistic,
                'p_value': p_value
            }

            def plot_lilliefors():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(f'Lilliefors Test Q-Q Plot - {col}')
                return fig, ax

            result = self.generate_plot(plot_lilliefors)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_lilliefors.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Lilliefors Test plot for {col} due to timeout.")
        
        self.interpret_results("Lilliefors Test Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

    def jarque_bera_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Jarque-Bera Test Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = df[col].dropna().values
            
            statistic, p_value, skew, kurtosis = jarque_bera(data)
            
            results[col] = {
                'test_statistic': statistic,
                'p_value': p_value,
                'skewness': skew,
                'kurtosis': kurtosis
            }

            def plot_jarque_bera():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                sns.histplot(data, kde=True, ax=ax1)
                ax1.set_title(f'Distribution - {col}')
                
                stats.probplot(data, dist="norm", plot=ax2)
                ax2.set_title(f'Q-Q Plot - {col}')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_jarque_bera)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_jarque_bera.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Jarque-Bera Test plot for {col} due to timeout.")
        
        self.interpret_results("Jarque-Bera Test Analysis", {
            'image_paths': image_paths,
            'results': results
        }, table_name)

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
        output_file = os.path.join(self.output_folder, "axda_b5_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    

    
    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 5) Report for {self.table_name}"
        
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
            filename=f"axda_b5_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None
