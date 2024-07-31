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
from scipy.stats import norm, anderson, pearsonr, probplot
from scipy.cluster.hierarchy import dendrogram
from scipy.signal import find_peaks

from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.manifold import MDS, TSNE
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.impute import SimpleImputer

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.outliers_influence import OLSInfluence
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator

from hmmlearn import hmm
from dtaidistance import dtw

from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator

import networkx as nx

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

class TimeoutException(Exception):
    pass

class AdvancedExploratoryDataAnalysisB3:
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

    def run(self):
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch3) on {self.db_path}"))
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch3) completed. Results saved in {self.output_folder}"))



    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_b3_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        analysis_methods = [
            self.factor_analysis,
            self.multidimensional_scaling,
            self.t_sne,
            self.conditional_plots,
            self.ice_plots,
            self.time_series_decomposition,
            self.autocorrelation_plots,
            self.bayesian_networks,
            self.isolation_forest,
            self.one_class_svm,
            self.local_outlier_factor,
            self.robust_pca,
            self.bayesian_change_point_detection,
            self.hidden_markov_models,
            self.dynamic_time_warping
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

    def factor_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Factor Analysis"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            def plot_factor_analysis():
                # Prepare the data
                X = df[numerical_columns]
                imputer = SimpleImputer(strategy='mean')
                X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                
                # Perform Factor Analysis
                fa = FactorAnalysis(n_components=min(5, len(numerical_columns)), random_state=42)
                fa_result = fa.fit_transform(X_imputed)
                
                # Create the plot
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.heatmap(fa.components_, annot=True, cmap='coolwarm', ax=ax)
                ax.set_xlabel('Original Features')
                ax.set_ylabel('Factors')
                ax.set_title('Factor Analysis Loadings')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_factor_analysis)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_factor_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Factor Analysis plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Factor Analysis.")
        self.interpret_results("Factor Analysis", {'image_paths': image_paths}, table_name)

    def multidimensional_scaling(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Multidimensional Scaling (MDS)"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_mds():
                X = df[numerical_columns]
                imputer = SimpleImputer(strategy='mean')
                X_imputed = imputer.fit_transform(X)
                
                mds = MDS(n_components=2, random_state=42)
                mds_result = mds.fit_transform(X_imputed)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(mds_result[:, 0], mds_result[:, 1], alpha=0.6)
                ax.set_xlabel('MDS Dimension 1')
                ax.set_ylabel('MDS Dimension 2')
                ax.set_title('Multidimensional Scaling (MDS)')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_mds)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_mds.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Multidimensional Scaling (MDS) plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Multidimensional Scaling (MDS).")
        self.interpret_results("Multidimensional Scaling (MDS)", {'image_paths': image_paths}, table_name)

    def t_sne(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - t-Distributed Stochastic Neighbor Embedding (t-SNE)"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_tsne():
                X = df[numerical_columns]
                imputer = SimpleImputer(strategy='mean')
                X_imputed = imputer.fit_transform(X)
                
                tsne = TSNE(n_components=2, random_state=42)
                tsne_result = tsne.fit_transform(X_imputed)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
                ax.set_xlabel('t-SNE Dimension 1')
                ax.set_ylabel('t-SNE Dimension 2')
                ax.set_title('t-Distributed Stochastic Neighbor Embedding (t-SNE)')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_tsne)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_tsne.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping t-SNE plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for t-SNE.")
        self.interpret_results("t-Distributed Stochastic Neighbor Embedding (t-SNE)", {'image_paths': image_paths}, table_name)

    def conditional_plots(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Conditional Plots"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        if len(numerical_columns) >= 2 and len(categorical_columns) > 0:
            x = numerical_columns[0]
            y = numerical_columns[1]
            z = categorical_columns[0]
            
            # Limit the number of categories to plot
            top_categories = df[z].value_counts().nlargest(5).index
            plot_data = df[df[z].isin(top_categories)].copy()
            
            for category in top_categories:
                def plot_conditional(cat):
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.scatterplot(data=plot_data[plot_data[z] == cat], x=x, y=y, ax=ax)
                    ax.set_title(f'{y} vs {x} for {z}={cat}')
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    return fig, ax

                result = self.generate_plot(lambda: plot_conditional(category))
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_conditional_plot_{category}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping Conditional Plot for {category} due to error in plot generation.")
        else:
            print("Not enough suitable columns for Conditional Plots.")
        self.interpret_results("Conditional Plots", {'image_paths': image_paths}, table_name)

    def ice_plots(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Individual Conditional Expectation (ICE) Plots"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_ice():
                X = df[numerical_columns]
                y = X.iloc[:, -1]  # Use the last column as the target
                X = X.iloc[:, :-1]  # Use all but the last column as features
                
                imputer = SimpleImputer(strategy='mean')
                X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_imputed, y)
                
                feature = X.columns[0]  # Use the first column as the feature for ICE plot
                
                ice_data = []
                x_range = np.linspace(X[feature].min(), X[feature].max(), num=50)
                for i in range(min(50, len(X))):  # Limit to 50 ICE curves
                    ice_curve = []
                    X_copy = X_imputed.iloc[[i]].copy()
                    for x in x_range:
                        X_copy[feature] = x
                        pred = model.predict(X_copy)[0]
                        ice_curve.append(pred)
                    ice_data.append(ice_curve)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                for curve in ice_data:
                    ax.plot(x_range, curve, color='blue', alpha=0.1)
                ax.set_xlabel(feature)
                ax.set_ylabel('Predicted')
                ax.set_title(f'ICE Plot for {feature}')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_ice)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_ice_plots.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping ICE Plots due to error in plot generation.")
        else:
            print("Not enough numerical columns for ICE Plots.")
        self.interpret_results("Individual Conditional Expectation (ICE) Plots", {'image_paths': image_paths}, table_name)

    def time_series_decomposition(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Time Series Decomposition"))
        image_paths = []
        
        date_columns = df.select_dtypes(include=['datetime64']).columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(date_columns) > 0 and len(numerical_columns) > 0:
            date_col = date_columns[0]
            num_col = numerical_columns[0]
            
            # Ensure the date column is set as the index and sort
            df_sorted = df.set_index(date_col).sort_index()
            ts = df_sorted[num_col]
            
            # Resample to daily frequency if necessary
            if ts.index.inferred_freq is None:
                ts = ts.resample('D').mean()
            
            # Interpolate missing values
            ts = ts.interpolate()
            
            def plot_decomposition():
                # Perform decomposition
                result = seasonal_decompose(ts, model='additive', period=7)  # Assuming weekly seasonality
                
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.calculate_figure_size())
                result.observed.plot(ax=ax1)
                ax1.set_title('Observed')
                result.trend.plot(ax=ax2)
                ax2.set_title('Trend')
                result.seasonal.plot(ax=ax3)
                ax3.set_title('Seasonal')
                result.resid.plot(ax=ax4)
                ax4.set_title('Residual')
                plt.tight_layout()
                return fig, (ax1, ax2, ax3, ax4)

            result = self.generate_plot(plot_decomposition)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_time_series_decomposition.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Time Series Decomposition due to error in plot generation.")
        else:
            print("No suitable date and numerical columns found for Time Series Decomposition.")
        self.interpret_results("Time Series Decomposition", {'image_paths': image_paths}, table_name)

    def autocorrelation_plots(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Autocorrelation Plots"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_acf():
                data = df[numerical_columns[0]].dropna()
                lag_acf = acf(data, nlags=40)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.bar(range(len(lag_acf)), lag_acf)
                ax.set_xlabel('Lag')
                ax.set_ylabel('Autocorrelation')
                ax.set_title(f'Autocorrelation Plot for {numerical_columns[0]}')
                
                # Add confidence intervals
                ax.axhline(y=0, linestyle='--', color='gray')
                ax.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
                ax.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
                
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_acf)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_autocorrelation_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Autocorrelation Plot due to error in plot generation.")
        else:
            print("No numerical columns found for Autocorrelation Plot.")
        self.interpret_results("Autocorrelation Plots", {'image_paths': image_paths}, table_name)

    def bayesian_networks(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Bayesian Networks"))
        image_paths = []
        
        columns = df.select_dtypes(include=['float64', 'int64', 'bool', 'category']).columns[:5]
        if len(columns) >= 2:
            def plot_bayesian_network():
                data = df[columns]
                
                # Learn the structure of the Bayesian Network
                hc = HillClimbSearch(data)
                best_model = hc.estimate()
                
                # Fit the parameters of the Bayesian Network
                model = BayesianNetwork(best_model.edges())
                model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")
                
                # Create a networkx graph from the model's edges
                G = nx.DiGraph()
                G.add_edges_from(model.edges())
                
                # Plot the Bayesian Network
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                        node_size=3000, font_size=10, font_weight='bold', ax=ax)
                
                # Add edge labels (probabilities)
                edge_labels = {(u, v): f"{u}->{v}" for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
                
                ax.set_title('Bayesian Network')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_bayesian_network)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_bayesian_network.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
                # Add a heatmap of the learned CPDs
                def plot_cpd_heatmap():
                    fig, axes = plt.subplots(1, len(columns), figsize=(20, 5))
                    for i, node in enumerate(columns):
                        cpd = model.get_cpds(node)
                        sns.heatmap(cpd.values, annot=True, cmap='YlGnBu', ax=axes[i])
                        axes[i].set_title(f'CPD for {node}')
                    plt.tight_layout()
                    return fig, axes

                result_cpd = self.generate_plot(plot_cpd_heatmap)
                if result_cpd is not None:
                    fig_cpd, _ = result_cpd
                    img_path_cpd = os.path.join(self.output_folder, f"{table_name}_bayesian_network_cpd.png")
                    plt.savefig(img_path_cpd, dpi=100, bbox_inches='tight')
                    plt.close(fig_cpd)
                    image_paths.append(img_path_cpd)
            else:
                print("Skipping Bayesian Network plot due to error in plot generation.")
        else:
            print("Not enough suitable columns for Bayesian Network analysis.")
        self.interpret_results("Bayesian Networks", {'image_paths': image_paths}, table_name)

    def isolation_forest(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Isolation Forest"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_isolation_forest():
                X = df[numerical_columns]
                
                # Fit Isolation Forest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(X)
                
                # Select two features for visualization
                feature1, feature2 = numerical_columns[:2]
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X[feature1], X[feature2], c=outlier_labels, cmap='viridis')
                ax.set_xlabel(feature1)
                ax.set_ylabel(feature2)
                ax.set_title('Isolation Forest Outlier Detection')
                plt.colorbar(scatter, label='Outlier Score')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_isolation_forest)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_isolation_forest.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Isolation Forest plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Isolation Forest analysis.")
        self.interpret_results("Isolation Forest", {'image_paths': image_paths}, table_name)

    def one_class_svm(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - One-Class SVM"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_one_class_svm():
                X = df[numerical_columns]
                
                # Fit One-Class SVM
                svm = OneClassSVM(kernel='rbf', nu=0.1)
                svm.fit(X)
                y_pred = svm.predict(X)
                
                # Select two features for visualization
                feature1, feature2 = numerical_columns[:2]
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X[feature1], X[feature2], c=y_pred, cmap='viridis')
                ax.set_xlabel(feature1)
                ax.set_ylabel(feature2)
                ax.set_title('One-Class SVM Anomaly Detection')
                plt.colorbar(scatter, label='Anomaly Score')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_one_class_svm)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_one_class_svm.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping One-Class SVM plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for One-Class SVM analysis.")
        self.interpret_results("One-Class SVM", {'image_paths': image_paths}, table_name)


    def local_outlier_factor(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Local Outlier Factor (LOF)"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_lof():
                X = df[numerical_columns]
                
                # Fit Local Outlier Factor
                lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                y_pred = lof.fit_predict(X)
                
                # Select two features for visualization
                feature1, feature2 = numerical_columns[:2]
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(X[feature1], X[feature2], c=y_pred, cmap='viridis')
                ax.set_xlabel(feature1)
                ax.set_ylabel(feature2)
                ax.set_title('Local Outlier Factor (LOF)')
                plt.colorbar(scatter, label='Outlier Score')
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_lof)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_local_outlier_factor.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Local Outlier Factor plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Local Outlier Factor analysis.")
        self.interpret_results("Local Outlier Factor (LOF)", {'image_paths': image_paths}, table_name)

    def robust_pca(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Robust Principal Component Analysis (RPCA)"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_rpca():
                X = df[numerical_columns]
                
                # Perform Robust PCA
                rpca = EllipticEnvelope(contamination=0.1, random_state=42)
                y_pred = rpca.fit_predict(X)
                
                # Perform standard PCA for comparison
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                
                # Plot standard PCA
                scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.6)
                ax1.set_title('Standard PCA')
                ax1.set_xlabel('PC1')
                ax1.set_ylabel('PC2')
                
                # Plot Robust PCA
                scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis')
                ax2.set_title('Robust PCA')
                ax2.set_xlabel('PC1')
                ax2.set_ylabel('PC2')
                
                plt.colorbar(scatter2, ax=ax2, label='Outlier Score')
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_rpca)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_robust_pca.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print("Skipping Robust PCA plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Robust PCA analysis.")
        self.interpret_results("Robust Principal Component Analysis (RPCA)", {'image_paths': image_paths}, table_name)


    def bayesian_change_point_detection(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Bayesian Change Point Detection"))
        image_paths = []
        
        date_columns = df.select_dtypes(include=['datetime64']).columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(date_columns) > 0 and len(numerical_columns) > 0:
            def plot_change_points():
                date_col = date_columns[0]
                num_col = numerical_columns[0]
                
                df_sorted = df.sort_values(by=date_col)
                ts = df_sorted[num_col].values
                
                # Simple change point detection using difference in means
                diff = np.abs(np.diff(ts))
                threshold = np.mean(diff) + 2 * np.std(diff)
                change_points = np.where(diff > threshold)[0]
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(df_sorted[date_col], ts)
                for cp in change_points:
                    ax.axvline(df_sorted[date_col].iloc[cp], color='r', linestyle='--')
                ax.set_title('Bayesian Change Point Detection')
                ax.set_xlabel('Date')
                ax.set_ylabel(num_col)
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_change_points)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_change_point_detection.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Change Point Detection plot due to error in plot generation.")
        else:
            print("No suitable date and numerical columns found for Change Point Detection.")
        self.interpret_results("Bayesian Change Point Detection", {'image_paths': image_paths}, table_name)

    def hidden_markov_models(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hidden Markov Models (HMMs)"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_hmm():
                X = df[numerical_columns].values
                
                # Fit HMM
                model = hmm.GaussianHMM(n_components=3, covariance_type="full")
                model.fit(X)
                
                # Predict hidden states
                hidden_states = model.predict(X)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                
                # Plot states
                for i in range(model.n_components):
                    idx = (hidden_states == i)
                    ax1.plot(X[idx, 0], X[idx, 1], 'o', label=f'State {i}')
                ax1.legend()
                ax1.set_title('Hidden Markov Model States')
                ax1.set_xlabel(numerical_columns[0])
                ax1.set_ylabel(numerical_columns[1])
                
                # Plot state sequence
                ax2.plot(hidden_states)
                ax2.set_title('Hidden State Sequence')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Hidden State')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_hmm)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_hidden_markov_model.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
                # Add transition matrix heatmap
                def plot_transition_matrix():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.heatmap(model.transmat_, annot=True, cmap='YlGnBu', ax=ax)
                    ax.set_title('HMM Transition Matrix')
                    plt.tight_layout()
                    return fig, ax

                result_transition = self.generate_plot(plot_transition_matrix)
                if result_transition is not None:
                    fig_transition, _ = result_transition
                    img_path_transition = os.path.join(self.output_folder, f"{table_name}_hmm_transition_matrix.png")
                    plt.savefig(img_path_transition, dpi=100, bbox_inches='tight')
                    plt.close(fig_transition)
                    image_paths.append(img_path_transition)
            else:
                print("Skipping Hidden Markov Model plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Hidden Markov Model analysis.")
        self.interpret_results("Hidden Markov Models (HMMs)", {'image_paths': image_paths}, table_name)

    def dynamic_time_warping(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Dynamic Time Warping (DTW)"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numerical_columns) >= 2:
            def plot_dtw():
                # Select two time series for comparison
                series1 = df[numerical_columns[0]].values
                series2 = df[numerical_columns[1]].values
                
                # Handle NaN values
                series1 = series1[~np.isnan(series1)]
                series2 = series2[~np.isnan(series2)]
                
                # Ensure series are of equal length
                min_length = min(len(series1), len(series2))
                series1 = series1[:min_length]
                series2 = series2[:min_length]
                
                # Compute DTW distance
                distance = dtw.distance(series1, series2)
                
                # Compute DTW path (limit to first 1000 points for efficiency)
                path = dtw.warping_path(series1[:1000], series2[:1000])
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.calculate_figure_size())
                
                # Plot original time series
                ax1.plot(series1[:1000], label=numerical_columns[0])
                ax1.plot(series2[:1000], label=numerical_columns[1])
                ax1.set_title('Original Time Series (First 1000 points)')
                ax1.legend()
                
                # Plot DTW alignment
                ax2.plot(series1[:1000])
                ax2.plot(series2[:1000])
                for i, j in path:
                    ax2.plot([i, j], [series1[i], series2[j]], 'r-', alpha=0.3)
                ax2.set_title(f'DTW Alignment (Distance: {distance:.2f})')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_dtw)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_dynamic_time_warping.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
            else:
                print("Skipping Dynamic Time Warping plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Dynamic Time Warping analysis.")
        self.interpret_results("Dynamic Time Warping (DTW)", {'image_paths': image_paths}, table_name)

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
        output_file = os.path.join(self.output_folder, "axda_b3_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 3) Report for {self.table_name}"
        
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
            filename=f"axda_b3_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None
