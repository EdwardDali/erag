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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.manifold import MDS, TSNE
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import EllipticEnvelope

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.outliers_influence import OLSInfluence
import pgmpy
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
        self.total_techniques = 15  # Updated to include new techniques
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

    def preprocess_data(self, df, min_unique_values=2, max_missing_percentage=30, sample_size=10000):
        # Select numerical columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        print(f"Original numerical columns: {list(numerical_columns)}")
        
        # Check for missing data and data types
        valid_columns = []
        for col in numerical_columns:
            missing_percentage = df[col].isnull().mean() * 100
            unique_count = df[col].nunique()
            
            if missing_percentage < max_missing_percentage and unique_count >= min_unique_values:
                valid_columns.append(col)
            else:
                print(f"Skipping column {col}: {missing_percentage:.2f}% missing, {unique_count} unique values")
        
        print(f"Valid numerical columns after preprocessing: {valid_columns}")
        
        # Select valid columns
        X = df[valid_columns]
        
        # Handle remaining NaN values using mean imputation
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=valid_columns)
        
        # Sample data if it's too large
        if len(X_imputed) > sample_size:
            X_imputed = X_imputed.sample(sample_size, random_state=42)
        
        return X_imputed

    def run(self):
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch 3) on {self.db_path}"))
        
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
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch 3) completed. Results saved in {self.output_folder}"))

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
            method(df, table_name)

    def factor_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Factor Analysis"))
        
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
                self.interpret_results("Factor Analysis", img_path, table_name)
            else:
                print("Skipping Factor Analysis plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Factor Analysis.")

    def multidimensional_scaling(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Multidimensional Scaling (MDS)"))
        
        preprocessed_df = self.preprocess_data(df)
        if preprocessed_df.shape[1] >= 2:
            def plot_mds():
                mds = MDS(n_components=2, random_state=42)
                mds_result = mds.fit_transform(preprocessed_df)
                
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
                self.interpret_results("Multidimensional Scaling (MDS)", img_path, table_name)
            else:
                print("Skipping Multidimensional Scaling (MDS) plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Multidimensional Scaling (MDS).")

    def t_sne(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - t-Distributed Stochastic Neighbor Embedding (t-SNE)"))
        
        preprocessed_df = self.preprocess_data(df)
        if preprocessed_df.shape[1] >= 2:
            def plot_tsne():
                tsne = TSNE(n_components=2, random_state=42)
                tsne_result = tsne.fit_transform(preprocessed_df)
                
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
                self.interpret_results("t-Distributed Stochastic Neighbor Embedding (t-SNE)", img_path, table_name)
            else:
                print("Skipping t-SNE plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for t-SNE.")

    def conditional_plots(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Conditional Plots"))
        
        try:
            preprocessed_df = self.preprocess_data(df)
            numerical_columns = preprocessed_df.columns
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns

            print(f"Numerical columns after preprocessing: {list(numerical_columns)}")
            print(f"Categorical columns: {list(categorical_columns)}")

            if len(numerical_columns) >= 2 and len(categorical_columns) > 0:
                def plot_coplots():
                    x = numerical_columns[0]
                    y = numerical_columns[1]
                    z = categorical_columns[0]
                    
                    print(f"Plotting conditional plots for x: {x}, y: {y}, conditioned on z: {z}")
                    
                    # Limit the number of categories to plot
                    top_categories = df[z].value_counts().nlargest(5).index
                    plot_data = df[df[z].isin(top_categories)].copy()
                    plot_data[x] = preprocessed_df[x]
                    plot_data[y] = preprocessed_df[y]
                    
                    g = sns.FacetGrid(plot_data, col=z, col_wrap=3, height=4, aspect=1.5)
                    g.map(sns.scatterplot, x, y)
                    g.add_legend()
                    g.fig.suptitle(f'Conditional Plots: {y} vs {x} conditioned on {z}')
                    plt.tight_layout()
                    return g.fig, g.axes

                result = self.generate_plot(plot_coplots)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_conditional_plots.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    self.interpret_results("Conditional Plots", img_path, table_name)
                else:
                    print("Skipping Conditional Plots due to error in plot generation.")
            else:
                print("Not enough suitable columns for Conditional Plots.")
                print(f"Number of numerical columns: {len(numerical_columns)}")
                print(f"Number of categorical columns: {len(categorical_columns)}")
        except Exception as e:
            print(f"An error occurred during Conditional Plots analysis: {str(e)}")
            print("Skipping Conditional Plots due to error.")

    def ice_plots(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Individual Conditional Expectation (ICE) Plots"))
        
        preprocessed_df = self.preprocess_data(df)
        if preprocessed_df.shape[1] >= 2:
            def plot_ice():
                X = preprocessed_df.iloc[:, :-1]
                y = preprocessed_df.iloc[:, -1]
                
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X, y)
                
                feature = X.columns[0]  # Use the first column as the feature for ICE plot
                
                ice_data = []
                x_range = np.linspace(X[feature].min(), X[feature].max(), num=50)
                for i in range(min(50, len(X))):  # Limit to 50 ICE curves
                    ice_curve = []
                    X_copy = X.iloc[[i]].copy()
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
                self.interpret_results("Individual Conditional Expectation (ICE) Plots", img_path, table_name)
            else:
                print("Skipping ICE Plots due to error in plot generation.")
        else:
            print("Not enough numerical columns for ICE Plots.")

    def time_series_decomposition(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Time Series Decomposition"))
        
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
                self.interpret_results("Time Series Decomposition", img_path, table_name)
            else:
                print("Skipping Time Series Decomposition due to error in plot generation.")
        else:
            print("No suitable date and numerical columns found for Time Series Decomposition.")

    def autocorrelation_plots(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Autocorrelation Plots"))
        
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
                self.interpret_results("Autocorrelation Plots", img_path, table_name)
            else:
                print("Skipping Autocorrelation Plot due to error in plot generation.")
        else:
            print("No numerical columns found for Autocorrelation Plot.")

    def bayesian_networks(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Bayesian Networks"))
        
        # Select a subset of columns for Bayesian Network analysis
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
                self.interpret_results("Bayesian Networks", img_path, table_name)
            else:
                print("Skipping Bayesian Network plot due to error in plot generation.")
        else:
            print("Not enough suitable columns for Bayesian Network analysis.")

    def isolation_forest(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Isolation Forest"))
        
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
                self.interpret_results("Isolation Forest", img_path, table_name)
            else:
                print("Skipping Isolation Forest plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Isolation Forest analysis.")

    def one_class_svm(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - One-Class SVM"))
        
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
                self.interpret_results("One-Class SVM", img_path, table_name)
            else:
                print("Skipping One-Class SVM plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for One-Class SVM analysis.")


    def local_outlier_factor(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Local Outlier Factor (LOF)"))
        
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
                self.interpret_results("Local Outlier Factor (LOF)", img_path, table_name)
            else:
                print("Skipping Local Outlier Factor plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Local Outlier Factor analysis.")

    def robust_pca(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Robust Principal Component Analysis (RPCA)"))
        
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
                self.interpret_results("Robust Principal Component Analysis (RPCA)", img_path, table_name)
            else:
                print("Skipping Robust PCA plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Robust PCA analysis.")

    def bayesian_change_point_detection(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Bayesian Change Point Detection"))
        
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
                self.interpret_results("Bayesian Change Point Detection", img_path, table_name)
            else:
                print("Skipping Change Point Detection plot due to error in plot generation.")
        else:
            print("No suitable date and numerical columns found for Change Point Detection.")

    def hidden_markov_models(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hidden Markov Models (HMMs)"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_hmm():
                X = df[numerical_columns].values
                
                # Fit HMM
                model = hmm.GaussianHMM(n_components=3, covariance_type="full")
                model.fit(X)
                
                # Predict hidden states
                hidden_states = model.predict(X)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                for i in range(model.n_components):
                    idx = (hidden_states == i)
                    ax.plot(X[idx, 0], X[idx, 1], 'o', label=f'State {i}')
                ax.legend()
                ax.set_title('Hidden Markov Model States')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_hmm)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_hidden_markov_model.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Hidden Markov Models (HMMs)", img_path, table_name)
            else:
                print("Skipping Hidden Markov Model plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Hidden Markov Model analysis.")

    def dynamic_time_warping(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Dynamic Time Warping (DTW)"))
        
        preprocessed_df = self.preprocess_data(df)
        numerical_columns = preprocessed_df.columns
        
        if len(numerical_columns) >= 2:
            def plot_dtw():
                # Select two time series for comparison
                series1 = preprocessed_df[numerical_columns[0]].values
                series2 = preprocessed_df[numerical_columns[1]].values
                
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
                self.interpret_results("Dynamic Time Warping (DTW)", img_path, table_name)
            else:
                print("Skipping Dynamic Time Warping plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Dynamic Time Warping analysis.")

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
            self.executive_summary = "No significant findings were identified during the advanced analysis. This could be due to a lack of data, uniform data distribution, or absence of notable patterns or anomalies in the dataset."
            return

        summary_prompt = f"""
        Based on the following findings from the Advanced Exploratory Data Analysis (Batch 3):
        
        {self.findings}
        
        Please provide an executive summary of the analysis. The summary should:
        1. Briefly introduce the purpose of the advanced analysis.
        2. Highlight the most significant insights and patterns discovered.
        3. Mention any potential issues or areas that require further investigation.
        4. Conclude with recommendations for next steps or areas to focus on.

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
        output_file = os.path.join(self.output_folder, "axda_b3_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 3) Report for {self.table_name}"
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.executive_summary,
            self.findings,
            self.pdf_content,
            self.image_data,
            filename=f"axda_b3_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
        else:
            print(error("Failed to generate PDF report"))
