import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, anderson, pearsonr, probplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import networkx as nx
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

class AdvancedExploratoryDataAnalysisB1:  # Updated class name
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


    def preprocess_data(self, df, min_unique_values=2, max_missing_percentage=30):
        # Select numerical columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Check for missing data and data types
        valid_columns = []
        for col in numerical_columns:
            missing_percentage = df[col].isnull().mean() * 100
            unique_count = df[col].nunique()
            
            if missing_percentage < max_missing_percentage and unique_count >= min_unique_values:
                valid_columns.append(col)
            else:
                print(f"Skipping column {col}: {missing_percentage:.2f}% missing, {unique_count} unique values")
        
        # Select valid columns
        X = df[valid_columns]
        
        # Handle remaining NaN values using mean imputation
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        return pd.DataFrame(X_imputed, columns=valid_columns)

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
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch 1) on {self.db_path}"))
        
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
        print(success(f"Advanced Exploratory Data Analysis (Batch 2) completed. Results saved in {self.output_folder}"))



    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_b1_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        analysis_methods = [
            self.value_counts_analysis,
            self.grouped_summary_statistics,
            self.frequency_distribution_analysis,
            self.kde_plot_analysis,
            self.violin_plot_analysis,
            self.pair_plot_analysis,
            self.box_plot_analysis,
            self.scatter_plot_analysis,
            self.time_series_analysis,
            self.outlier_detection,
            self.feature_importance_analysis,
            self.pca_analysis,
            self.cluster_analysis,
            self.correlation_network_analysis,
            self.qq_plot_analysis
        ]

        for method in analysis_methods:
            method(df, table_name)

    def value_counts_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Value Counts Analysis"))
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        results = {}
        
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            results[col] = value_counts.to_dict()
        
        self.interpret_results("Value Counts Analysis", results, table_name)

    def grouped_summary_statistics(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Grouped Summary Statistics"))
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        results = {}
        
        for cat_col in categorical_columns:
            for num_col in numerical_columns:
                grouped_stats = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'min', 'max'])
                results[f"{cat_col} - {num_col}"] = grouped_stats.to_dict()
        
        self.interpret_results("Grouped Summary Statistics", results, table_name)

    def frequency_distribution_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Frequency Distribution Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        
        for col in numerical_columns:
            def plot_frequency_distribution():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f'Frequency Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                return fig, ax

            result = self.generate_plot(plot_frequency_distribution)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_frequency_distribution.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping frequency distribution plot for {col} due to timeout.")
        
        self.interpret_results("Frequency Distribution Analysis", image_paths, table_name)

    def kde_plot_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - KDE Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        
        for col in numerical_columns:
            def plot_kde():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.kdeplot(df[col], shade=True, ax=ax)
                ax.set_title(f'KDE Plot for {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                return fig, ax

            result = self.generate_plot(plot_kde)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_kde_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping KDE plot for {col} due to timeout.")
        
        self.interpret_results("KDE Plot Analysis", image_paths, table_name)

    def violin_plot_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Violin Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        image_paths = []
        
        if len(categorical_columns) > 0:
            for num_col in numerical_columns:
                def plot_violin():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.violinplot(x=categorical_columns[0], y=num_col, data=df, ax=ax)
                    ax.set_title(f'Violin Plot of {num_col} grouped by {categorical_columns[0]}')
                    ax.set_xlabel(categorical_columns[0])
                    ax.set_ylabel(num_col)
                    plt.xticks(rotation=45)
                    return fig, ax

                result = self.generate_plot(plot_violin)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{num_col}_violin_plot.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping violin plot for {num_col} due to timeout.")
        else:
            print("No categorical columns found for violin plot analysis.")
        
        self.interpret_results("Violin Plot Analysis", image_paths, table_name)


    def pair_plot_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Pair Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            def plot_pair():
                fig = sns.pairplot(df[numerical_columns], height=3, aspect=1.2)
                fig.fig.suptitle("Pair Plot of Numerical Variables", y=1.02)
                return fig

            result = self.generate_plot(plot_pair)
            if result is not None:
                img_path = os.path.join(self.output_folder, f"{table_name}_pair_plot.png")
                result.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(result.fig)
                self.interpret_results("Pair Plot Analysis", img_path, table_name)
            else:
                print("Skipping pair plot due to timeout.")
        else:
            print("Not enough numerical columns for pair plot analysis.")

    def box_plot_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Box Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        
        for col in numerical_columns:
            def plot_box():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.boxplot(y=df[col], ax=ax)
                ax.set_title(f'Box Plot of {col}')
                ax.set_ylabel(col)
                return fig, ax

            result = self.generate_plot(plot_box)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_box_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping box plot for {col} due to timeout.")
        
        self.interpret_results("Box Plot Analysis", image_paths, table_name)

    def scatter_plot_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Scatter Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        
        if len(numerical_columns) > 1:
            for i in range(len(numerical_columns)):
                for j in range(i+1, len(numerical_columns)):
                    col1, col2 = numerical_columns[i], numerical_columns[j]
                    def plot_scatter():
                        fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                        sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
                        ax.set_title(f'Scatter Plot of {col1} vs {col2}')
                        ax.set_xlabel(col1)
                        ax.set_ylabel(col2)
                        return fig, ax

                    result = self.generate_plot(plot_scatter)
                    if result is not None:
                        fig, ax = result
                        img_path = os.path.join(self.output_folder, f"{table_name}_{col1}_vs_{col2}_scatter_plot.png")
                        plt.savefig(img_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        image_paths.append(img_path)
                    else:
                        print(f"Skipping scatter plot for {col1} vs {col2} due to timeout.")
        else:
            print("Not enough numerical columns for scatter plot analysis.")
        
        self.interpret_results("Scatter Plot Analysis", image_paths, table_name)

    def time_series_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Time Series Analysis"))
        
        date_columns = df.select_dtypes(include=['datetime64']).columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        
        if len(date_columns) > 0 and len(numerical_columns) > 0:
            date_col = date_columns[0]
            for num_col in numerical_columns:
                def plot_time_series():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    sns.lineplot(x=df[date_col], y=df[num_col], ax=ax)
                    ax.set_title(f'Time Series Plot of {num_col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(num_col)
                    plt.xticks(rotation=45)
                    return fig, ax

                result = self.generate_plot(plot_time_series)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{num_col}_time_series_plot.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping time series plot for {num_col} due to timeout.")
        else:
            print("No suitable columns for time series analysis.")
        
        self.interpret_results("Time Series Analysis", image_paths, table_name)

    def outlier_detection(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Outlier Detection"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        
        for col in numerical_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            results[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'range': (lower_bound, upper_bound)
            }
        
        self.interpret_results("Outlier Detection", results, table_name)

    def feature_importance_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Feature Importance Analysis"))
        
        preprocessed_df = self.preprocess_data(df)
        
        if preprocessed_df.shape[1] > 1:
            X = preprocessed_df.drop(preprocessed_df.columns[-1], axis=1)
            y = preprocessed_df[preprocessed_df.columns[-1]]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            def plot_feature_importance():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
                ax.set_title('Feature Importance')
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                return fig, ax

            result = self.generate_plot(plot_feature_importance)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_feature_importance.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Feature Importance Analysis", img_path, table_name)
            else:
                print("Skipping feature importance plot due to timeout.")
        else:
            print("Not enough numerical columns for feature importance analysis.")

    def pca_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - PCA Analysis"))
        
        preprocessed_df = self.preprocess_data(df)
        
        if preprocessed_df.shape[1] > 2:
            X_scaled = StandardScaler().fit_transform(preprocessed_df)
            
            pca = PCA()
            pca_result = pca.fit_transform(X_scaled)
            
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            def plot_pca():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Scree plot
                ax1.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, 'bo-')
                ax1.set_xlabel('Principal Component')
                ax1.set_ylabel('Explained Variance Ratio')
                ax1.set_title('Scree Plot')
                
                # Cumulative explained variance plot
                ax2.plot(range(1, len(cumulative_variance_ratio)+1), cumulative_variance_ratio, 'ro-')
                ax2.set_xlabel('Number of Components')
                ax2.set_ylabel('Cumulative Explained Variance Ratio')
                ax2.set_title('Cumulative Explained Variance')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_pca)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_pca_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("PCA Analysis", img_path, table_name)
            else:
                print("Skipping PCA analysis plot due to timeout.")
        else:
            print("Not enough numerical columns for PCA analysis.")

    def cluster_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Cluster Analysis"))
        
        # Preprocess the data
        preprocessed_df = self.preprocess_data(df)
        
        if preprocessed_df.shape[1] > 1:
            X_scaled = StandardScaler().fit_transform(preprocessed_df)
            
            # Determine optimal number of clusters using elbow method
            inertias = []
            max_clusters = min(10, preprocessed_df.shape[0] - 1)  # Limit to 10 clusters or one less than number of samples
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point
            elbow = next(i for i in range(1, len(inertias)) if inertias[i-1] - inertias[i] < (inertias[0] - inertias[-1]) / 10)
            
            # Perform K-means clustering with optimal number of clusters
            kmeans = KMeans(n_clusters=elbow, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            def plot_clusters():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Elbow plot
                ax1.plot(range(1, max_clusters + 1), inertias, 'bo-')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Inertia')
                ax1.set_title('Elbow Method for Optimal k')
                ax1.axvline(x=elbow, color='r', linestyle='--')
                
                # 2D projection of clusters
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
                ax2.set_xlabel('First Principal Component')
                ax2.set_ylabel('Second Principal Component')
                ax2.set_title(f'2D PCA Projection of {elbow} Clusters')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_clusters)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_cluster_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Cluster Analysis", img_path, table_name)
            else:
                print("Skipping cluster analysis plot due to timeout.")
        else:
            print("Not enough numerical columns for cluster analysis.")

    def correlation_network_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Correlation Network Analysis"))
        
        preprocessed_df = self.preprocess_data(df)
        
        if preprocessed_df.shape[1] > 1:
            corr_matrix = preprocessed_df.corr()
            
            # Create a graph from the correlation matrix
            G = nx.Graph()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:  # Only add edges for correlations > 0.5
                        G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=abs(corr_matrix.iloc[i, j]))
                
            def plot_correlation_network():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                pos = nx.spring_layout(G)
                nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
                edge_weights = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, ax=ax)
                ax.set_title('Correlation Network')
                return fig, ax

            result = self.generate_plot(plot_correlation_network)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_correlation_network.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Correlation Network Analysis", img_path, table_name)
            else:
                print("Skipping correlation network plot due to timeout.")
        else:
            print("Not enough numerical columns for correlation network analysis.")

    def qq_plot_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Q-Q Plot Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        
        for col in numerical_columns:
            def plot_qq():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                probplot(df[col], dist="norm", plot=ax)
                ax.set_title(f'Q-Q Plot of {col}')
                return fig, ax

            result = self.generate_plot(plot_qq)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_qq_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Q-Q plot for {col} due to timeout.")
        
        self.interpret_results("Q-Q Plot Analysis", image_paths, table_name)

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
        if isinstance(results, list):
            for i, item in enumerate(results):
                if isinstance(item, str) and item.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_data.append((f"{analysis_type} - Image {i+1}", item))
        
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
        Based on the following findings from the Advanced Exploratory Data Analysis:
        
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
        output_file = os.path.join(self.output_folder, "axda_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 1) Report for {self.table_name}"
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.executive_summary,
            self.findings,
            self.pdf_content,
            self.image_data,
            filename=f"axda_b1_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
        else:
            print(error("Failed to generate PDF report"))
