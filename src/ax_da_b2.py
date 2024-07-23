import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, anderson, pearsonr, probplot
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
import networkx as nx
import os
import folium
from wordcloud import WordCloud
from statsmodels.graphics.tsaplots import plot_pacf
import shap
from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import threading
import time
from functools import wraps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
import networkx as nx
import folium
from wordcloud import WordCloud
from statsmodels.graphics.tsaplots import plot_pacf
import shap
from scipy.cluster.hierarchy import dendrogram
from sklearn.impute import SimpleImputer
import joypy
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

class TimeoutException(Exception):
    pass

class AdvancedExploratoryDataAnalysisB2:
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
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch 2) on {self.db_path}"))
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
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
        self.output_folder = os.path.join(settings.output_folder, f"axda_b2_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        analysis_methods = [
            self.parallel_coordinates_plot,
            self.andrews_curves,
            self.radar_charts,
            self.sankey_diagrams,
            self.bubble_charts,
            self.geographical_plots,
            self.word_clouds,
            self.hierarchical_clustering_dendrogram,
            self.ecdf_plots,
            self.ridgeline_plots,
            self.hexbin_plots,
            self.mosaic_plots,
            self.lag_plots,
            self.shapley_value_analysis,
            self.partial_dependence_plots
        ]

        for method in analysis_methods:
            method(df, table_name)

    def parallel_coordinates_plot(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Parallel Coordinates Plot"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            def plot_parallel_coordinates():
                # Convert all columns to float
                df_plot = df[numerical_columns].astype(float)
                
                # If there's no 'target' column, use the first column as a proxy
                target_column = 'target' if 'target' in df_plot.columns else df_plot.columns[0]
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                pd.plotting.parallel_coordinates(df_plot, target_column, ax=ax)
                ax.set_title('Parallel Coordinates Plot')
                plt.xticks(rotation=45)
                return fig, ax

            result = self.generate_plot(plot_parallel_coordinates)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_parallel_coordinates.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Parallel Coordinates Plot", img_path, table_name)
            else:
                print("Skipping parallel coordinates plot due to timeout.")
        else:
            print("Not enough numerical columns for parallel coordinates plot.")

    def andrews_curves(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Andrews Curves"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            def plot_andrews_curves():
                # Convert all columns to float
                df_plot = df[numerical_columns].astype(float)
                
                # If there's no 'target' column, use the first column as a proxy
                target_column = 'target' if 'target' in df_plot.columns else df_plot.columns[0]
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                pd.plotting.andrews_curves(df_plot, target_column, ax=ax)
                ax.set_title('Andrews Curves')
                return fig, ax

            result = self.generate_plot(plot_andrews_curves)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_andrews_curves.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Andrews Curves", img_path, table_name)
            else:
                print("Skipping Andrews curves plot due to timeout.")
        else:
            print("Not enough numerical columns for Andrews curves.")

    def radar_charts(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Radar Charts"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 2:
            def plot_radar_chart():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size(), subplot_kw=dict(projection='polar'))
                values = df[numerical_columns].mean().values
                angles = np.linspace(0, 2*np.pi, len(numerical_columns), endpoint=False)
                values = np.concatenate((values, [values[0]]))
                angles = np.concatenate((angles, [angles[0]]))
                ax.plot(angles, values)
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(numerical_columns)
                ax.set_title('Radar Chart of Average Values')
                return fig, ax

            result = self.generate_plot(plot_radar_chart)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_radar_chart.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Radar Charts", img_path, table_name)
            else:
                print("Skipping radar chart plot due to timeout.")
        else:
            print("Not enough numerical columns for radar chart.")

    def sankey_diagrams(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Sankey Diagrams"))

        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) >= 2:
            def plot_sankey():
                source = df[categorical_columns[0]]
                target = df[categorical_columns[1]]
                value = df[df.columns[0]]  # Using the first column as value

                label = list(set(source) | set(target))
                color = plt.cm.Set3(np.linspace(0, 1, len(label)))

                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                
                sankey = pd.DataFrame({'source': source, 'target': target, 'value': value})
                G = nx.from_pandas_edgelist(sankey, 'source', 'target', 'value')
                pos = nx.spring_layout(G)
                
                nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=color)
                nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.5)
                nx.draw_networkx_labels(G, pos, font_size=10)
                
                ax.set_title('Sankey Diagram')
                ax.axis('off')
                return fig, ax

            result = self.generate_plot(plot_sankey)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_sankey_diagram.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Sankey Diagrams", img_path, table_name)
            else:
                print("Skipping Sankey diagram plot due to timeout.")
        else:
            print("Not enough categorical columns for Sankey diagram.")

    def bubble_charts(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Bubble Charts"))

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 3:
            def plot_bubble_chart():
                x = df[numerical_columns[0]]
                y = df[numerical_columns[1]]
                size = df[numerical_columns[2]]
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                scatter = ax.scatter(x, y, s=size, alpha=0.5)
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title('Bubble Chart')
                plt.colorbar(scatter)
                return fig, ax

            result = self.generate_plot(plot_bubble_chart)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_bubble_chart.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Bubble Charts", img_path, table_name)
            else:
                print("Skipping bubble chart plot due to timeout.")
        else:
            print("Not enough numerical columns for bubble chart.")

    def geographical_plots(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Geographical Plots"))

        if 'latitude' in df.columns and 'longitude' in df.columns:
            def plot_geographical():
                m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)
                for idx, row in df.iterrows():
                    folium.Marker([row['latitude'], row['longitude']]).add_to(m)
                img_path = os.path.join(self.output_folder, f"{table_name}_geographical_plot.html")
                m.save(img_path)
                return img_path

            result = self.generate_plot(plot_geographical)
            if result is not None:
                self.interpret_results("Geographical Plots", result, table_name)
            else:
                print("Skipping geographical plot due to timeout.")
        else:
            print("No latitude and longitude columns found for geographical plot.")

    def word_clouds(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Word Clouds"))

        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            def plot_word_cloud():
                text = " ".join(df[text_columns[0]].dropna())
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Word Cloud')
                return fig, ax

            result = self.generate_plot(plot_word_cloud)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_word_cloud.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Word Clouds", img_path, table_name)
            else:
                print("Skipping word cloud plot due to timeout.")
        else:
            print("No text columns found for word cloud.")

    def hierarchical_clustering_dendrogram(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hierarchical Clustering Dendrogram"))

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            def plot_dendrogram():
                # Select numerical columns and handle NaN values
                X = df[numerical_columns]
                
                # Use SimpleImputer to replace NaN values with the mean of the column
                imputer = SimpleImputer(strategy='mean')
                X_imputed = imputer.fit_transform(X)
                
                X_scaled = StandardScaler().fit_transform(X_imputed)
                
                model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
                model = model.fit(X_scaled)
                
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                
                def plot_dendrogram_recursive(model, ax):
                    counts = np.zeros(model.children_.shape[0])
                    n_samples = len(model.labels_)
                    for i, merge in enumerate(model.children_):
                        current_count = 0
                        for child_idx in merge:
                            if child_idx < n_samples:
                                current_count += 1
                            else:
                                current_count += counts[child_idx - n_samples]
                        counts[i] = current_count

                    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
                    ax.set_title('Hierarchical Clustering Dendrogram')
                    ax.set_xlabel('Number of points in node (or index of point if no parenthesis)')
                    ax.set_ylabel('Distance')
                    dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=3)
                
                plot_dendrogram_recursive(model, ax)
                ax.set_title('Hierarchical Clustering Dendrogram')
                ax.set_xlabel('Number of points in node (or index of point if no parenthesis)')
                ax.set_ylabel('Distance')
                return fig, ax

            result = self.generate_plot(plot_dendrogram)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_dendrogram.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Hierarchical Clustering Dendrogram", img_path, table_name)
            else:
                print("Skipping hierarchical clustering dendrogram plot due to timeout.")
        else:
            print("Not enough numerical columns for hierarchical clustering dendrogram.")

    def ecdf_plots(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - ECDF Plots"))

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_ecdf():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                for column in numerical_columns[:5]:  # Limit to 5 columns for readability
                    data = df[column].dropna()
                    x = np.sort(data)
                    y = np.arange(1, len(data) + 1) / len(data)
                    ax.step(x, y, label=column)
                ax.set_xlabel('Value')
                ax.set_ylabel('ECDF')
                ax.set_title('Empirical Cumulative Distribution Function')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_ecdf)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_ecdf_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("ECDF Plots", img_path, table_name)
            else:
                print("Skipping ECDF plot due to timeout.")
        else:
            print("No numerical columns found for ECDF plot.")



    def ridgeline_plots(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Ridgeline Plots"))

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numerical_columns) > 0 and len(categorical_columns) > 0:
            def plot_ridgeline():
                # Select the first numerical column and first categorical column
                numerical_col = numerical_columns[0]
                categorical_col = categorical_columns[0]
                
                # Ensure we have multiple categories
                if df[categorical_col].nunique() < 2:
                    print(f"Not enough categories in {categorical_col} for ridgeline plot.")
                    return None
                
                # Create the plot
                fig, axes = joypy.joyplot(
                    data=df,
                    by=categorical_col,
                    column=numerical_col,
                    colormap=plt.cm.viridis,
                    title=f"Ridgeline Plot of {numerical_col} by {categorical_col}",
                    labels=df[categorical_col].unique(),
                    figsize=self.calculate_figure_size()
                )
                
                # Adjust layout
                plt.tight_layout()
                return fig, axes

            result = self.generate_plot(plot_ridgeline)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_ridgeline_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Ridgeline Plots", img_path, table_name)
            else:
                print("Skipping ridgeline plot due to timeout or insufficient data.")
        else:
            print("Not enough numerical and categorical columns for ridgeline plot.")

    def hexbin_plots(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hexbin Plots"))

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) >= 2:
            def plot_hexbin():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                x = df[numerical_columns[0]]
                y = df[numerical_columns[1]]
                hb = ax.hexbin(x, y, gridsize=20, cmap='YlOrRd')
                ax.set_xlabel(numerical_columns[0])
                ax.set_ylabel(numerical_columns[1])
                ax.set_title('Hexbin Plot')
                plt.colorbar(hb)
                return fig, ax

            result = self.generate_plot(plot_hexbin)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_hexbin_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Hexbin Plots", img_path, table_name)
            else:
                print("Skipping hexbin plot due to timeout.")
        else:
            print("Not enough numerical columns for hexbin plot.")

    def mosaic_plots(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Mosaic Plots"))

        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) >= 2:
            def plot_mosaic():
                try:
                    # Select the first two categorical columns
                    cat_col1, cat_col2 = categorical_columns[:2]
                    
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    mosaic(df, [cat_col1, cat_col2], ax=ax, gap=0.05)
                    ax.set_title(f'Mosaic Plot of {cat_col1} vs {cat_col2}')
                    plt.tight_layout()
                    return fig, ax
                except Exception as e:
                    print(f"Error in creating mosaic plot: {str(e)}")
                    return None

            result = self.generate_plot(plot_mosaic)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_mosaic_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Mosaic Plots", img_path, table_name)
            else:
                print("Skipping mosaic plot due to timeout or error.")
        else:
            print("Not enough categorical columns for mosaic plot.")

    def lag_plots(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Lag Plots"))

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            def plot_lag():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                pd.plotting.lag_plot(df[numerical_columns[0]], lag=1, ax=ax)
                ax.set_title(f'Lag Plot for {numerical_columns[0]}')
                return fig, ax

            result = self.generate_plot(plot_lag)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_lag_plot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Lag Plots", img_path, table_name)
            else:
                print("Skipping lag plot due to timeout.")
        else:
            print("No numerical columns found for lag plot.")

    def shapley_value_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Shapley Value Analysis"))

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            def plot_shapley():
                try:
                    X = df[numerical_columns]
                    
                    # Use SimpleImputer to replace NaN values with the mean of each column
                    imputer = SimpleImputer(strategy='mean')
                    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                    
                    # Assuming the last column is the target variable
                    y = X_imputed.iloc[:, -1]
                    X = X_imputed.iloc[:, :-1]

                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)

                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)

                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                    ax.set_title('Shapley Value Analysis')
                    return fig, ax
                except Exception as e:
                    print(f"Error in creating Shapley value plot: {str(e)}")
                    return None

            result = self.generate_plot(plot_shapley)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_shapley_value_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Shapley Value Analysis", img_path, table_name)
            else:
                print("Skipping Shapley value analysis plot due to timeout or error.")
        else:
            print("Not enough numerical columns for Shapley value analysis.")

    def partial_dependence_plots(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Partial Dependence Plots"))
        
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_columns) > 1:
            # Choose the last column as the target variable
            target = numeric_columns[-1]
            features = numeric_columns[:-1]
            
            X = df[features]
            y = df[target]
            
            # Handle missing values in both X and y
            imputer_X = SimpleImputer(strategy='mean')
            imputer_y = SimpleImputer(strategy='mean')
            
            X_imputed = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns)
            y_imputed = pd.Series(imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel(), name=y.name)
            
            # Remove any remaining NaN values (if imputation wasn't possible for some reason)
            mask = ~np.isnan(y_imputed)
            X_imputed = X_imputed[mask]
            y_imputed = y_imputed[mask]
            
            if len(y_imputed) == 0:
                print("No valid data remaining after handling NaN values. Skipping Partial Dependence Plots.")
                return
            
            # Use HistGradientBoostingRegressor which can handle NaN values
            model = HistGradientBoostingRegressor(random_state=42)
            model.fit(X_imputed, y_imputed)
            
            def plot_pdp():
                fig, axes = plt.subplots(nrows=(len(features) + 1) // 2, ncols=2, figsize=(12, 4 * ((len(features) + 1) // 2)))
                axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
                
                for i, feature in enumerate(features):
                    try:
                        PartialDependenceDisplay.from_estimator(model, X_imputed, [feature], ax=axes[i])
                        axes[i].set_title(f'Partial Dependence of {target} on {feature}')
                    except Exception as e:
                        print(f"Error plotting partial dependence for feature '{feature}': {str(e)}")
                        axes[i].text(0.5, 0.5, f"Error plotting {feature}", ha='center', va='center')
                
                # Remove any unused subplots
                for j in range(i+1, len(axes)):
                    fig.delaxes(axes[j])
                
                plt.tight_layout()
                return fig, axes

            result = self.generate_plot(plot_pdp)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_partial_dependence_plots.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.interpret_results("Partial Dependence Plots", img_path, table_name)
            else:
                print("Skipping Partial Dependence Plots due to timeout.")
        else:
            print("Not enough numeric columns for Partial Dependence Plots.")

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

        2. SQLite Statements:
        [Provide the SQLite statements that would be used to perform this analysis]

        3. Positive Findings:
        [List any positive findings, or state "No significant positive findings" if none]

        4. Negative Findings:
        [List any negative findings, or state "No significant negative findings" if none]

        5. Conclusion:
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
        2. Ensuring the structure (Analysis, SQLite Statements, Positive Findings, Negative Findings, Conclusion) is maintained.
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
            if line.startswith("3. Positive Findings:") or line.startswith("4. Negative Findings:"):
                for finding in lines[i+1:]:
                    if finding.strip() and not finding.startswith(("3.", "4.", "5.")):
                        self.findings.append(f"{analysis_type}: {finding.strip()}")
                    elif finding.startswith(("3.", "4.", "5.")):
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
        output_file = os.path.join(self.output_folder, "axda_b2_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 2) Report for {self.table_name}"
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.executive_summary,
            self.findings,
            self.pdf_content,
            self.image_data,
            filename=f"axda_b2_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
        else:
            print(error("Failed to generate PDF report"))
