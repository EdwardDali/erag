import os
import time
import sqlite3
import threading
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import folium
import joypy
import shap

from scipy import stats
from scipy.stats import norm, anderson, pearsonr, probplot
from scipy.cluster.hierarchy import dendrogram

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.impute import SimpleImputer

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.mosaicplot import mosaic

from wordcloud import WordCloud

from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

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
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch2) on {self.db_path}"))
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch2) completed. Results saved in {self.output_folder}"))

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
            try:
                method(df, table_name)
            except Exception as e:
                error_message = f"An error occurred during {method.__name__}: {str(e)}"
                print(error(error_message))
                self.text_output += f"\n{error_message}\n"
                # Optionally, you can add this error to the PDF report as well
                self.pdf_content.append((method.__name__, [], error_message))
            finally:
                # Ensure we always increment the technique counter, even if the method fails
                self.technique_counter += 1

    def parallel_coordinates_plot(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Parallel Coordinates Plot"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            def plot_parallel_coordinates():
                try:
                    # Limit the number of columns and rows to prevent extremely large plots
                    columns_to_plot = numerical_columns[:10]  # Plot at most 10 columns
                    df_plot = df[columns_to_plot].head(1000)  # Limit to 1000 rows
                    
                    # If there's no 'target' column, use the first column as a proxy
                    target_column = 'target' if 'target' in df_plot.columns else df_plot.columns[0]
                    
                    # Calculate figure size based on the number of columns
                    width, height = self.calculate_figure_size()
                    fig, ax = plt.subplots(figsize=(width * len(columns_to_plot) / 10, height))
                    
                    pd.plotting.parallel_coordinates(df_plot, target_column, ax=ax)
                    ax.set_title('Parallel Coordinates Plot (Sample)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    return fig, ax
                except Exception as e:
                    print(f"Error in creating parallel coordinates plot: {str(e)}")
                    return None

            result = self.generate_plot(plot_parallel_coordinates)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_parallel_coordinates.png")
                try:
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Error saving parallel coordinates plot: {str(e)}")
                    print("Skipping parallel coordinates plot due to error.")
            else:
                print("Skipping parallel coordinates plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for parallel coordinates plot.")

        self.interpret_results("Parallel Coordinates Plot", {'image_paths': image_paths}, table_name)


    def andrews_curves(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Andrews Curves"))
        image_paths = []
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            def plot_andrews_curves():
                try:
                    # Limit the number of columns and rows to prevent extremely large plots
                    columns_to_plot = numerical_columns[:10]  # Plot at most 10 columns
                    df_plot = df[columns_to_plot].head(1000)  # Limit to 1000 rows
                    
                    # If there's no 'target' column, use the first column as a proxy
                    target_column = 'target' if 'target' in df_plot.columns else df_plot.columns[0]
                    
                    # Calculate figure size
                    width, height = self.calculate_figure_size()
                    fig, ax = plt.subplots(figsize=(width, height))
                    
                    pd.plotting.andrews_curves(df_plot, target_column, ax=ax)
                    ax.set_title('Andrews Curves (Sample)')
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.tight_layout()
                    return fig, ax
                except Exception as e:
                    print(f"Error in creating Andrews curves plot: {str(e)}")
                    return None

            result = self.generate_plot(plot_andrews_curves)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_andrews_curves.png")
                try:
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                    
                except Exception as e:
                    print(f"Error saving Andrews curves plot: {str(e)}")
                    print("Skipping Andrews curves plot due to error.")
            else:
                print("Skipping Andrews curves plot due to error in plot generation.")
        else:
            print("Not enough numerical columns for Andrews curves.")
        self.interpret_results("Andrews Curves", {'image_paths': image_paths}, table_name)

    def radar_charts(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Radar Charts"))
        image_paths = []
        
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
                image_paths.append(img_path)
                
            else:
                print("Skipping radar chart plot due to timeout.")
        else:
            print("Not enough numerical columns for radar chart.")
        self.interpret_results("Radar Charts", {'image_paths': image_paths}, table_name)

    def sankey_diagrams(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Sankey Diagrams"))
        image_paths = []

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
                image_paths.append(img_path)
                
            else:
                print("Skipping Sankey diagram plot due to timeout.")
        else:
            print("Not enough categorical columns for Sankey diagram.")
        self.interpret_results("Sankey Diagrams", {'image_paths': image_paths}, table_name)

    def bubble_charts(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Bubble Charts"))
        image_paths = []

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
                image_paths.append(img_path)
                
            else:
                print("Skipping bubble chart plot due to timeout.")
        else:
            print("Not enough numerical columns for bubble chart.")
        self.interpret_results("Bubble Charts", {'image_paths': image_paths}, table_name)

    def geographical_plots(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Geographical Plots"))
        image_paths = []

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
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Word Clouds"))
        image_paths = []

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
                image_paths.append(img_path)
                
            else:
                print("Skipping word cloud plot due to timeout.")
        else:
            print("No text columns found for word cloud.")
        self.interpret_results("Word Clouds", {'image_paths': image_paths}, table_name)

    def hierarchical_clustering_dendrogram(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hierarchical Clustering Dendrogram"))
        image_paths = []

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
                image_paths.append(img_path)
                
            else:
                print("Skipping hierarchical clustering dendrogram plot due to timeout.")
        else:
            print("Not enough numerical columns for hierarchical clustering dendrogram.")
        self.interpret_results("Hierarchical Clustering Dendrogram", {'image_paths': image_paths}, table_name)

    def ecdf_plots(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - ECDF Plots"))
        image_paths = []

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
                image_paths.append(img_path)
                
            else:
                print("Skipping ECDF plot due to timeout.")
        else:
            print("No numerical columns found for ECDF plot.")
        self.interpret_results("ECDF Plots", {'image_paths': image_paths}, table_name)



    def ridgeline_plots(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Ridgeline Plots"))
        image_paths = []

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
                image_paths.append(img_path)
                
            else:
                print("Skipping ridgeline plot due to timeout or insufficient data.")
        else:
            print("Not enough numerical and categorical columns for ridgeline plot.")
        self.interpret_results("Ridgeline Plots", {'image_paths': image_paths}, table_name)

    def hexbin_plots(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hexbin Plots"))
        image_paths = []

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
                image_paths.append(img_path)
                
            else:
                print("Skipping hexbin plot due to timeout.")
        else:
            print("Not enough numerical columns for hexbin plot.")
        self.interpret_results("Hexbin Plots", {'image_paths': image_paths}, table_name)

    def mosaic_plots(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Mosaic Plots"))
        image_paths = []

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
                image_paths.append(img_path)
                
            else:
                print("Skipping mosaic plot due to timeout or error.")
        else:
            print("Not enough categorical columns for mosaic plot.")
        self.interpret_results("Mosaic Plots", {'image_paths': image_paths}, table_name)

    def lag_plots(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Lag Plots"))
        image_paths = []

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
                image_paths.append(img_path)
            else:
                print("Skipping lag plot due to timeout.")
        else:
            print("No numerical columns found for lag plot.")
        self.interpret_results("Lag Plots", {'image_paths': image_paths}, table_name)


    def shapley_value_analysis(self, df, table_name):
        
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Shapley Value Analysis"))
        image_paths = []

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
                image_paths.append(img_path)
                
            else:
                print("Skipping Shapley value analysis plot due to timeout or error.")
        else:
            print("Not enough numerical columns for Shapley value analysis.")
        self.interpret_results("Shapley Value Analysis", {'image_paths': image_paths}, table_name)

    def partial_dependence_plots(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Partial Dependence Plots"))
        image_paths = []
        
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
            
            for feature in features:
                def plot_pdp():
                    fig, ax = plt.subplots(figsize=(8, 6))
                    try:
                        PartialDependenceDisplay.from_estimator(model, X_imputed, [feature], ax=ax)
                        ax.set_title(f'Partial Dependence of {target} on {feature}')
                    except Exception as e:
                        print(f"Error plotting partial dependence for feature '{feature}': {str(e)}")
                        ax.text(0.5, 0.5, f"Error plotting {feature}", ha='center', va='center')
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_pdp)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_partial_dependence_plot_{feature}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)
                else:
                    print(f"Skipping Partial Dependence Plot for {feature} due to timeout.")
        else:
            print("Not enough numeric columns for Partial Dependence Plots.")
        
        self.interpret_results("Partial Dependence Plots", {'image_paths': image_paths}, table_name)


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
        output_file = os.path.join(self.output_folder, "axda_b2_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 2) Report for {self.table_name}"
        
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
            filename=f"axda_b1_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None
