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
from src.helper_da import get_technique_info

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
            def plot_word_cloud_and_pie():
                text = " ".join(df[text_columns[0]].dropna())
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Word Cloud
                ax1.imshow(wordcloud, interpolation='bilinear')
                ax1.axis('off')
                ax1.set_title('Word Cloud')
                
                # Pie Chart of top 10 words
                word_freq = wordcloud.words_
                top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
                ax2.pie(top_words.values(), labels=top_words.keys(), autopct='%1.1f%%')
                ax2.set_title('Top 10 Most Frequent Words')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_word_cloud_and_pie)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_word_cloud_and_pie.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print("Skipping word cloud and pie chart plot due to timeout.")
        else:
            print("No text columns found for word cloud and pie chart.")
        self.interpret_results("Word Clouds and Frequency Pie Chart", {'image_paths': image_paths}, table_name)

    def hierarchical_clustering_dendrogram(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hierarchical Clustering Dendrogram"))
        image_paths = []

        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            def plot_dendrogram_and_pie():
                X = df[numerical_columns]
                imputer = SimpleImputer(strategy='mean')
                X_imputed = imputer.fit_transform(X)
                X_scaled = StandardScaler().fit_transform(X_imputed)
                
                model = AgglomerativeClustering(n_clusters=5)  # Set a fixed number of clusters
                model = model.fit(X_scaled)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Dendrogram
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
                
                plot_dendrogram_recursive(model, ax1)
                ax1.set_title('Hierarchical Clustering Dendrogram')
                ax1.set_xlabel('Number of points in node')
                ax1.set_ylabel('Distance')
                
                # Pie chart of cluster distribution
                cluster_counts = pd.Series(model.labels_).value_counts()
                ax2.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%')
                ax2.set_title('Cluster Distribution')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_dendrogram_and_pie)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_dendrogram_and_pie.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print("Skipping hierarchical clustering dendrogram and pie chart plot due to timeout.")
        else:
            print("Not enough numerical columns for hierarchical clustering dendrogram and pie chart.")
        self.interpret_results("Hierarchical Clustering Dendrogram and Cluster Distribution", {'image_paths': image_paths}, table_name)


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
            def plot_shapley_and_pie():
                X = df[numerical_columns]
                imputer = SimpleImputer(strategy='mean')
                X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                
                y = X_imputed.iloc[:, -1]
                X = X_imputed.iloc[:, :-1]

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Shapley summary plot
                shap.summary_plot(shap_values, X, plot_type="bar", show=False, ax=ax1)
                ax1.set_title('Shapley Value Analysis')
                
                # Pie chart of feature importance
                feature_importance = np.abs(shap_values).mean(0)
                ax2.pie(feature_importance, labels=X.columns, autopct='%1.1f%%')
                ax2.set_title('Feature Importance (Shapley Values)')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_shapley_and_pie)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_shapley_and_pie.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print("Skipping Shapley value analysis and pie chart plot due to timeout or error.")
        else:
            print("Not enough numerical columns for Shapley value analysis and pie chart.")
        self.interpret_results("Shapley Value Analysis and Feature Importance Pie Chart", {'image_paths': image_paths}, table_name)

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
            self.findings,
            self.pdf_content,
            formatted_image_data,  # Use the formatted image data
            filename=f"axda_b2_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None
