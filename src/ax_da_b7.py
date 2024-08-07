import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.signal import periodogram
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
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
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.tsa.stattools
import warnings
import concurrent.futures
from datetime import datetime, timedelta

class TimeoutException(Exception):
    pass

class AdvancedExploratoryDataAnalysisB7:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 16
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
        self.settings = settings

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
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch 7) on {self.db_path}"))
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch 7) completed. Results saved in {self.output_folder}"))

    def preprocess_date_column(self, df):
        # Check if 'Date' column exists
        if 'Date' not in df.columns:
            raise ValueError("No 'Date' column found in the dataset. Analysis cannot proceed.")

        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Check if any conversion failed
        if df['Date'].isnull().any():
            print(warning("Some dates could not be converted to datetime format. These will be treated as missing values."))

        # Remove rows with null dates
        df = df.dropna(subset=['Date'])

        # Sort the dataframe by date
        df = df.sort_values('Date')

        # Check for duplicate dates
        if df['Date'].duplicated().any():
            print(warning("Duplicate dates found. Aggregating data by date."))
            # Group by date and aggregate
            # For numeric columns, take the mean
            # For non-numeric columns, take the first value
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            agg_dict = {col: 'mean' if col in numeric_columns else 'first' for col in df.columns if col != 'Date'}
            df = df.groupby('Date').agg(agg_dict).reset_index()

        # Identify the most common frequency
        freq = pd.infer_freq(df['Date'])
        if freq is None:
            # If frequency can't be inferred, assume daily
            freq = 'D'
            print(warning("Date frequency could not be inferred. Assuming daily frequency."))

        # Create a complete date range
        date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq=freq)

        # Reindex the dataframe with the complete date range
        df = df.set_index('Date').reindex(date_range).reset_index()
        df = df.rename(columns={'index': 'Date'})

        print(info(f"Date column preprocessed. Data frequency: {freq}"))
        return df

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_b7_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            
        if df.empty:
            error_message = f"The table {table_name} is empty."
            print(error(error_message))
            self.text_output += f"\n{error_message}\n"
            return

        # Preprocess the date column
        try:
            df = self.preprocess_date_column(df)
        except ValueError as e:
            print(error(str(e)))
            return

        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values using interpolation and forward/backward fill
        df = df.interpolate().fillna(method='bfill').fillna(method='ffill')
        
        analysis_methods = [
            self.arima_analysis,
            self.auto_arimax_analysis,
            self.exponential_smoothing,
            self.moving_average,
            self.linear_regression_trend,
            self.seasonal_decomposition_analysis,
            self.holt_winters_method,
            self.sarimax_analysis,
            self.gradient_boosting_time_series,
            self.lstm_time_series,
            self.fourier_analysis,
            self.trend_extraction,
            self.cross_sectional_regression,
            self.ensemble_time_series,
            self.bootstrapping_time_series,
            self.theta_method
        ]

        for method in analysis_methods:
            try:
                method(df.copy(), table_name)
            except Exception as e:
                error_message = f"An error occurred during {method.__name__}: {str(e)}"
                print(error(error_message))
                self.text_output += f"\n{error_message}\n"
                self.pdf_content.append((method.__name__, [], error_message))
            finally:
                self.technique_counter += 1

    @staticmethod
    def model_fit_with_timeout(model, timeout):
        def fit_func():
            return model.fit()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fit_func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                return None

    def arima_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - ARIMA Analysis"))
        image_paths = []
        arima_results = {}

        # Ensure 'Date' column is datetime and set as index
        date_col = df.select_dtypes(include=['datetime64']).columns
        if len(date_col) > 0:
            df.set_index(date_col[0], inplace=True)
        else:
            # If no datetime column, create a date range index
            df.index = pd.date_range(start='1/1/2000', periods=len(df))

        # Ensure the date index has a frequency
        if df.index.freq is None:
            df = df.asfreq(pd.infer_freq(df.index))

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().all():
                continue

            # Handle missing values
            df[col] = df[col].interpolate().bfill().ffill()

            try:
                # Determine optimal ARIMA parameters
                p, d, q = self.determine_arima_parameters(df[col])

                # Fit the ARIMA model with a timeout
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    
                    model = ARIMA(df[col], order=(p, d, q))
                    results = self.model_fit_with_timeout(model, timeout=30)  # 30 seconds timeout

                    if results is None:
                        raise TimeoutError("ARIMA model fitting timed out")

                    # Check for convergence warning
                    if any("convergence" in str(warn.message).lower() for warn in w):
                        print(warning(f"Warning: ARIMA model for {col} did not converge. Trying alternative parameters."))
                        # Try alternative parameters
                        for alternative_order in [(1,1,1), (1,1,0), (0,1,1)]:
                            model = ARIMA(df[col], order=alternative_order)
                            results = self.model_fit_with_timeout(model, timeout=30)
                            if results is not None and not any("convergence" in str(warn.message).lower() for warn in w):
                                print(info(f"Alternative ARIMA parameters {alternative_order} converged for {col}"))
                                break
                        else:
                            raise ValueError("Could not find converging ARIMA parameters")

                def plot_arima():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Observed')
                    ax.plot(df.index, results.fittedvalues, color='red', label='ARIMA Fit')
                    ax.set_title(f'ARIMA Analysis: {col} (Order: {p},{d},{q})')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_arima)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_arima_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                # Perform forecasting
                forecast_steps = min(30, int(len(df) * 0.1))  # Forecast 10% of data length or 30 steps, whichever is smaller
                forecast = results.forecast(steps=forecast_steps)
                
                # Calculate error metrics
                mse = mean_squared_error(df[col][-forecast_steps:], results.fittedvalues[-forecast_steps:])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(df[col][-forecast_steps:], results.fittedvalues[-forecast_steps:])
                
                arima_results[col] = {
                    'aic': results.aic,
                    'bic': results.bic,
                    'order': (p, d, q),
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'forecast': forecast.tolist()
                }

            except Exception as e:
                print(error(f"Error in ARIMA analysis for column {col}: {str(e)}"))
                arima_results[col] = {'error': str(e)}

        self.interpret_results("ARIMA Analysis", {
            'image_paths': image_paths,
            'arima_results': arima_results
        }, table_name)

    def determine_arima_parameters(self, series):
        # Determine 'd' (differencing term)
        d = 0
        while d < 2 and not self.is_stationary(series):
            series = series.diff().dropna()
            d += 1

        # Use auto_arima to determine optimal p and q
        model = auto_arima(series, d=d, start_p=0, start_q=0, max_p=5, max_q=5, 
                           seasonal=False, stepwise=True, suppress_warnings=True, 
                           error_action="ignore", max_order=None, trace=False)
        
        return model.order[0], d, model.order[2]

    def is_stationary(self, series):
        return statsmodels.tsa.stattools.adfuller(series, autolag='AIC')[1] <= 0.05
           

    def auto_arimax_analysis(self, df, target_column, feature_columns):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Auto ARIMAX Analysis"))
        
        # Split data into train and test
        train_size = int(len(df) * 0.8)
        train, test = df[:train_size], df[train_size:]

        # Fit Auto ARIMA model
        model = auto_arima(train[target_column], exogenous=train[feature_columns], trace=True, error_action="ignore", suppress_warnings=True)
        model.fit(train[target_column], exogenous=train[feature_columns])

        # Make predictions
        forecast = model.predict(n_periods=len(test), exogenous=test[feature_columns])
        test['Forecast_ARIMAX'] = forecast

        # Plot results
        img_path = self.plot_forecast(test, target_column, 'Forecast_ARIMAX', 'Auto ARIMAX Forecast')

        # Calculate metrics
        rmse, mae = self.calculate_metrics(test[target_column], test['Forecast_ARIMAX'])

        results = {
            'image_paths': [img_path],
            'rmse': rmse,
            'mae': mae,
            'best_order': model.order,
            'best_seasonal_order': model.seasonal_order
        }

        self.interpret_results("Auto ARIMAX Analysis", results, self.table_name)
        
    def exponential_smoothing(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Exponential Smoothing"))
        image_paths = []
        exp_smoothing_results = {}

        date_col = df.select_dtypes(include=['datetime64']).columns
        if len(date_col) > 0:
            df.set_index(date_col[0], inplace=True)
        else:
            df.index = pd.date_range(start='1/1/2000', periods=len(df))

        if df.index.freq is None:
            df = df.asfreq(pd.infer_freq(df.index))

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().all():
                continue

            df[col] = df[col].interpolate().bfill().ffill()

            try:
                model = ExponentialSmoothing(df[col])
                results = self.model_fit_with_timeout(model, timeout=30)

                if results is None:
                    raise TimeoutError("Exponential Smoothing model fitting timed out")

                def plot_exp_smoothing():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Observed')
                    ax.plot(df.index, results.fittedvalues, color='red', label='Exponential Smoothing')
                    ax.set_title(f'Exponential Smoothing: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_exp_smoothing)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_exp_smoothing_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                forecast_steps = min(30, int(len(df) * 0.1))
                forecast = results.forecast(steps=forecast_steps)
                
                exp_smoothing_results[col] = {
                    'aic': results.aic,
                    'bic': results.bic,
                    'mse': mean_squared_error(df[col][-forecast_steps:], results.fittedvalues[-forecast_steps:]),
                    'forecast': forecast.tolist()
                }

            except Exception as e:
                print(error(f"Error in Exponential Smoothing analysis for column {col}: {str(e)}"))
                exp_smoothing_results[col] = {'error': str(e)}

        self.interpret_results("Exponential Smoothing", {
            'image_paths': image_paths,
            'exp_smoothing_results': exp_smoothing_results
        }, table_name)




    def moving_average(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Moving Average"))
        image_paths = []
        moving_average_results = {}

        try:
            date_col = df.select_dtypes(include=['datetime64']).columns
            if len(date_col) > 0:
                df.set_index(date_col[0], inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))

            if df.index.freq is None:
                df = df.asfreq(pd.infer_freq(df.index))

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                # Calculate simple moving average
                window_size = min(30, len(df) // 2)  # Use 30 days or half the data length, whichever is smaller
                ma = df[col].rolling(window=window_size).mean()

                def plot_moving_average():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Original')
                    ax.plot(df.index, ma, color='red', label=f'{window_size}-day Moving Average')
                    ax.set_title(f'Moving Average Analysis: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_moving_average)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_moving_average_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                moving_average_results[col] = {
                    'window_size': window_size,
                    'last_ma_value': ma.iloc[-1] if not ma.empty else None
                }

            if not moving_average_results:
                raise ValueError("No valid data for moving average analysis")

        except Exception as e:
            print(error(f"Error in Moving Average analysis: {str(e)}"))
            moving_average_results = {'error': str(e)}

        self.interpret_results("Moving Average Analysis", {
            'image_paths': image_paths,
            'moving_average_results': moving_average_results
        }, table_name)

    def linear_regression_trend(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Linear Regression Trend"))
        image_paths = []
        linear_trend_results = {}

        try:
            date_col = df.select_dtypes(include=['datetime64']).columns
            if len(date_col) > 0:
                df.set_index(date_col[0], inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))

            if df.index.freq is None:
                df = df.asfreq(pd.infer_freq(df.index))

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                X = np.arange(len(df)).reshape(-1, 1)
                y = df[col].values

                model = LinearRegression()
                model.fit(X, y)

                trend = model.predict(X)

                def plot_linear_trend():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, y, label='Original')
                    ax.plot(df.index, trend, color='red', label='Linear Trend')
                    ax.set_title(f'Linear Regression Trend: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_linear_trend)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_linear_trend_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                linear_trend_results[col] = {
                    'slope': model.coef_[0],
                    'intercept': model.intercept_,
                    'r_squared': model.score(X, y)
                }

            if not linear_trend_results:
                raise ValueError("No valid data for linear regression trend analysis")

        except Exception as e:
            print(error(f"Error in Linear Regression Trend analysis: {str(e)}"))
            linear_trend_results = {'error': str(e)}

        self.interpret_results("Linear Regression Trend", {
            'image_paths': image_paths,
            'linear_trend_results': linear_trend_results
        }, table_name)

    def seasonal_decomposition_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Seasonal Decomposition"))
        image_paths = []
        seasonal_decomposition_results = {}

        try:
            date_col = df.select_dtypes(include=['datetime64']).columns
            if len(date_col) > 0:
                df.set_index(date_col[0], inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))

            if df.index.freq is None:
                df = df.asfreq(pd.infer_freq(df.index))

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                # Ensure we have enough data for seasonal decomposition
                if len(df) < 2:
                    raise ValueError(f"Not enough data points for seasonal decomposition in column {col}")

                # Determine the period for seasonal decomposition
                if df.index.freq == 'D':
                    period = 7  # Weekly seasonality for daily data
                elif df.index.freq in ['M', 'MS']:
                    period = 12  # Yearly seasonality for monthly data
                else:
                    period = 1  # Default to no seasonality if frequency can't be determined

                result = seasonal_decompose(df[col], model='additive', period=period)

                def plot_seasonal_decomposition():
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

                result_plot = self.generate_plot(plot_seasonal_decomposition)
                if result_plot is not None:
                    fig, _ = result_plot
                    img_path = os.path.join(self.output_folder, f"{table_name}_seasonal_decomposition_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                seasonal_decomposition_results[col] = {
                    'trend_strength': np.var(result.trend) / np.var(result.observed),
                    'seasonality_strength': np.var(result.seasonal) / np.var(result.observed),
                    'residual_strength': np.var(result.resid) / np.var(result.observed)
                }

            if not seasonal_decomposition_results:
                raise ValueError("No valid data for seasonal decomposition analysis")

        except Exception as e:
            print(error(f"Error in Seasonal Decomposition analysis: {str(e)}"))
            seasonal_decomposition_results = {'error': str(e)}

        self.interpret_results("Seasonal Decomposition", {
            'image_paths': image_paths,
            'seasonal_decomposition_results': seasonal_decomposition_results
        }, table_name)

    def holt_winters_method(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Holt-Winters Method"))
        image_paths = []
        holt_winters_results = {}
        
        date_col = df.select_dtypes(include=['datetime64']).columns
        if len(date_col) > 0:
            df.set_index(date_col[0], inplace=True)
        else:
            df.index = pd.date_range(start='1/1/2000', periods=len(df))

        if df.index.freq is None:
            df = df.asfreq(pd.infer_freq(df.index))

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:
            if df[col].isna().all():
                continue

            df[col] = df[col].interpolate().bfill().ffill()

            try:
                model = ExponentialSmoothing(df[col], seasonal_periods=12, trend='add', seasonal='add')
                results = self.model_fit_with_timeout(model, timeout=30)

                if results is None:
                    raise TimeoutError("Holt-Winters model fitting timed out")

                def plot_holt_winters():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Observed')
                    ax.plot(df.index, results.fittedvalues, color='red', label='Holt-Winters')
                    ax.set_title(f'Holt-Winters Method: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_holt_winters)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_holt_winters_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                forecast_steps = min(30, int(len(df) * 0.1))
                forecast = results.forecast(steps=forecast_steps)
                
                holt_winters_results[col] = {
                    'aic': results.aic,
                    'bic': results.bic,
                    'mse': mean_squared_error(df[col][-forecast_steps:], results.fittedvalues[-forecast_steps:]),
                    'forecast': forecast.tolist()
                }

            except Exception as e:
                print(error(f"Error in Holt-Winters analysis for column {col}: {str(e)}"))
                holt_winters_results[col] = {'error': str(e)}

        self.interpret_results("Holt-Winters Method", {
            'image_paths': image_paths,
            'holt_winters_results': holt_winters_results
        }, table_name)

    def gradient_boosting_time_series(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Gradient Boosting for Time Series"))
        image_paths = []
        gb_results = {}

        try:
            date_col = df.select_dtypes(include=['datetime64']).columns
            if len(date_col) > 0:
                df.set_index(date_col[0], inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))

            if df.index.freq is None:
                df = df.asfreq(pd.infer_freq(df.index))

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                series = df[col]
                X = np.arange(len(series)).reshape(-1, 1)
                y = series.values

                if len(y) < 2:
                    raise ValueError(f"Not enough data points for Gradient Boosting in column {col}")

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = GradientBoostingRegressor(random_state=42)
                model.fit(X_train, y_train)

                predictions = model.predict(X)

                def plot_gradient_boosting():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, y, label='Observed')
                    ax.plot(df.index, predictions, color='red', label='Gradient Boosting')
                    ax.set_title(f'Gradient Boosting for Time Series: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_gradient_boosting)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_gradient_boosting_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                gb_results[col] = {
                    'mse': mean_squared_error(y_test, model.predict(X_test)),
                    'feature_importance': model.feature_importances_.tolist()
                }

            if not gb_results:
                raise ValueError("No valid data for Gradient Boosting analysis")

        except Exception as e:
            print(error(f"Error in Gradient Boosting analysis: {str(e)}"))
            gb_results = {'error': str(e)}

        self.interpret_results("Gradient Boosting for Time Series", {
            'image_paths': image_paths,
            'gb_results': gb_results
        }, table_name)

    def lstm_time_series(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - LSTM for Time Series"))
        image_paths = []
        lstm_results = {}

        try:
            date_col = df.select_dtypes(include=['datetime64']).columns
            if len(date_col) > 0:
                df.set_index(date_col[0], inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))

            if df.index.freq is None:
                df = df.asfreq(pd.infer_freq(df.index))

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                series = df[col].values
                if len(series) < 10:  # Ensure we have enough data points for LSTM
                    raise ValueError(f"Not enough data points for LSTM analysis in column {col}")

                series = (series - np.min(series)) / (np.max(series) - np.min(series))  # Normalize

                X = []
                y = []
                for i in range(len(series) - 5):
                    X.append(series[i:i+5])
                    y.append(series[i+5])
                X, y = np.array(X), np.array(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = Sequential([
                    LSTM(50, activation='relu', input_shape=(5, 1)),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')

                model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

                predictions = model.predict(X)

                def plot_lstm():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index[5:], y, label='Observed')
                    ax.plot(df.index[5:], predictions, color='red', label='LSTM')
                    ax.set_title(f'LSTM for Time Series: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_lstm)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_lstm_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                lstm_results[col] = {
                    'mse': mean_squared_error(y_test, model.predict(X_test)),
                    'final_loss': model.history.history['loss'][-1]
                }

            if not lstm_results:
                raise ValueError("No valid data for LSTM analysis")

        except Exception as e:
            print(error(f"Error in LSTM analysis: {str(e)}"))
            lstm_results = {'error': str(e)}

        self.interpret_results("LSTM for Time Series", {
            'image_paths': image_paths,
            'lstm_results': lstm_results
        }, table_name)

    def fourier_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Fourier Analysis"))
        image_paths = []
        fourier_results = {}

        try:
            date_col = df.select_dtypes(include=['datetime64']).columns
            if len(date_col) > 0:
                df.set_index(date_col[0], inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))

            if df.index.freq is None:
                df = df.asfreq(pd.infer_freq(df.index))

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                series = df[col].values
                if len(series) < 2:  # Ensure we have enough data points for Fourier analysis
                    raise ValueError(f"Not enough data points for Fourier analysis in column {col}")

                f, Pxx_den = periodogram(series)

                def plot_fourier():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.semilogy(f, Pxx_den)
                    ax.set_title(f'Fourier Analysis: {col}')
                    ax.set_xlabel('Frequency')
                    ax.set_ylabel('Power Spectral Density')
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_fourier)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_fourier_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                fourier_results[col] = {
                    'dominant_frequency': f[np.argmax(Pxx_den)],
                    'max_power': np.max(Pxx_den)
                }

            if not fourier_results:
                raise ValueError("No valid data for Fourier analysis")

        except Exception as e:
            print(error(f"Error in Fourier analysis: {str(e)}"))
            fourier_results = {'error': str(e)}

        self.interpret_results("Fourier Analysis", {
            'image_paths': image_paths,
            'fourier_results': fourier_results
        }, table_name)

    def trend_extraction(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Trend Extraction"))
        image_paths = []
        trend_results = {}

        try:
            date_col = df.select_dtypes(include=['datetime64']).columns
            if len(date_col) > 0:
                df.set_index(date_col[0], inplace=True)
            else:
                df.index = pd.date_range(start='1/1/2000', periods=len(df))

            if df.index.freq is None:
                df = df.asfreq(pd.infer_freq(df.index))

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the dataset")

            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[col].isna().all():
                    continue

                df[col] = df[col].interpolate().bfill().ffill()

                series = df[col].values
                if len(series) < 2:  # Ensure we have enough data points for trend extraction
                    raise ValueError(f"Not enough data points for trend extraction in column {col}")

                cycle, trend = hpfilter(series, lamb=1600)

                def plot_trend():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, series, label='Original')
                    ax.plot(df.index, trend, color='red', label='Trend')
                    ax.set_title(f'Trend Extraction: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_trend)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_trend_extraction_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                trend_results[col] = {
                    'trend_strength': np.var(trend) / np.var(series),
                    'cycle_strength': np.var(cycle) / np.var(series)
                }

            if not trend_results:
                raise ValueError("No valid data for trend extraction analysis")

        except Exception as e:
            print(error(f"Error in Trend Extraction analysis: {str(e)}"))
            trend_results = {'error': str(e)}

        self.interpret_results("Trend Extraction", {
            'image_paths': image_paths,
            'trend_results': trend_results
        }, table_name)

    def cross_sectional_regression(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Cross-Sectional Regression"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print(warning("Not enough numeric columns for cross-sectional regression."))
            return
        
        X = df[numeric_cols[:-1]]
        y = df[numeric_cols[-1]]
        
        model = LinearRegression()
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        def plot_cross_sectional():
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ax.scatter(y, predictions)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax.set_title('Cross-Sectional Regression')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            plt.tight_layout()
            return fig, ax

        result = self.generate_plot(plot_cross_sectional)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_cross_sectional_regression.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
        
        cross_sectional_results = {
            'r_squared': model.score(X, y),
            'coefficients': dict(zip(X.columns, model.coef_))
        }
        
        self.interpret_results("Cross-Sectional Regression", {
            'image_paths': image_paths,
            'cross_sectional_results': cross_sectional_results
        }, table_name)

    def ensemble_time_series(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Ensemble Time Series"))
        image_paths = []
        
        time_col = df.select_dtypes(include=['datetime64']).columns[0]
        df.set_index(time_col, inplace=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            series = df[col].values
            
            # Simple ensemble of ARIMA, ExponentialSmoothing, and SARIMAX
            arima_model = ARIMA(series, order=(1,1,1))
            arima_results = arima_model.fit()
            
            exp_model = ExponentialSmoothing(series)
            exp_results = exp_model.fit()
            
            sarimax_model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
            sarimax_results = sarimax_model.fit()
            
            ensemble_forecast = (arima_results.forecast(5) + exp_results.forecast(5) + sarimax_results.forecast(5)) / 3
            
            def plot_ensemble():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(df.index, series, label='Original')
                ax.plot(pd.date_range(start=df.index[-1], periods=6)[1:], ensemble_forecast, color='red', label='Ensemble Forecast')
                ax.set_title(f'Ensemble Time Series: {col}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.legend()
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_ensemble)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_ensemble_{col}.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
        
        ensemble_results = {
            'forecast': ensemble_forecast.tolist()
        }
        
        self.interpret_results("Ensemble Time Series", {
            'image_paths': image_paths,
            'ensemble_results': ensemble_results
        }, table_name)

    def bootstrapping_time_series(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Bootstrapping Time Series"))
        image_paths = []
        
        time_col = df.select_dtypes(include=['datetime64']).columns[0]
        df.set_index(time_col, inplace=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            series = df[col].values
            
            # Simple block bootstrapping
            block_size = 30
            n_bootstraps = 100
            bootstrapped_series = []
            
            for _ in range(n_bootstraps):
                bootstrapped = []
                for _ in range(0, len(series), block_size):
                    start = np.random.randint(0, len(series) - block_size)
                    bootstrapped.extend(series[start:start+block_size])
                bootstrapped_series.append(bootstrapped[:len(series)])
            
            bootstrapped_mean = np.mean(bootstrapped_series, axis=0)
            bootstrapped_lower = np.percentile(bootstrapped_series, 2.5, axis=0)
            bootstrapped_upper = np.percentile(bootstrapped_series, 97.5, axis=0)
            
            def plot_bootstrap():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(df.index, series, label='Original')
                ax.plot(df.index, bootstrapped_mean, color='red', label='Bootstrap Mean')
                ax.fill_between(df.index, bootstrapped_lower, bootstrapped_upper, color='gray', alpha=0.3, label='95% CI')
                ax.set_title(f'Bootstrapping Time Series: {col}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.legend()
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_bootstrap)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_bootstrap_{col}.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
        
        bootstrap_results = {
            'mean_forecast': bootstrapped_mean[-5:].tolist(),
            'lower_ci': bootstrapped_lower[-5:].tolist(),
            'upper_ci': bootstrapped_upper[-5:].tolist()
        }
        
        self.interpret_results("Bootstrapping Time Series", {
            'image_paths': image_paths,
            'bootstrap_results': bootstrap_results
        }, table_name)

    def sarimax_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - SARIMAX Analysis"))
        image_paths = []
        sarimax_results = {}
        
        date_col = df.select_dtypes(include=['datetime64']).columns
        if len(date_col) > 0:
            df.set_index(date_col[0], inplace=True)
        else:
            df.index = pd.date_range(start='1/1/2000', periods=len(df))

        if df.index.freq is None:
            df = df.asfreq(pd.infer_freq(df.index))

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:
            if df[col].isna().all():
                continue

            df[col] = df[col].interpolate().bfill().ffill()

            try:
                # Determine optimal SARIMAX parameters (similar to ARIMA)
                p, d, q = self.determine_arima_parameters(df[col])
                model = SARIMAX(df[col], order=(p, d, q), seasonal_order=(1, 1, 1, 12))
                results = self.model_fit_with_timeout(model, timeout=30)

                if results is None:
                    raise TimeoutError("SARIMAX model fitting timed out")

                def plot_sarimax():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df.index, df[col], label='Observed')
                    ax.plot(df.index, results.fittedvalues, color='red', label='SARIMAX Fit')
                    ax.set_title(f'SARIMAX Analysis: {col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value')
                    ax.legend()
                    plt.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_sarimax)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_sarimax_{col}.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(img_path)

                forecast_steps = min(30, int(len(df) * 0.1))
                forecast = results.forecast(steps=forecast_steps)
                
                sarimax_results[col] = {
                    'aic': results.aic,
                    'bic': results.bic,
                    'mse': mean_squared_error(df[col][-forecast_steps:], results.fittedvalues[-forecast_steps:]),
                    'forecast': forecast.tolist()
                }

            except Exception as e:
                print(error(f"Error in SARIMAX analysis for column {col}: {str(e)}"))
                sarimax_results[col] = {'error': str(e)}

        self.interpret_results("SARIMAX Analysis", {
            'image_paths': image_paths,
            'sarimax_results': sarimax_results
        }, table_name)


    
    def theta_method(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Theta Method"))
        image_paths = []
        
        time_col = df.select_dtypes(include=['datetime64']).columns[0]
        df.set_index(time_col, inplace=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            # Simple implementation of Theta method
            series = df[col]
            n = len(series)
            theta = 2
            
            # Local linear trend
            b = (series.iloc[-1] - series.iloc[0]) / (n - 1)
            a = series.mean() - b * (n + 1) / 2
            
            trend = a + b * np.arange(1, n+1)
            detrended = series - trend
            
            # SES on the detrended series
            alpha = 0.5
            ses = [detrended.iloc[0]]
            for i in range(1, n):
                ses.append(alpha * detrended.iloc[i] + (1-alpha) * ses[-1])
            
            # Combine
            theta_forecast = trend + theta * np.array(ses)
            
            def plot_theta():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(df.index, series, label='Observed')
                ax.plot(df.index, theta_forecast, color='red', label='Theta Method')
                ax.set_title(f'Theta Method: {col}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.legend()
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_theta)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_theta_{col}.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
        
        theta_results = {
            'mse': mean_squared_error(series, theta_forecast),
            'forecast': (trend[-1] + theta * ses[-1] + b * np.arange(1, 6)).tolist()
        }
        
        self.interpret_results("Theta Method", {
            'image_paths': image_paths,
            'theta_results': theta_results
        }, table_name)

    def save_results(self, analysis_type, results):
        if not self.settings.save_results_to_txt:
            return  # Skip saving if the option is disabled

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
            output_file = os.path.join(self.output_folder, "axda_b7_results.txt")
            with open(output_file, "w", encoding='utf-8') as f:
                f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 7) Report for {self.table_name}"
        
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
            formatted_image_data,
            filename=f"axda_b7_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None
