import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
import os
from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class FinancialExploratoryDataAnalysis:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.output_folder = os.path.join(os.path.dirname(db_path), "fxda_output")
        os.makedirs(self.output_folder, exist_ok=True)
        self.text_output = ""
        self.pdf_content = []
        self.findings = []
        self.llm_name = f"Worker: {self.worker_erag_api.model}, Supervisor: {self.supervisor_erag_api.model}"
        self.toc_entries = []
        self.executive_summary = ""
        self.image_paths = []
        self.max_pixels = 400000
        self.data = {}

    
    def run(self):
        print(info(f"Starting Financial Exploratory Data Analysis on {self.db_path}"))
        self.load_all_data()
        self.analyze_financial_data()
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Financial Exploratory Data Analysis completed. Results saved in {self.output_folder}"))

    def analyze_financial_data(self):
        self.analyze_profit_loss()
        self.analyze_balance_sheet()
        self.analyze_cash_flow()
        self.analyze_ratios()
        self.analyze_shareholding()

    def load_all_data(self):
        tables = self.get_tables()
        for table in tables:
            df = self.load_table(table)
            self.data[table] = self.preprocess_data(df)

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]
    def load_table(self, table_name):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn)

    def preprocess_data(self, df):
        df = df.set_index(df.columns[0])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].replace({',': '', '%': '', '₹ ': '', 'Cr.': ''}, regex=True), errors='coerce')
        return df

    def analyze_balance_sheet(self):
        print(info("Performing Balance Sheet Analysis"))
        try:
            balance_sheet = self.data.get('Yearly_Balance_Sheet')
            if balance_sheet is not None:
                total_assets = balance_sheet.loc['Total Assets']
                total_liabilities = balance_sheet.loc['Total Liabilities']
                equity = balance_sheet.loc['Equity Capital'] + balance_sheet.loc['Reserves']
                debt_to_equity = balance_sheet.loc['Borrowings'] / equity
                current_ratio = balance_sheet.loc['Other Assets'] / balance_sheet.loc['Other Liabilities']

                assets_liabilities_plot = self.plot_assets_liabilities(balance_sheet)
                debt_equity_plot = self.plot_debt_equity_ratio(debt_to_equity)

                results = {
                    "Total Assets": total_assets.to_dict(),
                    "Total Liabilities": total_liabilities.to_dict(),
                    "Equity": equity.to_dict(),
                    "Debt to Equity Ratio": debt_to_equity.to_dict(),
                    "Current Ratio": current_ratio.to_dict(),
                    "Plots": [plot for plot in [assets_liabilities_plot, debt_equity_plot] if plot is not None]
                }

                self.interpret_results("Balance Sheet Analysis", results, "Yearly")
        except Exception as e:
            error_message = f"Error in Balance Sheet Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Balance Sheet Analysis", {"Error": error_message}, "Yearly")

    def analyze_profit_loss(self):
        print(info("Performing Profit & Loss Analysis"))
        try:
            yearly_pl = self.data.get('Yearly_Profit_Loss')
            quarterly_pl = self.data.get('Quarterly_Profit_Loss')

            if yearly_pl is not None:
                revenue = yearly_pl.loc['Sales']
                net_profit = yearly_pl.loc['Net Profit']
                profit_margin = (net_profit / revenue) * 100

                revenue_profit_plot = self.plot_revenue_and_profit(revenue, net_profit)
                profit_margin_plot = self.plot_profit_margin(profit_margin)

                results = {
                    "Revenue": revenue.to_dict(),
                    "Net Profit": net_profit.to_dict(),
                    "Profit Margin": profit_margin.to_dict(),
                    "Plots": [plot for plot in [revenue_profit_plot, profit_margin_plot] if plot is not None]
                }

                self.interpret_results("Profit & Loss Analysis", results, "Yearly")

            if quarterly_pl is not None:
                # Perform quarterly analysis
                quarterly_revenue = quarterly_pl.loc['Sales']
                quarterly_net_profit = quarterly_pl.loc['Net Profit']
                quarterly_profit_margin = (quarterly_net_profit / quarterly_revenue) * 100

                quarterly_revenue_profit_plot = self.plot_revenue_and_profit(quarterly_revenue, quarterly_net_profit, title_prefix="Quarterly ")
                quarterly_profit_margin_plot = self.plot_profit_margin(quarterly_profit_margin, title_prefix="Quarterly ")

                quarterly_results = {
                    "Quarterly Revenue": quarterly_revenue.to_dict(),
                    "Quarterly Net Profit": quarterly_net_profit.to_dict(),
                    "Quarterly Profit Margin": quarterly_profit_margin.to_dict(),
                    "Plots": [plot for plot in [quarterly_revenue_profit_plot, quarterly_profit_margin_plot] if plot is not None]
                }

                self.interpret_results("Quarterly Profit & Loss Analysis", quarterly_results, "Quarterly")

        except Exception as e:
            error_message = f"Error in Profit & Loss Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Profit & Loss Analysis", {"Error": error_message}, "Yearly and Quarterly")

    def analyze_cash_flow(self):
        print(info("Performing Cash Flow Analysis"))
        try:
            cash_flow = self.data.get('Yearly_Cash_flow')
            if cash_flow is not None:
                cash_flow_plot = self.plot_cash_flow_components(cash_flow)

                results = {
                    "Operating Cash Flow": cash_flow.loc['Cash from Operating Activity'].to_dict(),
                    "Investing Cash Flow": cash_flow.loc['Cash from Investing Activity'].to_dict(),
                    "Financing Cash Flow": cash_flow.loc['Cash from Financing Activity'].to_dict(),
                    "Net Cash Flow": cash_flow.loc['Net Cash Flow'].to_dict(),
                    "Plots": [cash_flow_plot] if cash_flow_plot is not None else []
                }

                self.interpret_results("Cash Flow Analysis", results, "Yearly")
        except Exception as e:
            error_message = f"Error in Cash Flow Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Cash Flow Analysis", {"Error": error_message}, "Yearly")

    def analyze_ratios(self):
        print(info("Performing Financial Ratios Analysis"))
        try:
            ratios = self.data.get('Ratios')
            if ratios is not None:
                efficiency_ratios_plot = self.plot_efficiency_ratios(ratios)
                roce_plot = self.plot_roce(ratios.loc['ROCE %'])

                results = {
                    "Efficiency Ratios": ratios.loc[['Debtor Days', 'Inventory Days', 'Days Payable']].to_dict(),
                    "Cash Conversion Cycle": ratios.loc['Cash Conversion Cycle'].to_dict(),
                    "Working Capital Days": ratios.loc['Working Capital Days'].to_dict(),
                    "ROCE": ratios.loc['ROCE %'].to_dict(),
                    "Plots": [plot for plot in [efficiency_ratios_plot, roce_plot] if plot is not None]
                }

                self.interpret_results("Financial Ratios Analysis", results, "Yearly")
        except Exception as e:
            error_message = f"Error in Financial Ratios Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Financial Ratios Analysis", {"Error": error_message}, "Yearly")

    def analyze_shareholding(self):
        print(info("Performing Shareholding Pattern Analysis"))
        try:
            yearly_shareholding = self.data.get('Yearly_Shareholding_Pattern')
            quarterly_shareholding = self.data.get('Quarterly_Shareholding_Pattern')

            if yearly_shareholding is not None:
                yearly_shareholding = yearly_shareholding.drop('No. of Shareholders')
                shareholding_plot = self.plot_shareholding_pattern(yearly_shareholding)
                shareholding_change = yearly_shareholding.diff(axis=1)

                results = {
                    "Shareholding Percentages": yearly_shareholding.to_dict(),
                    "Shareholding Changes": shareholding_change.to_dict(),
                    "Plots": [shareholding_plot] if shareholding_plot is not None else []
                }

                self.interpret_results("Yearly Shareholding Pattern Analysis", results, "Yearly")

            if quarterly_shareholding is not None:
                quarterly_shareholding = quarterly_shareholding.drop('No. of Shareholders')
                quarterly_shareholding_plot = self.plot_shareholding_pattern(quarterly_shareholding, title_prefix="Quarterly ")
                quarterly_shareholding_change = quarterly_shareholding.diff(axis=1)

                quarterly_results = {
                    "Quarterly Shareholding Percentages": quarterly_shareholding.to_dict(),
                    "Quarterly Shareholding Changes": quarterly_shareholding_change.to_dict(),
                    "Plots": [quarterly_shareholding_plot] if quarterly_shareholding_plot is not None else []
                }

                self.interpret_results("Quarterly Shareholding Pattern Analysis", quarterly_results, "Quarterly")

        except Exception as e:
            error_message = f"Error in Shareholding Pattern Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Shareholding Pattern Analysis", {"Error": error_message}, "Yearly and Quarterly")

    def analyze_generic(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing analysis {self.technique_counter}/{self.total_techniques} - Generic Table Analysis"))
        
        # Perform basic statistical analysis
        stats = df.describe()
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Plot correlation heatmap for numerical columns
        corr_plot = None
        if len(numerical_cols) > 1:
            corr_plot = self.plot_correlation_heatmap(df[numerical_cols], numerical_cols)
        
        # Plot distribution of numerical columns
        dist_plots = []
        for col in numerical_cols[:5]:  # Limit to first 5 columns
            try:
                plot = self.plot_distribution(df, col)
                if plot:
                    dist_plots.append(plot)
            except Exception as e:
                print(error(f"Error creating distribution plot for {col}: {str(e)}"))
        
        results = {
            "Basic Statistics": stats.to_dict(),
            "Numerical Columns": list(numerical_cols),
            "Categorical Columns": list(categorical_cols),
            "Plots": [plot for plot in [corr_plot] + dist_plots if plot is not None]
        }
        
        self.interpret_results("Generic Table Analysis", results, table_name)


    def identify_columns(self, df):
        """Identify relevant columns based on keywords."""
        column_mapping = {
            'year': df.filter(regex=r'(?i)year|date').columns,
            'current_assets': df.filter(regex=r'(?i)current.*assets').columns,
            'current_liabilities': df.filter(regex=r'(?i)current.*liabilities').columns,
            'inventory': df.filter(regex=r'(?i)inventory').columns,
            'total_liabilities': df.filter(regex=r'(?i)total.*liabilities').columns,
            'shareholders_equity': df.filter(regex=r'(?i)shareholders.*equity|total.*equity').columns,
            'net_income': df.filter(regex=r'(?i)net.*income|profit').columns,
            'total_assets': df.filter(regex=r'(?i)total.*assets').columns,
            'revenue': df.filter(regex=r'(?i)revenue|sales').columns,
            'operating_cash_flow': df.filter(regex=r'(?i)operating.*cash.*flow').columns,
            'investing_cash_flow': df.filter(regex=r'(?i)investing.*cash.*flow').columns,
            'financing_cash_flow': df.filter(regex=r'(?i)financing.*cash.*flow').columns,
        }
        return {k: v[0] if len(v) > 0 else None for k, v in column_mapping.items()}

    def calculate_ratios(self, df, columns):
        """Calculate various financial ratios."""
        ratios = pd.DataFrame(index=df.index)
        
        if all(columns.get(col) for col in ['current_assets', 'current_liabilities']):
            ratios['Current Ratio'] = df[columns['current_assets']] / df[columns['current_liabilities']]
        
        if all(columns.get(col) for col in ['current_assets', 'inventory', 'current_liabilities']):
            ratios['Quick Ratio'] = (df[columns['current_assets']] - df[columns['inventory']]) / df[columns['current_liabilities']]
        
        if all(columns.get(col) for col in ['total_liabilities', 'shareholders_equity']):
            ratios['Debt to Equity'] = df[columns['total_liabilities']] / df[columns['shareholders_equity']]
        
        if all(columns.get(col) for col in ['net_income', 'total_assets']):
            ratios['ROA'] = df[columns['net_income']] / df[columns['total_assets']]
        
        return ratios

    def perform_trend_analysis(self, df, columns):
        """Perform trend analysis on key metrics."""
        key_metrics = [col for col in columns.values() if col is not None]
        trend = df[key_metrics] / df[key_metrics].iloc[0] * 100
        return trend

    def plot_ratios(self, ratios):
        """Plot financial ratios over time."""
        fig, ax = plt.subplots(figsize=self.calculate_figure_size())
        sns.lineplot(data=ratios, ax=ax)
        ax.set_title('Financial Ratios Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Ratio Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        img_path = os.path.join(self.output_folder, "financial_ratios_plot.png")
        plt.savefig(img_path)
        plt.close(fig)
        return img_path

    def plot_correlation_heatmap(self, df, columns):
        try:
            corr = df[columns].corr()
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
            ax.set_title('Correlation Heatmap of Key Financial Metrics')
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, "correlation_heatmap.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating correlation heatmap: {str(e)}"))
            return None

    def calculate_profit_margins(self, df, columns):
        """Calculate profit margins."""
        margins = pd.DataFrame(index=df.index)
        
        if all(columns.get(col) for col in ['net_income', 'revenue']):
            margins['Net Profit Margin'] = df[columns['net_income']] / df[columns['revenue']] * 100
        
        return margins

    def plot_profit_margins(self, margins):
        """Plot profit margins over time."""
        fig, ax = plt.subplots(figsize=self.calculate_figure_size())
        sns.lineplot(data=margins, ax=ax)
        ax.set_title('Profit Margins Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Margin (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        img_path = os.path.join(self.output_folder, "profit_margins_plot.png")
        plt.savefig(img_path)
        plt.close(fig)
        return img_path

    def calculate_cash_flow_ratios(self, df, columns):
        """Calculate cash flow ratios."""
        cf_ratios = pd.DataFrame(index=df.index)
        
        if all(columns.get(col) for col in ['operating_cash_flow', 'current_liabilities']):
            cf_ratios['Cash Flow to Current Liabilities'] = df[columns['operating_cash_flow']] / df[columns['current_liabilities']]
        
        return cf_ratios

    def plot_cash_flow_components(self, cash_flow):
        try:
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            cash_flow.loc[['Cash from Operating Activity', 'Cash from Investing Activity', 'Cash from Financing Activity', 'Net Cash Flow']].plot(kind='bar', ax=ax)
            ax.set_title('Cash Flow Components Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Amount')
            ax.legend(title='Cash Flow Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, "cash_flow_components_plot.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating cash flow components plot: {str(e)}"))
            return None

    def plot_all_ratios(self, df):
        """Plot all ratios over time."""
        fig, ax = plt.subplots(figsize=self.calculate_figure_size())
        sns.lineplot(data=df, ax=ax)
        ax.set_title('Financial Ratios Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Ratio Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        img_path = os.path.join(self.output_folder, "all_ratios_plot.png")
        plt.savefig(img_path)
        plt.close(fig)
        return img_path

    def plot_shareholding_pattern(self, shareholding, title_prefix=""):
        try:
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            shareholding.plot(kind='area', stacked=True, ax=ax)
            ax.set_title(f'{title_prefix}Shareholding Pattern Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Percentage')
            ax.legend(title='Shareholder Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, f"{title_prefix.lower()}shareholding_pattern_plot.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating shareholding pattern plot: {str(e)}"))
            return None

    def plot_number_of_shareholders(self, no_of_shareholders):
        """Plot number of shareholders over time."""
        try:
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            no_of_shareholders.plot(kind='line', marker='o', ax=ax)
            ax.set_title('Number of Shareholders Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Shareholders')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, "number_of_shareholders_plot.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating number of shareholders plot: {str(e)}"))
            return None

    def plot_distribution(self, df, column):
        try:
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            sns.histplot(data=df, x=column, kde=True, ax=ax)
            ax.set_title(f'Distribution of {column}')
            plt.tight_layout()
            
            # Sanitize the column name for use in the filename
            safe_column_name = ''.join(c if c.isalnum() else '_' for c in column)
            img_path = os.path.join(self.output_folder, f"{safe_column_name}_distribution_plot.png")
            
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating distribution plot for {column}: {str(e)}"))
            return None

    def plot_assets_liabilities(self, balance_sheet):
        try:
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            balance_sheet.loc[['Total Assets', 'Total Liabilities']].plot(kind='bar', ax=ax)
            ax.set_title('Assets vs Liabilities Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Amount')
            ax.legend(title='Category')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, "assets_liabilities_plot.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating assets vs liabilities plot: {str(e)}"))
            return None

    def plot_debt_equity_ratio(self, debt_to_equity):
        try:
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            debt_to_equity.plot(kind='line', marker='o', ax=ax)
            ax.set_title('Debt to Equity Ratio Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Debt to Equity Ratio')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, "debt_equity_ratio_plot.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating debt to equity ratio plot: {str(e)}"))
            return None

    def plot_revenue_and_profit(self, revenue, net_profit, title_prefix=""):
        try:
            fig, ax1 = plt.subplots(figsize=self.calculate_figure_size())
            ax1.plot(revenue.index, revenue.values, 'b-', label='Revenue')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Revenue', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax1.twinx()
            ax2.plot(net_profit.index, net_profit.values, 'r-', label='Net Profit')
            ax2.set_ylabel('Net Profit', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            plt.title(f'{title_prefix}Revenue and Net Profit Over Time')
            fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
            plt.xticks(rotation=45)
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, f"{title_prefix.lower()}revenue_profit_plot.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating revenue and profit plot: {str(e)}"))
            return None


    def plot_profit_margin(self, profit_margin, title_prefix=""):
        try:
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            profit_margin.plot(kind='line', marker='o', ax=ax)
            ax.set_title(f'{title_prefix}Profit Margin Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Profit Margin (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, f"{title_prefix.lower()}profit_margin_plot.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating profit margin plot: {str(e)}"))
            return None

    def plot_cash_flow_components(self, cash_flow):
        try:
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            cash_flow[['Cash from Operating Activity', 'Cash from Investing Activity', 'Cash from Financing Activity', 'Net Cash Flow']].plot(kind='bar', ax=ax)
            ax.set_title('Cash Flow Components Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Amount')
            ax.legend(title='Cash Flow Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, "cash_flow_components_plot.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating cash flow components plot: {str(e)}"))
            return None

    def plot_efficiency_ratios(self, ratios):
        try:
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            ratios.loc[['Debtor Days', 'Inventory Days', 'Days Payable']].plot(kind='line', marker='o', ax=ax)
            ax.set_title('Efficiency Ratios Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Days')
            ax.legend(title='Ratio Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, "efficiency_ratios_plot.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating efficiency ratios plot: {str(e)}"))
            return None

    def plot_roce(self, roce):
        try:
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            roce.plot(kind='line', marker='o', ax=ax)
            ax.set_title('Return on Capital Employed (ROCE) Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('ROCE (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, "roce_plot.png")
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            print(error(f"Error creating ROCE plot: {str(e)}"))
            return None

    def interpret_results(self, analysis_type, results, time_frame):
        results_str = self.format_results(results)
        prompt = f"""
        As an expert financial analyst, provide a detailed interpretation of these financial results:

        Analysis type: {analysis_type}
        Time frame: {time_frame}
        Results:
        {results_str}

        Highlight noteworthy patterns, anomalies, or insights. Focus on the most important aspects valuable for financial analysis. Provide specific figures and numbers where possible. If there are errors in the data, try to infer meaningful insights from the available information.

        Structure your response as follows:

        1. Analysis:
        [Provide a detailed description of the financial analysis performed]

        2. Key Findings:
        [List the most significant findings from the analysis, including specific figures]

        3. Trends and Patterns:
        [Describe notable trends or patterns observed in the data, with numerical examples]

        4. Potential Risks or Opportunities:
        [Identify potential risks or opportunities based on the analysis, quantifying where possible]

        5. Recommendations:
        [Provide actionable recommendations based on the analysis, with specific targets or thresholds]

        If there are no significant findings in any section, briefly explain why.

        Interpretation:
        """
        interpretation = self.worker_erag_api.chat([{"role": "system", "content": "You are an expert financial analyst providing insights on exploratory financial data analysis results. Use your expertise to infer meaningful insights even if there are data inconsistencies. Respond in the requested format."}, 
                                                    {"role": "user", "content": prompt}])
        
        # Supervisory review to enhance the interpretation
        check_prompt = f"""
        As an expert financial analyst, review and enhance the following financial interpretation:

        Original financial data and analysis type:
        {prompt}

        Previous interpretation:
        {interpretation}

        Improve the interpretation by:
        1. Verifying the accuracy against the original financial data.
        2. Ensuring the structure (Analysis, Key Findings, Trends and Patterns, Potential Risks or Opportunities, Recommendations) is maintained.
        3. Making the interpretation more narrative and detailed by adding context and explanations specific to financial analysis.
        4. Addressing any important aspects of the financial data that weren't covered.
        5. Providing more specific figures, percentages, and numerical examples throughout the analysis.
        6. Inferring meaningful insights even if there are inconsistencies in the data.

        Provide your response in the same format, maintaining the original structure. 
        Do not add comments, questions, or explanations about the changes - simply provide the improved version.
        """

        enhanced_interpretation = self.supervisor_erag_api.chat([
            {"role": "system", "content": "You are an expert financial analyst improving interpretations of exploratory financial data analysis results. Provide direct enhancements without adding meta-comments or detailing the changes done."},
            {"role": "user", "content": check_prompt}
        ])

        print(success(f"Expert Financial Analysis Interpretation for {analysis_type}:"))
        print(enhanced_interpretation.strip())
        
        self.text_output += f"\n{enhanced_interpretation.strip()}\n\n"
        
        # Handle images
        image_data = []
        if 'Plots' in results:
            for i, plot_path in enumerate(results['Plots']):
                if plot_path:
                    image_data.append((f"{analysis_type} - Plot {i+1}", plot_path))
        
        self.pdf_content.append((analysis_type, image_data, enhanced_interpretation.strip()))
        
        # Extract important findings
        lines = enhanced_interpretation.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith("2. Key Findings:") or line.startswith("3. Trends and Patterns:") or line.startswith("4. Potential Risks or Opportunities:"):
                for finding in lines[i+1:]:
                    if finding.strip() and not finding.startswith(("2.", "3.", "4.", "5.")):
                        self.findings.append(f"{analysis_type}: {finding.strip()}")
                    elif finding.startswith(("2.", "3.", "4.", "5.")):
                        break

    def format_results(self, results):
        formatted = []
        for key, value in results.items():
            if isinstance(value, dict):
                formatted.append(f"{key}:\n{pd.DataFrame(value).to_string()}")
            elif isinstance(value, list) and key == "Plots":
                formatted.append(f"{key}: {len(value)} plots generated")
            else:
                formatted.append(f"{key}: {value}")
        return "\n\n".join(formatted)

                                 
    def generate_executive_summary(self):
        if not self.findings:
            self.executive_summary = "No significant findings were identified during the financial analysis. This could be due to a lack of data, uniform data distribution, or absence of notable patterns or anomalies in the financial dataset."
            return

        summary_prompt = f"""
        As an expert financial analyst, provide an executive summary of the following financial exploratory data analysis findings:
        
        {self.findings}
        
        The summary should:
        1. Briefly introduce the purpose of the financial analysis.
        2. Highlight the most significant financial insights and patterns discovered, using specific figures and percentages.
        3. Mention any potential financial issues or areas that require further investigation, quantifying the concerns where possible.
        4. Discuss any limitations of the analysis due to data constraints or other factors, and how you've attempted to infer insights despite these limitations.
        5. Conclude with recommendations for next steps or areas to focus on from a financial perspective, providing specific targets or thresholds where applicable.

        Structure the summary in multiple paragraphs for readability.
        Provide your response in plain text format, without any special formatting or markup.
        """
        
        try:
            interpretation = self.worker_erag_api.chat([
                {"role": "system", "content": "You are an expert financial analyst providing an executive summary of a financial exploratory data analysis. Use your expertise to provide meaningful insights even if there are data inconsistencies. Respond in plain text format."},
                {"role": "user", "content": summary_prompt}
            ])
            
            if interpretation is not None:
                # Updated supervisory review to focus on direct improvements
                check_prompt = f"""
                As an expert financial analyst, review and enhance the following financial executive summary:

                {interpretation}

                Improve the summary by:
                1. Making it more comprehensive and narrative by adding financial context and explanations.
                2. Addressing any important aspects of the financial analysis that weren't covered.
                3. Ensuring it includes a clear introduction, highlights of significant financial insights with specific figures, mention of potential issues with quantified concerns, and recommendations for next steps with specific targets.
                4. Discussing the implications of any data limitations on the overall financial analysis and how you've attempted to infer insights despite these limitations.
                5. Adding more specific figures, percentages, and numerical examples throughout the summary.

                Provide your response in plain text format, without any special formatting or markup.
                Do not add comments, questions, or explanations about the changes - simply provide the improved version.
                """

                enhanced_summary = self.supervisor_erag_api.chat([
                    {"role": "system", "content": "You are an expert financial analyst improving an executive summary of a financial exploratory data analysis. Provide direct enhancements without adding meta-comments."},
                    {"role": "user", "content": check_prompt}
                ])

                self.executive_summary = enhanced_summary.strip()
            else:
                self.executive_summary = "Error: Unable to generate financial executive summary."
        except Exception as e:
            print(error(f"An error occurred while generating the financial executive summary: {str(e)}"))
            self.executive_summary = "Error: Unable to generate financial executive summary due to an exception."

        print(success("Enhanced Expert Financial Executive Summary generated successfully."))
        print(self.executive_summary)

    def save_text_output(self):
        output_file = os.path.join(self.output_folder, "fxda_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = "Financial Exploratory Data Analysis Report"
        pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, "Financial Analysis")
        pdf_file = pdf_generator.create_enhanced_pdf_report(
            self.executive_summary,
            self.findings,
            self.pdf_content,
            [],  # We're not using image_data in this class
            filename="fxda_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"Financial PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate financial PDF report"))
            return None

    def calculate_figure_size(self, aspect_ratio=16/9):
        max_width = int(np.sqrt(self.max_pixels * aspect_ratio))
        max_height = int(max_width / aspect_ratio)
        return (max_width / 100, max_height / 100)
