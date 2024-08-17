# Standard library imports
import os
import sqlite3

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Local imports
from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator

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
        self.all_image_paths = []
        self.image_data = []

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
        self.all_image_paths = []  # Reset image paths at the start of analysis
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

                image_paths = []
                assets_liabilities_plot = self.plot_assets_liabilities(balance_sheet)
                if assets_liabilities_plot:
                    image_paths.append(assets_liabilities_plot)
                debt_equity_plot = self.plot_debt_equity_ratio(debt_to_equity)
                if debt_equity_plot:
                    image_paths.append(debt_equity_plot)

                self.all_image_paths.extend(image_paths)  # Add to overall image paths

                results = {
                    "Total Assets": total_assets.to_dict(),
                    "Total Liabilities": total_liabilities.to_dict(),
                    "Equity": equity.to_dict(),
                    "Debt to Equity Ratio": debt_to_equity.to_dict(),
                    "Current Ratio": current_ratio.to_dict(),
                }

                self.interpret_results("Balance Sheet Analysis", {
                    'image_paths': image_paths,
                    'results': results
                }, "Yearly")
        except Exception as e:
            error_message = f"Error in Balance Sheet Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Balance Sheet Analysis", {
                'image_paths': [],
                'results': {"Error": error_message}
            }, "Yearly")

    def analyze_profit_loss(self):
        print(info("Performing Profit & Loss Analysis"))
        try:
            yearly_pl = self.data.get('Yearly_Profit_Loss')
            quarterly_pl = self.data.get('Quarterly_Profit_Loss')

            image_paths = []
            results = {}

            if yearly_pl is not None:
                revenue = yearly_pl.loc['Sales']
                net_profit = yearly_pl.loc['Net Profit']
                profit_margin = (net_profit / revenue) * 100

                revenue_profit_plot = self.plot_revenue_and_profit(revenue, net_profit)
                if revenue_profit_plot:
                    image_paths.append(revenue_profit_plot)
                profit_margin_plot = self.plot_profit_margin(profit_margin)
                if profit_margin_plot:
                    image_paths.append(profit_margin_plot)

                results.update({
                    "Yearly Revenue": revenue.to_dict(),
                    "Yearly Net Profit": net_profit.to_dict(),
                    "Yearly Profit Margin": profit_margin.to_dict(),
                })

            if quarterly_pl is not None:
                quarterly_revenue = quarterly_pl.loc['Sales']
                quarterly_net_profit = quarterly_pl.loc['Net Profit']
                quarterly_profit_margin = (quarterly_net_profit / quarterly_revenue) * 100

                quarterly_revenue_profit_plot = self.plot_revenue_and_profit(quarterly_revenue, quarterly_net_profit, title_prefix="Quarterly ")
                if quarterly_revenue_profit_plot:
                    image_paths.append(quarterly_revenue_profit_plot)
                quarterly_profit_margin_plot = self.plot_profit_margin(quarterly_profit_margin, title_prefix="Quarterly ")
                if quarterly_profit_margin_plot:
                    image_paths.append(quarterly_profit_margin_plot)

                results.update({
                    "Quarterly Revenue": quarterly_revenue.to_dict(),
                    "Quarterly Net Profit": quarterly_net_profit.to_dict(),
                    "Quarterly Profit Margin": quarterly_profit_margin.to_dict(),
                })

            self.all_image_paths.extend(image_paths)  # Add to overall image paths

            self.interpret_results("Profit & Loss Analysis", {
                'image_paths': image_paths,
                'results': results
            }, "Yearly and Quarterly")

        except Exception as e:
            error_message = f"Error in Profit & Loss Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Profit & Loss Analysis", {
                'image_paths': [],
                'results': {"Error": error_message}
            }, "Yearly and Quarterly")

    def analyze_cash_flow(self):
        print(info("Performing Cash Flow Analysis"))
        try:
            cash_flow = self.data.get('Yearly_Cash_flow')
            if cash_flow is not None:
                image_paths = []
                cash_flow_plot = self.plot_cash_flow_components(cash_flow)
                if cash_flow_plot:
                    image_paths.append(cash_flow_plot)

                self.all_image_paths.extend(image_paths)  # Add to overall image paths

                results = {
                    "Operating Cash Flow": cash_flow.loc['Cash from Operating Activity'].to_dict(),
                    "Investing Cash Flow": cash_flow.loc['Cash from Investing Activity'].to_dict(),
                    "Financing Cash Flow": cash_flow.loc['Cash from Financing Activity'].to_dict(),
                    "Net Cash Flow": cash_flow.loc['Net Cash Flow'].to_dict(),
                }

                self.interpret_results("Cash Flow Analysis", {
                    'image_paths': image_paths,
                    'results': results
                }, "Yearly")
        except Exception as e:
            error_message = f"Error in Cash Flow Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Cash Flow Analysis", {
                'image_paths': [],
                'results': {"Error": error_message}
            }, "Yearly")


    def analyze_ratios(self):
        print(info("Performing Financial Ratios Analysis"))
        try:
            ratios = self.data.get('Ratios')
            if ratios is not None:
                image_paths = []
                efficiency_ratios_plot = self.plot_efficiency_ratios(ratios)
                if efficiency_ratios_plot:
                    image_paths.append(efficiency_ratios_plot)
                roce_plot = self.plot_roce(ratios.loc['ROCE %'])
                if roce_plot:
                    image_paths.append(roce_plot)

                self.all_image_paths.extend(image_paths)  # Add to overall image paths

                results = {
                    "Efficiency Ratios": ratios.loc[['Debtor Days', 'Inventory Days', 'Days Payable']].to_dict(),
                    "Cash Conversion Cycle": ratios.loc['Cash Conversion Cycle'].to_dict(),
                    "Working Capital Days": ratios.loc['Working Capital Days'].to_dict(),
                    "ROCE": ratios.loc['ROCE %'].to_dict(),
                }

                self.interpret_results("Financial Ratios Analysis", {
                    'image_paths': image_paths,
                    'results': results
                }, "Yearly")
        except Exception as e:
            error_message = f"Error in Financial Ratios Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Financial Ratios Analysis", {
                'image_paths': [],
                'results': {"Error": error_message}
            }, "Yearly")

    def analyze_shareholding(self):
        print(info("Performing Shareholding Pattern Analysis"))
        try:
            yearly_shareholding = self.data.get('Yearly_Shareholding_Pattern')
            quarterly_shareholding = self.data.get('Quarterly_Shareholding_Pattern')

            image_paths = []
            results = {}

            if yearly_shareholding is not None:
                yearly_shareholding = yearly_shareholding.drop('No. of Shareholders')
                shareholding_plot = self.plot_shareholding_pattern(yearly_shareholding)
                if shareholding_plot:
                    image_paths.append(shareholding_plot)
                shareholding_change = yearly_shareholding.diff(axis=1)

                results.update({
                    "Yearly Shareholding Percentages": yearly_shareholding.to_dict(),
                    "Yearly Shareholding Changes": shareholding_change.to_dict(),
                })

            if quarterly_shareholding is not None:
                quarterly_shareholding = quarterly_shareholding.drop('No. of Shareholders')
                quarterly_shareholding_plot = self.plot_shareholding_pattern(quarterly_shareholding, title_prefix="Quarterly ")
                if quarterly_shareholding_plot:
                    image_paths.append(quarterly_shareholding_plot)
                quarterly_shareholding_change = quarterly_shareholding.diff(axis=1)

                results.update({
                    "Quarterly Shareholding Percentages": quarterly_shareholding.to_dict(),
                    "Quarterly Shareholding Changes": quarterly_shareholding_change.to_dict(),
                })

            self.all_image_paths.extend(image_paths)  # Add to overall image paths

            self.interpret_results("Shareholding Pattern Analysis", {
                'image_paths': image_paths,
                'results': results
            }, "Yearly and Quarterly")

        except Exception as e:
            error_message = f"Error in Shareholding Pattern Analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Shareholding Pattern Analysis", {
                'image_paths': [],
                'results': {"Error": error_message}
            }, "Yearly and Quarterly")

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

    def interpret_results(self, analysis_type, results, time_frame):
        image_paths = results.get('image_paths', [])
        results_data = results.get('results', {})
        results_str = self.format_results(results_data)  # Add this line to create results_str
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
        
        image_data = [(f"{analysis_type} - Plot {i+1}", path) for i, path in enumerate(image_paths)]
        self.pdf_content.append((analysis_type, image_data, enhanced_interpretation.strip()))
        self.image_data.extend(image_data) 
    
        
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
            elif isinstance(value, list):
                formatted.append(f"{key}: {', '.join(map(str, value))}")
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
        try:
            report_title = "Financial Exploratory Data Analysis Report"
            pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, "Financial Analysis")
            
            # No need to collect image data here, as we're now using self.image_data
            
            pdf_file = pdf_generator.create_enhanced_pdf_report(
                self.executive_summary,
                self.findings,
                self.pdf_content,
                self.image_data,  # Use self.image_data directly
                filename="fxda_report",
                report_title=report_title
            )
            
            if pdf_file:
                print(success(f"Financial PDF report generated successfully: {pdf_file}"))
                return pdf_file
            else:
                print(error("Failed to generate financial PDF report"))
                return None
        
        except Exception as e:
            error_message = f"An error occurred while generating the PDF report: {str(e)}"
            print(error(error_message))
            return None

    def calculate_figure_size(self, aspect_ratio=16/9):
        max_width = int(np.sqrt(self.max_pixels * aspect_ratio))
        max_height = int(max_width / aspect_ratio)
        return (max_width / 100, max_height / 100)
