import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from reportlab.lib.enums import TA_JUSTIFY


class ExploratoryDataAnalysis:
    def __init__(self, erag_api, db_path):
        self.erag_api = erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 5
        self.output_folder = os.path.join(os.path.dirname(db_path), "xda_output")
        os.makedirs(self.output_folder, exist_ok=True)
        self.text_output = ""
        self.pdf_content = []
        self.findings = []
        self.llm_name = self.erag_api.model
        self.toc_entries = []
        self.executive_summary = ""

    def run(self):
        print(info(f"Starting Exploratory Data Analysis on {self.db_path}"))
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        
        self.save_text_output()
        self.create_enhanced_pdf_report()
        print(success(f"Exploratory Data Analysis completed. Results saved in {self.output_folder}"))

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        self.basic_statistics(df, table_name)
        self.data_types_and_missing_values(df, table_name)
        self.numerical_features_distribution(df, table_name)
        self.correlation_analysis(df, table_name)
        self.categorical_features_analysis(df, table_name)

    def basic_statistics(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Basic Statistics"))
        stats = df.describe()
        self.interpret_results(f"{self.technique_counter}. Basic Statistics", stats, table_name)

    def data_types_and_missing_values(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Data Types and Missing Values"))
        data_types = df.dtypes.to_frame(name='Data Type')
        missing_values = df.isnull().sum().to_frame(name='Missing Values')
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_frame(name='Missing Percentage')
        results = pd.concat([data_types, missing_values, missing_percentage], axis=1)
        self.interpret_results(f"{self.technique_counter}. Data Types and Missing Values", results, table_name)

    def numerical_features_distribution(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Numerical Features Distribution"))
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            results = []
            for col in numerical_columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col], kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_distribution.png")
                plt.savefig(img_path, dpi=300, bbox_inches='tight')
                plt.close()
                results.append((f"Distribution stats for {col}:\n{df[col].describe().to_string()}", img_path))
        else:
            results = "N/A - No numerical features found"
        self.interpret_results(f"{self.technique_counter}. Numerical Features Distribution", results, table_name)

    def correlation_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Correlation Analysis"))
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 1:
            correlation_matrix = df[numerical_columns].corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            plt.title('Correlation Matrix Heatmap')
            img_path = os.path.join(self.output_folder, f"{table_name}_correlation_matrix.png")
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            results = (correlation_matrix, img_path)
        else:
            results = "N/A - Not enough numerical features for correlation analysis"
        self.interpret_results(f"{self.technique_counter}. Correlation Analysis", results, table_name)

    def categorical_features_analysis(self, df, table_name):
        self.technique_counter += 1
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Categorical Features Analysis"))
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            results = []
            for col in categorical_columns:
                plt.figure(figsize=(10, 6))
                value_counts = df[col].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_distribution.png")
                plt.savefig(img_path, dpi=300, bbox_inches='tight')
                plt.close()
                results.append((f"Value counts for {col}:\n{value_counts.to_string()}", img_path))
        else:
            results = "N/A - No categorical features found"
        self.interpret_results(f"{self.technique_counter}. Categorical Features Analysis", results, table_name)

    def interpret_results(self, analysis_type, results, table_name):
        if isinstance(results, pd.DataFrame):
            results_str = f"DataFrame with shape {results.shape}:\n{results.to_string()}"
        elif isinstance(results, tuple) and isinstance(results[0], pd.DataFrame):
            results_str = f"DataFrame with shape {results[0].shape}:\n{results[0].to_string()}\nImage path: {results[1]}"
        elif isinstance(results, list):
            results_str = "\n".join([str(item) for item in results])
        else:
            results_str = str(results)

        prompt = f"""
        Analysis type: {analysis_type}
        Table name: {table_name}
        Results:
        {results_str}

        Please provide a detailed interpretation of these results, highlighting any noteworthy patterns, anomalies, or insights. Focus on the most important aspects that would be valuable for data analysis. If there are no significant findings, state "No significant findings" and briefly explain why.

        Structure your response in multiple paragraphs, covering different aspects of the analysis. If you identify any important findings, start the relevant sentence with "IMPORTANT:".

        Interpretation:
        """
        interpretation = self.erag_api.chat([{"role": "system", "content": "You are a data analyst providing insights on exploratory data analysis results."}, 
                                             {"role": "user", "content": prompt}])
        
        print(success(f"AI Interpretation for {analysis_type}:"))
        print(interpretation.strip())
        
        self.text_output += f"\n{analysis_type}\n"
        self.text_output += f"Results:\n{results_str}\n"
        self.text_output += f"AI Interpretation:\n{interpretation.strip()}\n\n"
        
        self.pdf_content.append((analysis_type, results, interpretation.strip()))
        
        for line in interpretation.strip().split('\n'):
            if line.startswith("IMPORTANT:"):
                self.findings.append(f"{analysis_type}: {line}")

    def generate_executive_summary(self):
        if not self.findings:
            self.executive_summary = "No significant findings were identified during the analysis. This could be due to a lack of data, uniform data distribution, or absence of notable patterns or anomalies in the dataset."
            return

        summary_prompt = f"""
        Based on the following findings from the Exploratory Data Analysis:
        
        {self.findings}
        
        Please provide an executive summary of the analysis. The summary should:
        1. Briefly introduce the purpose of the analysis.
        2. Highlight the most significant insights and patterns discovered.
        3. Mention any potential issues or areas that require further investigation.
        4. Conclude with recommendations for next steps or areas to focus on.

        Structure the summary in multiple paragraphs for readability.
        """
        
        try:
            interpretation = self.erag_api.chat([
                {"role": "system", "content": "You are a data analyst providing an executive summary of an exploratory data analysis."},
                {"role": "user", "content": summary_prompt}
            ])
            
            if interpretation is not None:
                self.executive_summary = interpretation.strip()
            else:
                self.executive_summary = "Error: Unable to generate executive summary."
        except Exception as e:
            print(error(f"An error occurred while generating the executive summary: {str(e)}"))
            self.executive_summary = "Error: Unable to generate executive summary due to an exception."

        print(success("Executive Summary generated successfully."))
        print(self.executive_summary)

    def save_text_output(self):
        output_file = os.path.join(self.output_folder, "xda_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def create_enhanced_pdf_report(self):
        pdf_path = os.path.join(self.output_folder, 'eda_report.pdf')
        doc = BaseDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Modify the default style to justify text
        styles['Normal'].alignment = TA_JUSTIFY
        
        story = []

        # Create cover page
        cover_style = ParagraphStyle(
            name='CoverTitle',
            fontSize=24,
            leading=30,
            alignment=1,  # Center alignment
            textColor=HexColor("#000080"),
            spaceAfter=20
        )
        story.append(Paragraph("Exploratory Data Analysis Report", cover_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
        story.append(Paragraph(f"AI-powered analysis by ERAG using {self.llm_name}", styles['Normal']))
        story.append(PageBreak())

        # Add table of contents
        toc_style = ParagraphStyle(
            name='TOC',
            fontSize=18,
            leading=22,
            spaceAfter=10,
            textColor=HexColor("#000080"),
            alignment=TA_JUSTIFY  # Justify alignment for TOC
        )
        story.append(Paragraph("Table of Contents", toc_style))
        story.append(Spacer(1, 12))
        for i, entry in enumerate(self.toc_entries):
            story.append(Paragraph(f"{i+1}. {entry}", styles['Normal']))
        story.append(PageBreak())

        # Add executive summary
        executive_summary_style = ParagraphStyle(
            name='ExecutiveSummary',
            fontSize=16,
            leading=20,
            spaceAfter=20,
            textColor=HexColor("#000080"),
            alignment=TA_JUSTIFY  # Justify alignment for Executive Summary
        )
        story.append(Paragraph("Executive Summary", executive_summary_style))
        story.append(Spacer(1, 12))
        for paragraph in self.executive_summary.split('\n\n'):
            story.append(Paragraph(paragraph, styles['Normal']))
        story.append(PageBreak())

        # Add key findings
        if self.findings:
            story.append(Paragraph("Key Findings", executive_summary_style))
            story.append(Spacer(1, 12))
            for finding in self.findings:
                story.append(Paragraph(finding, styles['Normal']))
            story.append(PageBreak())

        # Add detailed analysis sections
        for i, (analysis_type, results, interpretation) in enumerate(self.pdf_content):
            section_title_style = ParagraphStyle(
                name='SectionTitle',
                fontSize=14,
                leading=18,
                spaceAfter=12,
                textColor=HexColor("#000080"),
                alignment=TA_JUSTIFY  # Justify alignment for Section Titles
            )
            story.append(Paragraph(f"Section {i+1}: {analysis_type}", section_title_style))
            story.append(Spacer(1, 12))
            
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, tuple) and len(item) == 2:
                        description, img_path = item
                        story.append(Paragraph(f"Reference to image: {os.path.basename(img_path)}", styles['Normal']))
            elif isinstance(results, str) and results.endswith('.png'):
                story.append(Paragraph(f"Reference to image: {os.path.basename(results)}", styles['Normal']))
            
            story.append(Spacer(1, 12))
            for paragraph in interpretation.split('\n\n'):
                story.append(Paragraph(paragraph.replace("IMPORTANT:", "<b>IMPORTANT:</b>"), styles['Normal']))
            story.append(PageBreak())


        
        def header_footer(canvas, doc):
            canvas.saveState()
            header = Paragraph("Exploratory Data Analysis Report", styles['Normal'])
            w, h = header.wrap(doc.width, doc.topMargin)
            header.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - h + 20)

            footer_text = f"Page {doc.page}"
            footer = Paragraph(footer_text, styles['Normal'])
            w, h = footer.wrap(doc.width, doc.bottomMargin)
            footer.drawOn(canvas, doc.leftMargin, h)

            canvas.restoreState()

        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
        template = PageTemplate(id='test', frames=[frame], onPage=header_footer)
        doc.addPageTemplates([template])
        
        doc.build(story)
        print(success(f"PDF report saved to {pdf_path}"))
