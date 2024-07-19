# -*- coding: utf-8 -*-


import os
from datetime import datetime
import markdown
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
import re

# Define RGB values for custom colors
SAGE_GREEN_RGB = (125/255, 169/255, 133/255)
DUSTY_PINK_RGB = (173/255, 142/255, 148/255)

class PDFReportGenerator:
    def __init__(self, output_folder, llm_name):
        self.output_folder = output_folder
        self.llm_name = llm_name

    def create_enhanced_pdf_report(self, executive_summary, findings, pdf_content, image_paths):
        pdf_file = os.path.join(self.output_folder, "xda_report.pdf")
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)

        elements = []
        styles = self._create_styles()

        # Cover page
        elements.extend(self._create_cover_page(doc, styles))

        # Executive Summary
        elements.append(Paragraph("Executive Summary", styles['XDA_Important']))
        elements.extend(self._markdown_to_reportlab(executive_summary, styles))
        elements.append(PageBreak())

        # Key Findings
        if findings:
            elements.append(Paragraph("Key Findings", styles['XDA_Important']))
            for finding in findings:
                elements.extend(self._markdown_to_reportlab(finding, styles))
            elements.append(PageBreak())

        # Main content
        for analysis_type, results, interpretation in pdf_content:
            elements.append(Paragraph(analysis_type, styles['XDA_Important']))
            elements.extend(self._markdown_to_reportlab(interpretation, styles))

            if isinstance(results, list):
                for item in results:
                    if isinstance(item, tuple) and len(item) == 2:
                        description, img_path = item
                        elements.append(Paragraph(f"Reference to image: {os.path.basename(img_path)}", styles['XDA_Normal']))
                        elements.append(Image(img_path, width=6*inch, height=4*inch))
            elif isinstance(results, tuple) and len(results) == 2 and isinstance(results[1], str) and results[1].endswith('.png'):
                elements.append(Paragraph(f"Reference to image: {os.path.basename(results[1])}", styles['XDA_Normal']))
                elements.append(Image(results[1], width=6*inch, height=4*inch))

            elements.append(Spacer(1, 12))
            elements.append(PageBreak())

        try:
            doc.build(elements)
            print(f"PDF report saved to {pdf_file}")
            return pdf_file
        except Exception as e:
            print(f"Error building PDF: {str(e)}")
            print("Attempting to save partial PDF...")
            try:
                doc.build(elements[:len(elements)//2])  # Try to build with only half the content
                print(f"Partial PDF report saved to {pdf_file}")
                return pdf_file
            except:
                print("Failed to save even a partial PDF.")
                return None

    def _create_styles(self):
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='XDA_Title', parent=styles['Title'], fontSize=24, alignment=TA_CENTER, spaceAfter=24, textColor=colors.white, backColor=colors.Color(*SAGE_GREEN_RGB)))
        styles.add(ParagraphStyle(name='XDA_Normal', parent=styles['Normal'], fontSize=12, alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.black))
        styles.add(ParagraphStyle(name='XDA_Important', parent=styles['Normal'], fontSize=12, alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.blue, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='XDA_Positive', parent=styles['Normal'], fontSize=12, alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.green, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='XDA_Negative', parent=styles['Normal'], fontSize=12, alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.red, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='XDA_Conclusion', parent=styles['Normal'], fontSize=12, alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.Color(*SAGE_GREEN_RGB), fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='XDA_Bullet', parent=styles['Normal'], fontSize=12, alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=20, textColor=colors.black))
        return styles

    def _create_cover_page(self, doc, styles):
        def draw_background(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(colors.Color(*SAGE_GREEN_RGB))
            canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1)
            canvas.restoreState()

        cover_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='CoverFrame')
        cover_template = PageTemplate(id='CoverPage', frames=[cover_frame], onPage=draw_background)
        doc.addPageTemplates([cover_template])

        elements = []
        elements.append(Paragraph("Exploratory Data Analysis Report", styles['XDA_Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", styles['XDA_Normal']))
        elements.append(Paragraph(f"AI-powered analysis by ERAG using {self.llm_name}", styles['XDA_Normal']))
        elements.append(PageBreak())

        # Add a normal template for subsequent pages
        normal_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='NormalFrame')
        normal_template = PageTemplate(id='NormalPage', frames=[normal_frame])
        doc.addPageTemplates([normal_template])

        return elements

    def _markdown_to_reportlab(self, md_text, styles):
        try:
            html = markdown.markdown(md_text)
        except Exception as e:
            print(f"Error converting markdown to HTML: {str(e)}")
            return [Paragraph(md_text, styles['XDA_Normal'])]

        elements = []
        current_section = 'Normal'
        for line in html.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            try:
                if line.startswith('<h1>') or line.startswith('<h2>') or line.startswith('<h3>') or line.startswith('<h4>'):
                    heading = re.sub('<[^<]+?>', '', line)
                    heading = re.sub(r'^\d+\.?\s*', '', heading)
                    elements.append(Paragraph(heading, styles['XDA_Important']))
                elif line.lower().startswith('<p>analysis') or line.lower().startswith('<p>the '):
                    current_section = 'Normal'
                    elements.append(Paragraph(re.sub('<[^<]+?>', '', line), styles['XDA_Normal']))
                elif line.lower().startswith('<p>important'):
                    current_section = 'Important'
                    elements.append(Paragraph(re.sub('<[^<]+?>', '', line), styles['XDA_Important']))
                elif line.lower().startswith('<p>positive findings'):
                    current_section = 'Positive'
                    elements.append(Paragraph(re.sub('<[^<]+?>', '', line), styles['XDA_Positive']))
                elif line.lower().startswith('<p>negative findings'):
                    current_section = 'Negative'
                    elements.append(Paragraph(re.sub('<[^<]+?>', '', line), styles['XDA_Negative']))
                elif line.lower().startswith('<p>conclusion'):
                    current_section = 'Conclusion'
                    elements.append(Paragraph(re.sub('<[^<]+?>', '', line), styles['XDA_Conclusion']))
                elif line.startswith('<p>'):
                    elements.append(Paragraph(re.sub('<[^<]+?>', '', line), styles[f'XDA_{current_section}']))
                elif line.startswith('<ul>'):
                    pass  # We'll handle list items individually
                elif line.startswith('</ul>'):
                    pass  # End of list, no specific action needed
                elif line.startswith('<li>'):
                    bullet_text = re.sub('<[^<]+?>', '', line)
                    elements.append(Paragraph(f"• {bullet_text}", styles['XDA_Bullet']))
                elif line == '<hr />':
                    elements.append(Spacer(1, 12))
                else:
                    # For any other content, just add it as normal text
                    cleaned_line = re.sub('<[^<]+?>', '', line)
                    elements.append(Paragraph(cleaned_line, styles[f'XDA_{current_section}']))
            except Exception as e:
                print(f"Error processing line: {line}. Error: {str(e)}")
                # Add the problematic line as plain text
                elements.append(Paragraph(line, styles['XDA_Normal']))

        return elements
