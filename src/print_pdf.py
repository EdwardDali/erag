# -*- coding: utf-8 -*-
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus.paragraph import ParaLines
import re

# Define RGB values for custom colors
DARK_BLUE_RGB = (34/255, 34/255, 59/255)
MEDIUM_BLUE_RGB = (43/255, 116/255, 238/255)

class PDFReportGenerator:
    def __init__(self, output_folder, llm_name, project_name):
        self.output_folder = output_folder
        self.llm_name = llm_name
        self.project_name = project_name
        self.report_title = None
        self.styles = self._create_styles()

    def create_enhanced_pdf_report(self, findings, pdf_content, image_data, filename="report", report_title=None):
        self.report_title = report_title or f"Analysis Report for {self.project_name}"
        pdf_file = os.path.join(self.output_folder, f"{filename}.pdf")
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)

        elements = []

        # Cover page
        elements.extend(self._create_cover_page(doc))

        # Table of Contents
        elements.append(Paragraph("Table of Contents", self.styles['Heading1']))
        elements.append(Paragraph("Key Findings", self.styles['Normal']))
        for analysis_type, _, _ in pdf_content:
            elements.append(Paragraph(analysis_type, self.styles['Normal']))
        elements.append(PageBreak())

        # Key Findings
        if findings:
            elements.append(Paragraph("Key Findings", self.styles['Heading1']))
            for finding in findings:
                elements.extend(self._text_to_reportlab(finding))
            elements.append(PageBreak())

        # Main content
        for analysis_type, image_paths, interpretation in pdf_content:
            elements.append(Paragraph(analysis_type, self.styles['Heading1']))
            elements.extend(self._text_to_reportlab(interpretation))

            # Add images for this analysis type
            for description, img_path in image_paths:
                if os.path.exists(img_path):
                    img = Image(img_path)
                    available_width = doc.width
                    aspect = img.drawHeight / img.drawWidth
                    img.drawWidth = available_width
                    img.drawHeight = available_width * aspect
                    elements.append(img)
                    elements.append(Paragraph(description, self.styles['Caption']))
                    elements.append(Spacer(1, 12))

            elements.append(PageBreak())

        try:
            doc.build(elements, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)
            print(f"PDF report saved to {pdf_file}")
            return pdf_file
        except Exception as e:
            print(f"Error building PDF: {str(e)}")
            return None

    def _create_styles(self):
        styles = getSampleStyleSheet()
        
        # Modify existing styles
        styles['Title'].fontSize = 24
        styles['Title'].alignment = TA_CENTER
        styles['Title'].textColor = colors.white
        styles['Title'].backColor = colors.Color(*MEDIUM_BLUE_RGB)
        styles['Title'].spaceAfter = 12
        styles['Title'].spaceBefore = 12
        styles['Title'].leading = 30

        styles['Heading1'].fontSize = 18
        styles['Heading1'].alignment = TA_JUSTIFY
        styles['Heading1'].spaceAfter = 12
        styles['Heading1'].textColor = colors.Color(*MEDIUM_BLUE_RGB)

        # Modify or add Heading2 style
        if 'Heading2' in styles:
            styles['Heading2'].fontSize = 16
            styles['Heading2'].textColor = colors.Color(*DARK_BLUE_RGB)
        else:
            styles.add(ParagraphStyle(
                name='Heading2',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.Color(*DARK_BLUE_RGB)
            ))

        styles['Normal'].fontSize = 10
        styles['Normal'].alignment = TA_JUSTIFY
        styles['Normal'].spaceAfter = 6
        styles['Normal'].textColor = colors.black

        # Add or modify Caption style
        if 'Caption' in styles:
            styles['Caption'].fontSize = 8
            styles['Caption'].alignment = TA_CENTER
            styles['Caption'].spaceAfter = 6
            styles['Caption'].textColor = colors.Color(*DARK_BLUE_RGB)
            styles['Caption'].fontName = 'Helvetica-Bold'
        else:
            styles.add(ParagraphStyle(
                name='Caption',
                parent=styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                spaceAfter=6,
                textColor=colors.Color(*DARK_BLUE_RGB),
                fontName='Helvetica-Bold'
            ))

        return styles




    def _create_cover_page(self, doc):
        def draw_background(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(colors.Color(*MEDIUM_BLUE_RGB))
            canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1)
            canvas.restoreState()

        cover_frame = Frame(
            doc.leftMargin, 
            doc.bottomMargin, 
            doc.width, 
            doc.height,
            id='CoverFrame'
        )
        cover_template = PageTemplate(id='CoverPage', frames=[cover_frame], onPage=draw_background)
        doc.addPageTemplates([cover_template])

        elements = []
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(self.report_title, self.styles['Title']))
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", self.styles['Normal']))
        elements.append(Paragraph(f"AI-powered analysis by ERAG using {self.llm_name}", self.styles['Normal']))
        elements.append(PageBreak())

        # Add a normal template for subsequent pages
        normal_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='NormalFrame')
        normal_template = PageTemplate(id='NormalPage', frames=[normal_frame], onPage=self._add_header_footer)
        doc.addPageTemplates([normal_template])

        return elements

    def _add_header_footer(self, canvas, doc):
        canvas.saveState()
        
        # Header
        canvas.setFillColor(colors.Color(*DARK_BLUE_RGB))
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawString(inch, doc.pagesize[1] - 0.5*inch, self.report_title)
        
        # Footer
        canvas.setFillColor(colors.Color(*DARK_BLUE_RGB))
        canvas.setFont('Helvetica', 8)
        canvas.drawString(inch, 0.5 * inch, f"Page {doc.page}")
        canvas.drawRightString(doc.pagesize[0] - inch, 0.5 * inch, "Powered by ERAG")

        canvas.restoreState()

    def _text_to_reportlab(self, text):
        elements = []
        for paragraph in text.split('\n'):
            if paragraph.strip():
                try:
                    # Check for Markdown-style headers
                    if paragraph.startswith('# '):
                        elements.append(Paragraph(paragraph[2:], self.styles['Heading1']))
                    elif paragraph.startswith('## '):
                        elements.append(Paragraph(paragraph[3:], self.styles['Heading2']))
                    else:
                        # Try to create a Paragraph object
                        para = Paragraph(paragraph, self.styles['Normal'])
                        # Test if the paragraph can be split into lines without error
                        _ = para.split(self.styles['Normal'].fontSize, self.styles['Normal'].leading)
                        elements.append(para)
                except Exception as e:
                    # If there's an error, clean the text and try again
                    cleaned_text = self._clean_text(paragraph)
                    try:
                        para = Paragraph(cleaned_text, self.styles['Normal'])
                        _ = para.split(self.styles['Normal'].fontSize, self.styles['Normal'].leading)
                        elements.append(para)
                    except:
                        # If it still fails, add the text as a simple string
                        print(f"Warning: Could not parse paragraph: {cleaned_text[:50]}...")
                        elements.append(cleaned_text)
        return elements

    def _clean_text(self, text):
        # Remove any HTML-like tags
        text = re.sub('<[^<]+?>', '', text)
        # Replace special characters with their HTML entities
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Remove any non-printable characters
        text = ''.join(char for char in text if ord(char) > 31 or ord(char) == 9)
        return text
