# -*- coding: utf-8 -*-

import os
from datetime import datetime
import markdown
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, Frame, PageTemplate, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
import re

# Define RGB values for custom colors
ERAG_RGB = (252/255, 243/255, 207/255)
DARK_BLUE_RGB = (34/255, 34/255, 59/255)
LIGHT_GREY_RGB = (242/255, 242/255, 242/255)
MEDIUM_BLUE_RGB = (70/255, 130/255, 180/255)  # Steel Blue
DARK_GREEN_RGB = (0/255, 100/255, 0/255)  # Dark Green
DARK_RED_RGB = (139/255, 0/255, 0/255)  # Dark Red

class PDFReportGenerator:
    def __init__(self, output_folder, llm_name, project_name):
        self.output_folder = output_folder
        self.llm_name = llm_name
        self.project_name = project_name
        self.report_title = None

    
    def create_enhanced_pdf_report(self, executive_summary, findings, pdf_content, image_data, filename="report", report_title=None):
        self.report_title = report_title or f"Analysis Report for {self.project_name}"
        pdf_file = os.path.join(self.output_folder, f"{filename}.pdf")
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)

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
         # Main content
        for analysis_type, results, interpretation in pdf_content:
            elements.append(Paragraph(analysis_type, styles['XDA_Important']))
            elements.extend(self._markdown_to_reportlab(interpretation, styles))

            # Add images for this analysis type
            analysis_images = [img for img in image_data if img[0].startswith(analysis_type)]
            for description, img_path in analysis_images:
                if os.path.exists(img_path):
                    elements.append(Paragraph(description, styles['XDA_Caption']))
                    img = Image(img_path, width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 12))

            elements.append(PageBreak())

        try:
            doc.build(elements, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)
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
        styles.add(ParagraphStyle(name='XDA_Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=24, textColor=colors.white, backColor=colors.Color(*DARK_BLUE_RGB)))
        styles.add(ParagraphStyle(name='XDA_Normal', parent=styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.black, fontName='Helvetica'))
        styles.add(ParagraphStyle(name='XDA_Important', parent=styles['Normal'], fontSize=10, alignment=TA_LEFT, spaceAfter=12, textColor=colors.Color(*MEDIUM_BLUE_RGB), fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='XDA_Positive', parent=styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.Color(*DARK_GREEN_RGB), fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='XDA_Negative', parent=styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.Color(*DARK_RED_RGB), fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='XDA_Conclusion', parent=styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.Color(*MEDIUM_BLUE_RGB), fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='XDA_Bullet', parent=styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=20, textColor=colors.black))
        styles.add(ParagraphStyle(name='XDA_Code', parent=styles['Code'], fontSize=8, textColor=colors.black, backColor=colors.Color(*LIGHT_GREY_RGB), fontName='Courier'))
        styles.add(ParagraphStyle(name='XDA_Limitations', parent=styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.black, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='XDA_Caption', parent=styles['Normal'], fontSize=8, alignment=TA_LEFT, spaceAfter=6, textColor=colors.Color(*DARK_BLUE_RGB), fontName='Helvetica-Bold'))
        return styles

    def _create_cover_page(self, doc, styles):
        def draw_background(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(colors.Color(*ERAG_RGB))
            canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1)
            canvas.restoreState()

        cover_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='CoverFrame')
        cover_template = PageTemplate(id='CoverPage', frames=[cover_frame], onPage=draw_background)
        doc.addPageTemplates([cover_template])

        elements = []
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(self.report_title, styles['XDA_Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", styles['XDA_Normal']))
        elements.append(Paragraph(f"AI-powered analysis by ERAG using {self.llm_name}", styles['XDA_Normal']))
        elements.append(PageBreak())

        # Add a normal template for subsequent pages
        normal_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='NormalFrame')
        normal_template = PageTemplate(id='NormalPage', frames=[normal_frame], onPage=self._add_header_footer)
        doc.addPageTemplates([normal_template])

        return elements

    def _add_header_footer(self, canvas, doc):
        canvas.saveState()
        
        # Header
        canvas.setFillColor(colors.Color(*MEDIUM_BLUE_RGB))
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawString(inch, doc.pagesize[1] - 0.5*inch, self.report_title)
        
        # Footer
        canvas.setFillColor(colors.Color(*DARK_BLUE_RGB))
        canvas.setFont('Helvetica', 8)
        canvas.drawString(inch, 0.5 * inch, f"Page {doc.page}")
        canvas.drawRightString(doc.pagesize[0] - inch, 0.5 * inch, "Powered by ERAG")

        canvas.restoreState()


    def _markdown_to_reportlab(self, md_text, styles):
        elements = []
        try:
            md_text = re.sub(r'^```markdown\s*$', '', md_text, flags=re.MULTILINE)
            md_text = re.sub(r'^```\s*$', '', md_text, flags=re.MULTILINE)

            html = markdown.markdown(md_text)
            current_section = 'Normal'
            in_code_block = False
            code_block_content = []

            for line in html.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('<pre><code>'):
                    in_code_block = True
                    continue
                
                if in_code_block:
                    if line == '</code></pre>':
                        in_code_block = False
                        elements.append(Preformatted('\n'.join(code_block_content), styles['XDA_Code']))
                        code_block_content = []
                    else:
                        code_block_content.append(line)
                    continue

                if line.startswith('<h1>') or line.startswith('<h2>') or line.startswith('<h3>') or line.startswith('<h4>'):
                    heading = re.sub('<[^<]+?>', '', line)
                    heading = re.sub(r'^\d+\.?\s*', '', heading)
                    elements.append(Paragraph(heading, styles['XDA_Important']))
                elif line.lower().startswith('<p>limitations'):
                    elements.append(Paragraph(re.sub('<[^<]+?>', '', line), styles['XDA_Limitations']))
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
            print(f"Error processing markdown: {str(e)}")
            elements.append(Paragraph(md_text, styles['XDA_Normal']))

        # Add captions to images
        for i, element in enumerate(elements):
            if isinstance(element, Image):
                caption = elements[i-1].text if i > 0 and isinstance(elements[i-1], Paragraph) else "Figure"
                elements[i-1] = Paragraph(f"<b>{caption}</b>", styles['XDA_Caption'])

        return elements
