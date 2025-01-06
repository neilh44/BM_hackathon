from datetime import datetime
import logging
from markitdown import MarkItDown
import fitz
import pytesseract
from PIL import Image
import re
import os
from pathlib import Path

class PDFAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_pdf(self, pdf_path: str) -> tuple[str, float]:
        """
        Analyzes a PDF to determine if it's primarily text or images.
        Returns: ('text' or 'image', confidence_score)
        """
        try:
            doc = fitz.open(pdf_path)
            total_text_length = 0
            total_images = 0
            
            for page in doc:
                text = page.get_text()
                total_text_length += len(text)
                image_list = page.get_images()
                total_images += len(image_list)
                
            doc.close()
            
            text_density = total_text_length / (total_images + 1)
            
            if text_density > 500:
                confidence = min(text_density / 1000, 0.95)
                return 'text', confidence
            else:
                image_confidence = min(0.95, (total_images * 100) / (total_text_length + 1))
                return 'image', image_confidence
                
        except Exception as e:
            self.logger.error(f"Error analyzing PDF: {str(e)}")
            raise

class OCRProcessor:
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 6'
        self.logger = logging.getLogger(__name__)
        
    def convert_to_markdown(self, image_path: str) -> str:
        """Convert image to markdown text"""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            extracted_text = pytesseract.image_to_string(
                image,
                config=self.tesseract_config
            )
            
            return self._format_for_markdown(extracted_text)
            
        except Exception as e:
            self.logger.error(f"Error converting image to markdown: {str(e)}")
            raise
            
    def _format_for_markdown(self, text: str) -> str:
        """Format the extracted text for markdown"""
        lines = text.split('\n')
        markdown_text = []
        in_list = False
        
        for line in lines:
            clean_line = line.strip()
            
            if not clean_line:
                markdown_text.append('')
                in_list = False
                continue
            
            if re.match(r'^\d+\.', clean_line):
                markdown_text.append(clean_line)
                in_list = True
            elif clean_line.startswith('•') or clean_line.startswith('-'):
                markdown_text.append(f"* {clean_line[1:].strip()}")
                in_list = True
            else:
                if in_list:
                    markdown_text.append('')
                    in_list = False
                markdown_text.append(clean_line)
        
        return '\n'.join(markdown_text)

class PDFProcessor:
    def __init__(self):
        self.md = MarkItDown()
        self.analyzer = PDFAnalyzer()
        self.ocr_processor = OCRProcessor()
        
        logging.basicConfig(
            filename=f'pdf_processor_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def convert_to_markdown(self, pdf_path: str) -> str:
        """
        Convert PDF to markdown, automatically detecting if it needs OCR or text processing
        """
        try:
            # Analyze PDF content type
            content_type, confidence = self.analyzer.analyze_pdf(pdf_path)
            self.logger.info(f"PDF analyzed as {content_type} with {confidence:.2%} confidence")
            
            output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            if content_type == 'text':
                # Process text-based PDF
                result = self.md.convert(pdf_path)
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(result.text_content)
            else:
                # Process image-based PDF
                temp_dir = self._pdf_to_images(pdf_path)
                try:
                    markdown_contents = []
                    for image_file in sorted(os.listdir(temp_dir)):
                        if image_file.endswith(('.png', '.jpg')):
                            image_path = os.path.join(temp_dir, image_file)
                            markdown_content = self.ocr_processor.convert_to_markdown(image_path)
                            markdown_contents.append(markdown_content)
                    
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        f.write('\n\n'.join(markdown_contents))
                finally:
                    self._cleanup_temp_files(temp_dir)
            
            self.logger.info(f"Converted {pdf_path} to {output_filename}")
            return output_filename
            
        except Exception as e:
            self.logger.error(f"PDF conversion error: {str(e)}")
            raise

    def _pdf_to_images(self, pdf_path: str) -> str:
        """Convert PDF pages to images"""
        temp_dir = f"temp_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                pix.save(image_path)
            doc.close()
            return temp_dir
        except Exception as e:
            self._cleanup_temp_files(temp_dir)
            raise

    def _cleanup_temp_files(self, temp_dir: str):
        """Remove temporary files and directories"""
        try:
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        except Exception as e:
            self.logger.warning(f"Error cleaning up temporary files: {str(e)}")