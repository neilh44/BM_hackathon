from datetime import datetime
import logging
from markitdown import MarkItDown

class PDFProcessor:
    def __init__(self):
        self.md = MarkItDown()
        logging.basicConfig(
            filename=f'pdf_processor_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def convert_to_markdown(self, pdf_path):
        try:
            result = self.md.convert(pdf_path)
            output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(result.text_content)
            
            logging.info(f"Converted {pdf_path} to {output_filename}")
            return output_filename
        except Exception as e:
            logging.error(f"PDF conversion error: {str(e)}")
            raise

