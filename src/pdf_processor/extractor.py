import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from PyPDF2 import PdfReader
from tqdm import tqdm

class PDFExtractor:
    """
    Enhanced PDF text extraction with proper page tracking and content structuring.
    This class ensures that we maintain the connection between text and its source page
    throughout the extraction process.
    """
    
    def __init__(self):
        """Initialize the PDF extractor with logging configuration."""
        self.logger = logging.getLogger(__name__)

    def extract_text(self, pdf_path: str) -> List[Dict[str, Union[str, int]]]:
        """
        Extract text from PDF while maintaining page numbers and structure.
        Instead of returning just text, we return a list of dictionaries containing
        both the text and its metadata (like page numbers).
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Dict]: List of dictionaries containing:
                - text: The extracted text content
                - page_number: The page number where the text was found
                
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is empty or corrupted
        """
        self.logger.info(f"Starting text extraction from: {pdf_path}")
        
        # Validate PDF path
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Initialize storage for our extracted content
            extracted_content = []
            
            # Open and read PDF
            with open(pdf_file, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Validate PDF content
                if len(pdf_reader.pages) == 0:
                    self.logger.error("PDF file is empty")
                    raise ValueError("PDF file is empty")
                
                # Extract text from each page with progress bar
                for page_num in tqdm(range(len(pdf_reader.pages)), desc="Extracting pages"):
                    try:
                        # Get the page and extract its text
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        
                        # Only process pages with actual content
                        if text and text.strip():
                            # Clean and preprocess the text
                            processed_text = self._preprocess_text(text)
                            
                            # Store both text and page number
                            if processed_text:
                                extracted_content.append({
                                    'text': processed_text,
                                    'page_number': page_num + 1  # Use 1-based page numbering
                                })
                                
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
                
                if not extracted_content:
                    self.logger.error("No text content could be extracted from the PDF")
                    raise ValueError("No text content could be extracted from the PDF")
                
                self.logger.info(f"Successfully extracted {len(extracted_content)} pages of text")
                return extracted_content
                
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess extracted text to clean and normalize it.
        This ensures consistent text format across all extracted content.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        
        # Remove redundant whitespace while preserving paragraph structure
        text = ' '.join(text.split())
        
        # Fix common PDF extraction issues
        text = self._fix_common_issues(text)
        
        return text

    def _fix_common_issues(self, text: str) -> str:
        """
        Fix common issues in PDF-extracted text to improve quality.
        Handles special characters, broken words, and formatting issues.
        
        Args:
            text (str): Text to fix
            
        Returns:
            str: Fixed text
        """
        # Replace common problematic characters
        replacements = {
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\xa0': ' ',    # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Fix broken sentences (sentences split across lines)
        text = text.replace('- ', '')
        
        # Remove page numbers and headers/footers
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if not self._is_header_footer(line)]
        
        return '\n'.join(cleaned_lines)

    def _is_header_footer(self, line: str) -> bool:
        """
        Check if a line is likely a header or footer.
        This helps remove non-content elements from the extracted text.
        
        Args:
            line (str): Line of text to check
            
        Returns:
            bool: True if line appears to be header/footer
        """
        line = line.strip()
        
        if not line:
            return True
            
        # Check for page numbers
        if line.isdigit():
            return True
        
        # Check for common header/footer patterns
        common_patterns = [
            "project management",
            "chapter",
            "page",
            "©",
            "all rights reserved"
        ]
        
        return any(pattern in line.lower() for pattern in common_patterns)

    def get_metadata(self, pdf_path: str) -> dict:
        """
        Extract metadata from the PDF file.
        Provides additional information about the document.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: PDF metadata including title, author, etc.
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                metadata = pdf_reader.metadata
                return {
                    'title': metadata.get('/Title', 'Unknown'),
                    'author': metadata.get('/Author', 'Unknown'),
                    'subject': metadata.get('/Subject', 'Unknown'),
                    'creator': metadata.get('/Creator', 'Unknown'),
                    'page_count': len(pdf_reader.pages)
                }
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return {}
