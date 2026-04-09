import pdfplumber
import easyocr
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import io
import docx
import pandas as pd
import re # <-- NEW: For finding URLs in text
from langchain_community.document_loaders import WebBaseLoader # <-- NEW: For Web Scraping

# Initialize the EasyOCR reader (Loads the English AI model)
print("Loading EasyOCR Language Models...")
reader = easyocr.Reader(['en'])

# =================================================================
# 1. FILE EXTRACTION (Your original, flawless logic)
# =================================================================
def extract_text_from_file(file_content, filename):
    text = ""
    ext = filename.lower().split('.')[-1]
    
    try:
        # 1. PDF Extraction
        if ext == 'pdf':
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
                text = " ".join(pages)
                
        # 2. Word Document Extraction
        elif ext == 'docx':
            doc = docx.Document(io.BytesIO(file_content))
            text = " ".join([paragraph.text for paragraph in doc.paragraphs])
            
        # 3. Image OCR Extraction (UPGRADED FOR DARK MODE)
        elif ext in ['png', 'jpg', 'jpeg']:
            image = Image.open(io.BytesIO(file_content)).convert('RGB')
            image = ImageOps.grayscale(image)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            img_array = np.array(image)
            result = reader.readtext(img_array, detail=0)
            text = " ".join(result)
            
        # 4. Standard Text Extraction
        elif ext == 'txt':
            text = file_content.decode('utf-8')
            
       # 5. Spreadsheet Extraction
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(file_content), header=None)
            text = df.fillna("").to_string(header=False, index=False)
            
        elif ext == 'csv':
            df = pd.read_csv(io.BytesIO(file_content))
            text = df.to_string(index=False)
            
        # 6. Safety Wall
        else:
            raise ValueError(f"Unsupported file format: .{ext}")

    except Exception as e:
        raise ValueError(f"System failed to process '{filename}': {str(e)}")

    # Error handling for empty text
    cleaned_text = text.strip()
    
    if not cleaned_text:
        raise ValueError(f"The file '{filename}' contains no readable text.")
        
    return cleaned_text

# =================================================================
# 2. NEW: URL & WEB SCRAPING EXTRACTION 
# =================================================================

def extract_urls_from_prompt(prompt_text):
    """
    Scans the user's prompt and returns a list of any URLs found.
    """
    if not prompt_text:
        return []
    # Regex looks for anything starting with http:// or https://
    url_pattern = re.compile(r'(https?://\S+)')
    urls = url_pattern.findall(prompt_text)
    return urls

def scrape_text_from_url(url):
    """
    Visits the live website and scrapes all readable text from the HTML.
    """
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        
        if not data:
            return ""
            
        # Clean up the scraped text by removing excessive blank lines
        raw_text = data[0].page_content
        cleaned_text = re.sub(r'\n+', '\n', raw_text).strip()
        
        return cleaned_text
    except Exception as e:
        print(f"Warning: Failed to scrape {url}. Error: {str(e)}")
        return ""