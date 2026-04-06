import pdfplumber

import easyocr

from PIL import Image, ImageOps, ImageEnhance

import numpy as np

import io

import docx

import pandas as pd



# Initialize the EasyOCR reader (Loads the English AI model)

print("Loading EasyOCR Language Models...")

reader = easyocr.Reader(['en'])



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

            # Open the image and convert to RGB

            image = Image.open(io.BytesIO(file_content)).convert('RGB')

           

            # Pre-processing to help EasyOCR read Dark Mode or low contrast

            image = ImageOps.grayscale(image)

            enhancer = ImageEnhance.Contrast(image)

            image = enhancer.enhance(2.0) # Boost contrast by 2x

           

            # Convert the PIL image to a numpy array (EasyOCR requires this for processed images)

            img_array = np.array(image)

           

            # Read the text

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

       

    # 👇 FIX: Just return the text string here so it doesn't break your other files!

    return cleaned_text

