import os
import io
import logging
import anthropic
from dotenv import load_dotenv
import pdfplumber
import docx
import base64
import requests
import zipfile
from pdf2image import convert_from_bytes
import google.generativeai as genai
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY") 

genai.configure(api_key=GEMINI_API_KEY)

class RAGPipeline:
    def __init__(self):
        self.GOOGLE_VISION_API_KEY = GOOGLE_VISION_API_KEY

    def analyze_images(self, images: List[Tuple[str, Tuple[str, bytes, str]]]) -> str:
        texts = []
        for _, image_tuple in images[:25]:  # Limit to 25 pages for OCR
            try:
                _, img_bytes, _ = image_tuple
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                result = self.analyze_single_image(base64_image)
                text = result['responses'][0]['textAnnotations'][0]['description']
                texts.append(text)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
        return "\n\n".join(texts)

    def analyze_single_image(self, base64_image: str) -> Dict:
        url = f'https://vision.googleapis.com/v1/images:annotate?key={self.GOOGLE_VISION_API_KEY}'
        payload = {
            "requests": [
                {
                    "image": {
                        "content": base64_image
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION",
                            "maxResults": 10
                        }
                    ]
                }
            ]
        }
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
        return response.json()

def extract_text_from_file(file_path, rag_pipeline: RAGPipeline):
    logger.info(f"Extracting text from file: {file_path}")
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        logger.info("Processing PDF file")
        try:
            with pdfplumber.open(file_path) as pdf:
                first_page = pdf.pages[0]
                text = first_page.extract_text() or ""
                if len(text) > 100:
                    return " ".join([page.extract_text() or "" for page in pdf.pages])
                else:
                    logger.info("PDF seems to be image-based. Attempting OCR.")
                    with open(file_path, 'rb') as file:
                        images = convert_from_bytes(file.read())
                    image_files = []
                    for i, image in enumerate(images[:25]):  # Limit to 25 pages for OCR
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        image_files.append(('image', ('image.jpg', img_byte_arr, 'image/jpeg')))
                    
                    ocr_text = rag_pipeline.analyze_images(image_files)
                    return ocr_text
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return ""
    elif file_extension.lower() in ['.doc', '.docx']:
        logger.info("Processing Word document")
        doc = docx.Document(file_path)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    elif file_extension.lower() == '.txt':
        logger.info("Processing text file")
        with open(file_path, 'r') as file:
            return file.read()
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}")

def extract_text_from_zip(zip_file_path, rag_pipeline: RAGPipeline):
    logger.info(f"Extracting text from ZIP file: {zip_file_path}")
    extracted_texts = {}
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            try:   
                with zip_ref.open(file_name) as file:
                    file_like_object = io.BytesIO(file.read())
                    file_like_object.name = file_name  # Add name attribute
                    text = extract_text_from_file(file_like_object, rag_pipeline)
                    if text.strip():
                        extracted_texts[file_name] = text
                        logger.info(f"Successfully extracted text from {file_name}")
                    else:
                        logger.warning(f"No text could be extracted from {file_name}")
                        extracted_texts[file_name] = "No text could be extracted"
            except Exception as e:
                logger.error(f"Error processing {file_name} from ZIP: {str(e)}")
                extracted_texts[file_name] = f"Error: {str(e)}"
    return extracted_texts

# def perform_analysis(analysis_type, text):
#     logger.info(f"Performing analysis: {analysis_type}")
    
#     if analysis_type == 'summary':
#         prompt = "Provide a brief summary of this document."
#     elif analysis_type == 'risky':
#         prompt = """
#         Analyze the document and identify potentially risky clauses or terms. For each risky clause:
#         1. Start with the actual clause number as it appears in the document.
#         2. Quote the relevant part of the clause.
#         3. Explain why it's potentially risky.

#         Format your response as follows:

#         Clause [X]: "[Quote the relevant part]"
#         Risk: [Explain the potential risk]

#         Where [X] is the actual clause number from the document.
#         IF NO CLAUSE NUMBER IS PRESENT IN THE DOCUMENT, DO NOT GIVE ANY NUMBER TO THE CLAUSE BY YOURSELF THEN.
#         """
#     elif analysis_type == 'conflict':
#         prompt = """
#         Perform a conflict check across all the provided documents. For each document, identify any clauses or terms that may conflict with clauses or terms in the other documents. 

#         Provide your analysis in the following format:

#         Document: [Filename1]
#         Conflicts:
#         1. Clause [X] conflicts with [Filename2], Clause [Y]:
#            - [Brief explanation of the conflict]
#         2. ...

#         Document: [Filename2]
#         Conflicts:
#         1. ...

#         If no conflicts are found for a document, state "No conflicts found."

#         Focus on significant conflicts that could impact the legal or business relationship between the parties involved.
#         """
#     elif analysis_type == 'structure':
#         prompt = """
#         Analyze the structure of the document and provide a summary of its organization. Include the following:

#         1. A brief outline of the document's flow and how information is presented
#         2. Main sections or parts of the document
#         3. Any notable formatting or structural elements (e.g., numbered clauses, appendices)

#         Format your response as a concise summary, highlighting the key structural elements of the document.
#         """
#     elif analysis_type == 'ask':
#         prompt = "You are an AI assistant. Please provide a response to the user's query based on the given document content."
#     else:
#         logger.error(f"Invalid analysis type: {analysis_type}")
#         raise ValueError(f"Invalid analysis type: {analysis_type}")

#     logger.info(f"Using prompt: {prompt}")
    
#     try:
#         result = claude_call(text, prompt)
#         logger.info("Claude API call successful")
#         return result
#     except Exception as e:
#         logger.exception("Error calling Claude API")
#         raise

# def claude_call(text, prompt):
#     logger.info("Calling Claude API")
#     client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

#     try:
#         message = client.messages.create(
#             model="claude-3-5-sonnet-20240620",
#             max_tokens=3000,
#             temperature=0,
#             system="You are a general counsel of a fortune 500 company.",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": text,
#                         },
#                         {
#                             "type": "text",
#                             "text": prompt
#                         },
#                     ],
#                 }
#             ],
#         )
#         logger.info("Claude API call successful")
#         return ''.join(block.text for block in message.content if block.type == 'text')
#     except Exception as e:
#         logger.error(f"Error calling Claude API: {str(e)}")
#         raise Exception(f"An error occurred while calling Claude API: {e}")
    

def perform_analysis(analysis_type, text):
    logger.info(f"Performing analysis: {analysis_type}")
    
    if analysis_type == 'summary':
        prompt = "Provide a brief summary of this document."
    elif analysis_type == 'risky':
        prompt = """
        Analyze the document and identify potentially risky clauses or terms. For each risky clause:
        1. Start with the actual clause number as it appears in the document.
        2. Quote the relevant part of the clause.
        3. Explain why it's potentially risky.

        Format your response as follows:

        Clause [X]: "[Quote the relevant part]"
        Risk: [Explain the potential risk]

        Where [X] is the actual clause number from the document.
        IF NO CLAUSE NUMBER IS PRESENT IN THE DOCUMENT, DO NOT GIVE ANY NUMBER TO THE CLAUSE BY YOURSELF THEN.
        """
    elif analysis_type == 'conflict':
        prompt = """
        Perform a conflict check across all the provided documents. For each document, identify any clauses or terms that may conflict with clauses or terms in the other documents. 

        Provide your analysis in the following format:

        Document: [Filename1]
        Conflicts:
        1. Clause [X] conflicts with [Filename2], Clause [Y]:
           - [Brief explanation of the conflict]
        2. ...

        Document: [Filename2]
        Conflicts:
        1. ...

        If no conflicts are found for a document, state "No conflicts found."

        Focus on significant conflicts that could impact the legal or business relationship between the parties involved.
        """
    elif analysis_type == 'structure':
        prompt = """
        Analyze the structure of the document and provide a summary of its organization. Include the following:

        1. A brief outline of the document's flow and how information is presented
        2. Main sections or parts of the document
        3. Any notable formatting or structural elements (e.g., numbered clauses, appendices)

        Format your response as a concise summary, highlighting the key structural elements of the document.
        """
    elif analysis_type == 'ask':
        prompt = "You are an AI assistant. Please provide a response to the user's query based on the given document content."
    else:
        logger.error(f"Invalid analysis type: {analysis_type}")
        raise ValueError(f"Invalid analysis type: {analysis_type}")

    logger.info(f"Using prompt: {prompt}")
    
    try:
        result = gemini_call(text, prompt)
        logger.info("Gemini API call successful")
        return result
    except Exception as e:
        logger.exception("Error calling Gemini API")
        raise

def gemini_call(text, prompt):
    logger.info("Calling Gemini API")
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(
            [text, prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=3000,
            )
        )
        logger.info("Gemini API call successful")
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise Exception(f"An error occurred while calling Gemini API: {e}")