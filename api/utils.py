import os
import io
import logging
from dotenv import load_dotenv
import pdfplumber
import docx
import base64
import tempfile
import requests
import zipfile
from pdf2image import convert_from_bytes
import google.generativeai as genai
from PIL import Image
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
        for _, image_tuple in images[:50]:  # Limit to 50 pages for OCR
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

def ocr_process(file_path, rag_pipeline):
    try:
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as file:
                images = convert_from_bytes(file.read())
        else:
            images = [Image.open(file_path)]
        
        image_files = []
        for i, image in enumerate(images[:50]):  # Limit to 50 pages/images for OCR
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            image_files.append(('image', ('image.jpg', img_byte_arr, 'image/jpeg')))
        
        ocr_text = rag_pipeline.analyze_images(image_files)
        return ocr_text.encode('utf-8', errors='ignore').decode('utf-8')

    except Exception as e:
        logger.error(f"Error processing image with OCR: {e}")
        return ""

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
                    return ocr_process(file_path, rag_pipeline)
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return ""
    elif file_extension.lower() in ['.doc', '.docx']:
        logger.info("Processing Word document")
        try:
            doc = docx.Document(file_path)
            return " ".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error processing Word document: {e}")
            return ""
    elif file_extension.lower() == '.txt':
        logger.info("Processing text file")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            return ""
    elif file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        logger.info(f"Processing image file: {file_extension}")
        return ocr_process(file_path, rag_pipeline)
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}")

def extract_text_from_zip(zip_file_path, rag_pipeline: RAGPipeline):
    logger.info(f"Extracting text from ZIP file: {zip_file_path}")
    extracted_contents = {}
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Create a temporary directory to extract files
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_ref.extractall(temp_dir)
            for root, _, files in os.walk(temp_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    try:
                        extracted_text = extract_text_from_file(file_path, rag_pipeline)
                        if file_name.lower().endswith('.pdf'):
                            with open(file_path, 'rb') as pdf_file:
                                pdf_content = pdf_file.read()
                                extracted_contents[file_name] = {
                                    'type': 'pdf',
                                    'content': extracted_text,
                                    'base64': base64.b64encode(pdf_content).decode('utf-8')
                                }
                        else:
                            extracted_contents[file_name] = {
                                'type': 'text',
                                'content': extracted_text
                            }
                        logger.info(f"Successfully processed {file_name}")
                    except Exception as e:
                        logger.error(f"Error processing {file_name} from ZIP: {str(e)}")
                        extracted_contents[file_name] = {
                            'type': 'error',
                            'content': f"Error: {str(e)}"
                        }
    return extracted_contents

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
    elif analysis_type == 'ask':
        prompt = "You are an AI assistant. Please provide a response to the user's query based on the given document content."
    elif analysis_type == 'draft':
        prompt = """
        Based on the provided document content, create a professional legal draft. Follow these guidelines:

        1. Maintain a formal and precise legal language.
        2. Include all necessary sections typically found in this type of legal document (e.g., definitions, parties involved, terms and conditions, etc.).
        3. Ensure the draft is well-structured with clear headings and subheadings.
        4. Include any specific clauses or terms mentioned in the original content.
        5. If any information is missing or unclear, use placeholders like [PARTY A] or [SPECIFIC DATE] to indicate where additional information is needed.

        Begin the draft with an appropriate title and continue with the full content of the legal document.
        """
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
    
def has_common_party(texts):
    if len(texts) < 2:
        return False

    prompt = """
    Analyze the following documents and determine if there is at least one common party present in all of them. 
    Answer with only 'Yes' if there is at least one common party across all documents, or 'No' if there isn't.

    Documents:
    """

    for filename, content in texts.items():
        prompt += f"\n\nFilename: {filename}\nContent (truncated):\n{content[:10000]}"

    prompt += "\n\nIs there at least one common party present in all of the above documents? Answer with only 'Yes' or 'No'."

    try:
        result = gemini_call("", prompt)
        logger.info(f"Gemini API response for common party check: {result}")
        print(result.strip().lower())
        return result.strip().lower() == 'yes'
    except Exception as e:
        logger.exception("Error checking for common party")
        return False
    

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

def analyze_conflicts_and_common_parties(texts: Dict[str, str]) -> str:
    logger.info("Analyzing conflicts and common parties")
    
    prompt = """
    Analyze the following documents for two tasks:
    1. Determine if there is at least one common party present in all documents.
    2. If there is at least one common party, perform a conflict check across all documents.

    For each document, identify any clauses or terms that may conflict with clauses or terms in the other documents.

    Provide your analysis in the following format:
    Common Party Check:
    [Yes/No] - There [is/are] [a common party/no common parties] involved across the selected documents.

    If the answer is Yes, continue with:

    Common Parties:
    - [Name of common party 1]
    - [Name of common party 2]
    - ...


    Conflict Analysis:
    Document: [Filename1]
    Conflicts:
    1. Clause [X] conflicts with [Filename2], Clause [Y]:
       - [Brief explanation of the conflict]
    2. ...

    Document: [Filename2]
    Conflicts:
    1. ...

    If no conflicts are found for a document, state "No conflicts found."

    If there is no common party, only provide the Common Party Check result.

    Focus on significant conflicts that could impact the legal or business relationship between the parties involved.

    Documents:
    """

    for filename, content in texts.items():
        prompt += f"\n\nFilename: {filename}\nContent (truncated):\n{content}"

    try:
        result = gemini_call("", prompt)
        logger.info("Gemini API call for conflict and common party analysis successful")
        return result
    except Exception as e:
        logger.exception("Error analyzing conflicts and common parties")
        raise Exception(f"An error occurred while analyzing conflicts and common parties: {e}")
