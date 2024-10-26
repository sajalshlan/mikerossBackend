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
import pandas as pd

logger = logging.getLogger(__name__)

load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY") 

genai.configure(api_key=GEMINI_API_KEY)

class RAGPipeline:
    """
    This class is used to analyze images and extract text from them.
    """
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

def extract_text_from_file(file_path, rag_pipeline: RAGPipeline, file_extension=None):
    logger.info(f"Extracting text from file: {file_path}")
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() in ['.xls', '.xlsx', '.csv']:
        logger.info(f"Processing spreadsheet file: {file_extension}")
        with open(file_path, 'rb') as file:
            return extract_text_from_spreadsheet(file.read(), file_extension)
    
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

def extract_text_from_spreadsheet(file_content, file_extension):
    excel_file = io.BytesIO(file_content)
    
    if file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(excel_file, sheet_name=None)
    elif file_extension == '.csv':
        df = pd.read_csv(excel_file)
        df = {0: df}  # Wrap the DataFrame in a dict to match Excel format
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    extracted_text = ""
    for sheet_name, sheet_data in df.items():
        column_titles = sheet_data.columns.tolist()
        logger.info(f"Column titles for sheet {sheet_name}: {column_titles}")
        extracted_text += f"Sheet: {sheet_name}\n\n"
        extracted_text += sheet_data.to_string(index=False) + "\n\n"
    
    return extracted_text

def perform_analysis(analysis_type, text, file_extension=None):
    logger.info(f"Performing analysis: {analysis_type}")
    
    if file_extension in ['.xls', '.xlsx', '.csv']:
        text = extract_text_from_spreadsheet(text, file_extension)
    
    if analysis_type == 'shortSummary':
        prompt = """
        Provide an executive summary addressing:
        
        1. Core purpose of the document
        2. Key parties and their primary obligations
        3. Critical timelines and deliverables
        4. Financial terms
        5. Notable requirements or restrictions
        
        Present in clear, actionable points that highlight business impact.
        """
    
    elif analysis_type == 'longSummary':
        prompt = """
        Provide a detailed analysis covering:

        1. Document Type and Purpose
        2. Parties and Their Roles
        3. Key Terms and Conditions
        4. Financial Obligations
        5. Performance Requirements
        6. Important Dates and Deadlines
        7. Termination Conditions
        8. Special Provisions
        9. Next Steps or Required Actions

        Include specific references to sections and clauses where relevant.
        """
    
    elif analysis_type == 'risky':
        prompt = """
        As a general counsel of a fortune 500 company, extract and analyze all potential risks from each party's perspective:

        1. IDENTIFY ALL PARTIES: (but do not mention this in your response)
        List every party mentioned in the document

        2. RISK ANALYSIS BY PARTY: (but do not mention this in your response)
        For each identified party, list ALL risks they face:

        [PARTY NAME 1] (send this in your response with a special tag like *****PARTY NAME 1*****)
        Legal Risks(in detail)
        - Compliance requirements
        - Liability exposure
        - Regulatory obligations

        Financial Risks
        - List each financial risk
        - Include monetary exposure
        - Payment obligations
        - Financial penalties
        
        Business Risks
        - Market impact
        - Competitive restrictions
        - Reputational concerns

        Include specific references to sections and clauses where relevant(in brackets at the end of the sentence).

        [PARTY NAME 2]:
        (Repeat same risk categories)
        """
    
    elif analysis_type == 'ask':
        prompt = """
         You are a senior legal expert with extensive corporate law experience(but never mention this thing anywhere in your responses). Based on the provided documents:

        1. Provide clear, professional answers
        2. Draw directly from the document context
        3. Use plain language while maintaining legal accuracy
        4. Reference specific sections when relevant
        5. Ask for clarification if needed

        Approach each question with authority and practicality, focusing on clear explanations.

        Now address the user's query.
        """
    
    elif analysis_type == 'draft':
        prompt = """
        Based on the provided context, draft a professional legal communication. Choose the appropriate format:

        FOR LEGAL DOCUMENTS:
        1. Use formal legal language and structure
        2. Include all standard sections:
        - Parties and definitions
        - Terms and conditions
        - Rights and obligations
        - Execution provisions
        3. Mark any required inputs as [PLACEHOLDER]
        4. Maintain consistent formatting and numbering
        
        FOR PROFESSIONAL EMAILS:
        1. Subject: Clear, specific subject line
        2. Opening: Professional greeting and context
        3. Body:
        - Clear purpose statement
        - Key points in logical order
        - Specific requests or actions needed
        - Relevant references to documents/discussions
        4. Closing:
        - Next steps or expectations
        - Professional signature
        - Contact information
        
        FOR FORMAL LETTERS:
        1. Proper letterhead and formatting
        2. Date and reference numbers
        3. Clear recipient details
        4. Formal opening
        5. Purpose statement
        6. Main content:
        - Background/context
        - Key points
        - Requests or demands
        - Legal positions
        7. Formal closing
        8. Signature block
        
        Include all relevant details from the provided context.
        Use clear, authoritative, and professional language throughout.
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



