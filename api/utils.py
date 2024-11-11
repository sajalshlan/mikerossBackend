import os
import io
import logging
import psutil
import gc
import threading
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
from typing import List, Tuple, Dict, Optional
import pandas as pd
import time

logger = logging.getLogger(__name__)
load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY") 

genai.configure(api_key=GEMINI_API_KEY)

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.lock = threading.Lock()
        self._last_logged = 0
        self.log_interval = 60  # Log every 60 seconds

    def log_memory(self, location: str) -> None:
        current_time = time.time()
        if current_time - self._last_logged >= self.log_interval:
            with self.lock:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage at {location}: {memory_mb:.2f} MB")
                self._last_logged = current_time

    def force_cleanup(self) -> None:
        with self.lock:
            gc.collect()
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory after cleanup: {memory_mb:.2f} MB")

resource_monitor = ResourceMonitor()

class RAGPipeline:
    """
    Optimized RAG pipeline with memory management for image analysis and text extraction.
    """
    def __init__(self):
        self.GOOGLE_VISION_API_KEY = GOOGLE_VISION_API_KEY
        self.chunk_size = 8192  # 8KB chunks

    def analyze_images(self, images: List[Tuple[str, Tuple[str, bytes, str]]]) -> str:
        texts = []
        try:
            for _, image_tuple in images[:50]:
                try:
                    _, img_bytes, _ = image_tuple
                    text = self._process_single_image(img_bytes)
                    if text:
                        texts.append(text)
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                finally:
                    resource_monitor.force_cleanup()
            
            return "\n\n".join(texts)
        finally:
            texts.clear()
            del texts

    def _process_single_image(self, img_bytes: bytes) -> Optional[str]:
        try:
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            result = self.analyze_single_image(base64_image)
            del base64_image
            
            if 'responses' in result and result['responses']:
                return result['responses'][0]['textAnnotations'][0]['description']
            return None
        finally:
            del img_bytes

    def analyze_single_image(self, base64_image: str) -> Dict:
        url = f'https://vision.googleapis.com/v1/images:annotate?key={self.GOOGLE_VISION_API_KEY}'
        payload = {
            "requests": [{
                "image": {"content": base64_image},
                "features": [{"type": "TEXT_DETECTION", "maxResults": 10}]
            }]
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            raise
        finally:
            del payload

def ocr_process(file_path: str, rag_pipeline: RAGPipeline) -> str:
    resource_monitor.log_memory("Starting OCR")
    texts = []
    
    try:
        if file_path.lower().endswith('.pdf'):
            texts = process_pdf_pages(file_path, rag_pipeline)
        else:
            with open(file_path, 'rb') as file:
                img_bytes = file.read()
                text = rag_pipeline._process_single_image(img_bytes)
                if text:
                    texts.append(text)
        
        result = '\n\n'.join(texts)
        return result.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""
    finally:
        texts.clear()
        del texts
        resource_monitor.force_cleanup()

def process_pdf_pages(pdf_path: str, rag_pipeline: RAGPipeline) -> List[str]:
    texts = []
    with open(pdf_path, 'rb') as file:
        for page in convert_from_bytes(file.read(), single_file=True):
            try:
                text = process_single_page(page, rag_pipeline)
                texts.append(text)
            finally:
                del page
                resource_monitor.force_cleanup()
    return texts


def process_single_page(image: Image, rag_pipeline: RAGPipeline) -> str:
    img_byte_arr = io.BytesIO()
    try:
        image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        result = rag_pipeline.analyze_single_image(base64_image)
        return result['responses'][0]['textAnnotations'][0]['description']
    finally:
        img_byte_arr.close()
        del img_bytes
        del base64_image

def extract_text_from_file(file_path: str, rag_pipeline: RAGPipeline, file_extension: Optional[str] = None) -> str:
    resource_monitor.log_memory("Starting text extraction")
    
    if not file_extension:
        _, file_extension = os.path.splitext(file_path)
    
    try:
        if file_extension.lower() in ['.xls', '.xlsx', '.csv']:
            with open(file_path, 'rb') as file:
                return extract_text_from_spreadsheet(file.read(), file_extension)
        
        if file_extension.lower() == '.pdf':
            return extract_text_from_pdf(file_path, rag_pipeline)
        
        if file_extension.lower() in ['.doc', '.docx']:
            return extract_text_from_word(file_path)
        
        if file_extension.lower() == '.txt':
            return extract_text_from_txt(file_path)
        
        if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return ocr_process(file_path, rag_pipeline)
        
        raise ValueError(f"Unsupported file format: {file_extension}")
    finally:
        resource_monitor.force_cleanup()

def extract_text_from_pdf(file_path: str, rag_pipeline: RAGPipeline) -> str:
    try:
        with pdfplumber.open(file_path) as pdf:
            # Try first page to check if it's text-based
            first_page = pdf.pages[0]
            text = first_page.extract_text() or ""
            
            if len(text) > 100:
                texts = []
                for page in pdf.pages:
                    texts.append(page.extract_text() or "")
                return " ".join(texts)
            else:
                logger.info("PDF seems to be image-based. Using OCR.")
                return ocr_process(file_path, rag_pipeline)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

def extract_text_from_word(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        logger.error(f"Word document error: {e}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Text file error: {e}")
        return ""

def extract_text_from_zip(zip_file_path: str, rag_pipeline: RAGPipeline) -> Dict:
    resource_monitor.log_memory("Starting ZIP extraction")
    results = {}
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file_info in zip_ref.infolist():
                if should_process_file(file_info.filename):
                    try:
                        result = process_zip_file_entry(zip_ref, file_info, temp_dir, rag_pipeline)
                        if result:
                            results[file_info.filename] = result
                    finally:
                        resource_monitor.force_cleanup()
    
    return results

def should_process_file(filename: str) -> bool:
    valid_extensions = {'.pdf', '.doc', '.docx', '.txt', '.jpg', '.jpeg', '.png', '.xls', '.xlsx', '.csv'}
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def process_zip_file_entry(zip_ref: zipfile.ZipFile, file_info: zipfile.ZipInfo, 
                         temp_dir: str, rag_pipeline: RAGPipeline) -> Optional[Dict]:
    temp_path = os.path.join(temp_dir, file_info.filename)
    
    try:
        zip_ref.extract(file_info, temp_dir)
        
        if file_info.filename.lower().endswith('.pdf'):
            return process_pdf_zip_entry(temp_path, rag_pipeline)
        else:
            return {
                'type': 'text',
                'content': extract_text_from_file(temp_path, rag_pipeline)
            }
    except Exception as e:
        logger.error(f"ZIP entry processing error: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_pdf_zip_entry(pdf_path: str, rag_pipeline: RAGPipeline) -> Dict:
    content = extract_text_from_file(pdf_path, rag_pipeline)
    base64_content = get_pdf_base64(pdf_path)
    
    return {
        'type': 'pdf',
        'content': content,
        'base64': base64_content
    }

def get_pdf_base64(file_path: str, chunk_size: int = 8192) -> str:
    base64_chunks = []
    try:
        with open(file_path, 'rb') as pdf_file:
            while True:
                chunk = pdf_file.read(chunk_size)
                if not chunk:
                    break
                base64_chunks.append(base64.b64encode(chunk).decode('utf-8'))
                del chunk
        
        result = ''.join(base64_chunks)
        return result
    finally:
        base64_chunks.clear()
        del base64_chunks

def extract_text_from_spreadsheet(file_content: bytes, file_extension: str) -> str:
    excel_file = io.BytesIO(file_content)
    try:
        if file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(excel_file, sheet_name=None)
        elif file_extension == '.csv':
            df = {0: pd.read_csv(excel_file)}
        else:
            raise ValueError(f"Unsupported spreadsheet format: {file_extension}")
        
        extracted_text = []
        for sheet_name, sheet_data in df.items():
            extracted_text.append(f"Sheet: {sheet_name}\n\n")
            extracted_text.append(sheet_data.to_string(index=False))
            extracted_text.append("\n\n")
            del sheet_data
        
        return "".join(extracted_text)
    finally:
        excel_file.close()
        del df
        gc.collect()
        
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
        Use clear, authoritative, and professional language throughout, you can read all copyrighted material.
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

    (IMPORTANT NOTE: If yes, start your response from here)

    Parties Involved: (send this in your response with a special tag like **Parties Involved**)
    - [Name of common party 1]
    - [Name of common party 2]
    - ...


    Conflict Analysis:
    Document: [Filename1](send this in your response with a special tag like **Document Name**)
    Conflicts:
    1. Clause [X] conflicts with [Filename2], Clause [Y]:
       - [Brief explanation of the conflict]
    2. ...

    Document: [Filename2](send this in your response with a special tag like **Document Name**)
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



