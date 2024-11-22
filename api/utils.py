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
from .prompts import (
    DOCUMENT_TYPES,
    GENERAL_SHORT_SUMMARY_PROMPT,
    GENERAL_LONG_SUMMARY_PROMPT,
    GENERAL_RISK_ANALYSIS_PROMPT,
    SHORT_SUMMARY_PROMPTS,
    RISK_ANALYSIS_PROMPTS,
    CONFLICT_ANALYSIS_PROMPT,
    LONG_SUMMARY_PROMPTS,
    DRAFT_PROMPT,
    ASK_PROMPT
)
import anthropic  # Add this import at the top

logger = logging.getLogger(__name__)
load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY") 

genai.configure(api_key=GEMINI_API_KEY)

# Add these at the top level with other constants



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
        
def classify_document(text: str) -> str:
    classification_prompt = """
    You are a legal document classifier. Based on the document provided, classify it into ONE of the following categories:
    
    {}
    
    Respond ONLY with the exact category name from the list above. If the document doesn't match any category exactly, 
    choose the closest match. Provide ONLY the category name, no other text or explanation.
    """.format("\n".join(DOCUMENT_TYPES))
    
    try:
        result = claude_call(text, classification_prompt)
        classified_type = result.strip()
        if classified_type in DOCUMENT_TYPES:
            return classified_type
        # If response doesn't match exactly, return None to trigger general prompt
        logger.warning(f"Invalid classification result: {classified_type}")
        return None
    except Exception as e:
        logger.error(f"Error in document classification: {e}")
        return None

def perform_analysis(analysis_type: str, text: str, custom_prompt: str = None, use_gemini: bool = True) -> str:
    logger.info(f"Performing analysis: {analysis_type}")
    print(f"[Analysis] 🔍 Custom prompt provided: {bool(custom_prompt)}")
    
    # Always classify the document first
    doc_type = None
    if analysis_type in ['shortSummary', 'longSummary', 'risky']:
        doc_type = classify_document(text[:100])
        print(f"{analysis_type}: classified into {doc_type}")
        logger.info(f"Document classified as: {doc_type}")

    if custom_prompt:
        print(f"[Analysis] 📋 Custom prompt content (first 100 chars): {custom_prompt[:100]}...")
        prompt = custom_prompt
    else:
        # Your existing prompt selection logic
        if analysis_type == 'shortSummary':
            if doc_type and doc_type in SHORT_SUMMARY_PROMPTS:
                prompt = f"""
                Provide a comprehensive summary of the document. 
                {SHORT_SUMMARY_PROMPTS[doc_type]}
                Do not mention about the framework. Follow the framework strictly.
                """
            else:
                logger.info("Using general summary prompt for short summary")
                prompt = GENERAL_SHORT_SUMMARY_PROMPT
        
        elif analysis_type == 'longSummary':
            if doc_type and doc_type in LONG_SUMMARY_PROMPTS:
                prompt = f"""
                Provide a comprehensive summary of the document. 
                {LONG_SUMMARY_PROMPTS[doc_type]}
                Do not mention about the framework. Follow the framework strictly.
                """
            else:
                logger.info("Using general summary prompt for long summary")
                prompt = GENERAL_LONG_SUMMARY_PROMPT
        
        elif analysis_type == 'risky':
            if doc_type and doc_type in RISK_ANALYSIS_PROMPTS:
                logger.info(f"Document classified as: {doc_type}")
                risk_content = RISK_ANALYSIS_PROMPTS.get(doc_type, RISK_ANALYSIS_PROMPTS)
            else:
                logger.info("Using general risk analysis prompt")
                risk_content = GENERAL_RISK_ANALYSIS_PROMPT
            
            prompt = f"""
            You are a General Counsel of a Fortune 500 company with over 20 years of experience. 
            First, thoroughly analyze this {doc_type} for all potential risks using the following framework:
            Structure your response in the following specific format:

            OUTPUT STRUCTURE:
            1. First, identify all parties silently (do not list them in the output)
            2. For each party, present the risks they are exposed to using this format.
            3. For each risk, provide a detailed analysis of the impact on the party's interests.
            4. Keep each party's analysis separate, independent and distinct from the other parties'.
            5. Give fresh numbers to each party's analysis while formatting.

            *****[PARTY NAME]*****
            PERSPECTIVE: Brief overview of this party's position and key objectives in the agreement

            **RISK EXPOSURE**
            
            {risk_content}


            [Repeat for each party]

            IMPORTANT FORMATTING RULES:
            - Use ***** only for party names
            - For section headers, use **
            - Replace the A, B, C with the **
            - Include specific clause references in [square brackets]
            - Maintain professional, clear language
            """

            try:
                result = gemini_call(text, prompt)
                logger.info("Risk analysis completed successfully")
                return result
            except Exception as e:
                logger.exception("Error in risk analysis")
                raise
        
        elif analysis_type == 'ask':
            prompt = ASK_PROMPT
        
        elif analysis_type == 'draft':
            prompt = DRAFT_PROMPT
        else:
            logger.error(f"Invalid analysis type: {analysis_type}")
            raise ValueError(f"Invalid analysis type: {analysis_type}")
        
        logger.info(f"Using default prompt for {analysis_type}")

    print(f"\n[Analysis] 🚀 Final prompt being sent to Gemini (first 100 chars): {prompt[:100]}...")


    try:
        if use_gemini:
            result = gemini_call(text, prompt)
        else:
            result = claude_call(text, prompt)
        return result
    except Exception as e:
        print(f"[Analysis] ❌ Error calling API: {str(e)}")
        logger.exception("Error calling API")
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
        return result.strip().lower() == 'yes'
    except Exception as e:
        logger.exception("Error checking for common party")
        return False
    
def gemini_call(text, prompt):
    print("[Analysis] ⏳ Calling Gemini API...")
    
    system_prompt = """You are a highly experienced General Counsel of a Fortune 500 company with over 20 years of experience in corporate law."""

    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(
            [system_prompt, text, prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Slightly increased for more natural language while maintaining precision
                max_output_tokens=4000,  # Increased token limit for more comprehensive responses
                top_p=0.8,  # Added for better response quality
                top_k=40,  # Added for better response diversity while maintaining relevance
            )
        )
        logger.info("Gemini API call successful")
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise Exception(f"An error occurred while calling Gemini API: {e}")

def claude_call(text, prompt):
    print("[Analysis] ⏳ Calling Claude API...")
    
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 3000,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": f"{text}\n\n{prompt}"}
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        logger.info("Claude API call successful")
        return response.json()["content"][0]["text"]
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        raise Exception(f"An error occurred while calling Claude API: {e}")

def analyze_conflicts_and_common_parties(texts: Dict[str, str]) -> str:
    logger.info("Analyzing conflicts and common parties")
    
    prompt = CONFLICT_ANALYSIS_PROMPT

    for filename, content in texts.items():
        prompt += f"\n\nFilename: {filename}\nContent (truncated):\n{content}"

    try:
        result = gemini_call("", prompt)
        logger.info("Gemini API call for conflict and common party analysis successful")
        return result
    except Exception as e:
        logger.exception("Error analyzing conflicts and common parties")
        raise Exception(f"An error occurred while analyzing conflicts and common parties: {e}")



