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
import json
from pdf2image import convert_from_bytes
import google.generativeai as genai
from PIL import Image
from typing import List, Tuple, Dict, Optional
import pandas as pd
import anthropic 
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
    ASK_PROMPT,
)
from pdf2docx import Converter

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
            for _, image_tuple in images:
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
        except Exception as e:
            return ""

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

def process_pdf_pages(pdf_path: str, rag_pipeline: RAGPipeline) -> List[str]:
    texts = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_data = file.read()
            all_pages = convert_from_bytes(pdf_data)
            total_pages = len(all_pages)
            del all_pages
            
            for page_num in range(1, total_pages + 1):
                try:
                    pages = convert_from_bytes(
                        pdf_data, 
                        first_page=page_num, 
                        last_page=page_num,
                        single_file=False
                    )
                    if pages:
                        text = process_single_page(pages[0], rag_pipeline)
                        if text:
                            texts.append(text)
                finally:
                    del pages
                    resource_monitor.force_cleanup()
        return texts
    except Exception as e:
        logger.error(f"Error processing PDF pages: {e}")
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

def get_pdf_base64(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding PDF to base64: {e}")
        raise

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
    choose the "None of the above" category present at the end of the list. Provide ONLY the category name, no other text or explanation.
    """.format("\n".join(DOCUMENT_TYPES))
    try:
        result = claude_call_haiku(text, classification_prompt)
        classified_type = result.strip()
        print(f"classified_type: {classified_type}")
        if classified_type in DOCUMENT_TYPES:
            return classified_type
        # If response doesn't match exactly, return None to trigger general prompt
        logger.warning(f"Invalid classification result: {classified_type}")
        return None
    except Exception as e:
        logger.error(f"Error in document classification: {e}")
        return None

def classify_document_new(text: str) -> str:
    classification_prompt = """
    You are a legal document classifier. Based on the document provided, classify it into ONE of the following categories:
    
    {}
    
    Consider the document's primary purpose, structure, and content. Respond ONLY with the exact category name from the list above. 
    Provide ONLY the category name, no other text or explanation.
    """.format("\n".join(DOCUMENT_CATEGORIES))

    try:
        result = claude_call(text, classification_prompt)
        classified_type = result.strip()
        if classified_type in DOCUMENT_CATEGORIES:
            print(f"classified_type: {classified_type}")
            return classified_type
        logger.warning(f"Invalid classification result: {classified_type}")
        return None
    except Exception as e:
        logger.error(f"Error in document classification: {e}")
        return None

def perform_analysis(analysis_type: str, text: str, file_extension=None) -> str:
    """
    Modified to include document classification for certain analysis types
    """
    logger.info(f"Performing analysis: {analysis_type}")
    
    if file_extension in ['.xls', '.xlsx', '.csv']:
        text = extract_text_from_spreadsheet(text, file_extension)
    
    if analysis_type == 'explain':
        prompt = text 
        logger.info("Using explanation prompt")
    
    elif analysis_type == 'shortSummary':
        # First classify the document
        doc_type = classify_document(text[:1000])

        logger.info(f"Document classified as: {doc_type}")
        
        if doc_type and doc_type in SHORT_SUMMARY_PROMPTS:
            prompt = f"""
            {SHORT_SUMMARY_PROMPTS[doc_type]}
           Important: Provide your analysis directly without any disclaimers or comments about document classification or type mismatches. Focus only on analyzing the actual content and terms of the document.

           Provide only the analysis, no other text or comments.
            """
        else:
            logger.info("Using general summary prompt for short summary")
            prompt = GENERAL_SHORT_SUMMARY_PROMPT
    
    elif analysis_type == 'longSummary':
        doc_type = classify_document(text[:100])
        print(f"long summary: classified into {doc_type}")
        logger.info(f"Document classified as: {doc_type}")

        if doc_type and doc_type in LONG_SUMMARY_PROMPTS:
            prompt = f"""
            {LONG_SUMMARY_PROMPTS[doc_type]}
            Important: Provide your analysis directly without any disclaimers or comments about document classification or type mismatches. Focus only on analyzing the actual content and terms of the document.

            Provide only the analysis, no other text or comments.
            """
        else:
            logger.info("Using general summary prompt for long summary")
            prompt = GENERAL_LONG_SUMMARY_PROMPT
    
    elif analysis_type == 'risky':
        doc_type = classify_document(text[:100])
        print(f"risky: classified into {doc_type}")

        if doc_type and doc_type in RISK_ANALYSIS_PROMPTS:
            logger.info(f"Document classified as: {doc_type}")
            risk_content = RISK_ANALYSIS_PROMPTS.get(doc_type, RISK_ANALYSIS_PROMPTS)
        else:
            logger.info("Using general risk analysis prompt")
            risk_content = GENERAL_RISK_ANALYSIS_PROMPT
        
        prompt = f"""
        First, thoroughly analyze this {doc_type} for all potential risks using the following framework:

        IMPORTANT REFERENCE RULES:
        1. When referencing specific clauses or sections, always include the actual text content (first 50-70 characters) within [[double brackets]], not the clause numbers.
        2. Add the filename after each citation using {{{{filename}}}}:
           - Single reference: [[The Seller shall deliver...]]{{{{Agreement.pdf}}}}
           - Multiple references: [[The Buyer agrees to pay...]]{{{{Agreement.pdf}}}} [1] [[All disputes shall be...]]{{{{Agreement.pdf}}}} [2] [[This agreement shall be...]]{{{{Agreement.pdf}}}} [3]
        3. Never combine multiple references within a single bracket.
        4. Always use the exact text as it appears in the document to ensure searchability - do not paraphrase or summarize.
        5. Do not reference clause numbers (like "Clause 6.3") - instead use the actual text content from that clause.
        6. Always include the filename after each citation in DOUBLE curly braces.
        7. Place citations at the end of each analysis point, not at the beginning.

        Risk Exposure additional formatting rules:
        Title/Risk Point: Clear description of the specific risk and its implications. Supporting citation: [[exact text from document]]{{{{filename}}}}


        Structure your response in the following specific format:

        *****[PARTY NAME]*****
        PERSPECTIVE: Brief overview of this party's position and key objectives in the agreement

        **RISK EXPOSURE**
        
        {risk_content}

        [Repeat for each party]

        IMPORTANT FORMATTING RULES:
        - Use ***** only for party names
        - For section headers, use **
        - Include exact quotes using the format specified above with filenames
        - Keep citations concise (30-40 characters)
        - Never include formatting characters in citations
        - Do not use ellipsis (...), just use the first part of the text
        - Maintain professional, clear language
        - Always place citations at the end of the analysis point
        """
    
    elif analysis_type == 'ask':
        prompt = ASK_PROMPT
    
    elif analysis_type == 'draft':
        prompt = DRAFT_PROMPT
    else:
        logger.error(f"Invalid analysis type: {analysis_type}")
        raise ValueError(f"Invalid analysis type: {analysis_type}")

    logger.info(f"Using prompt: {prompt}")

    try:
        # For explanations, we'll use Claude for more nuanced responses
        if analysis_type == 'explain':
            result = claude_call(text, prompt)
        else:
            result = gemini_call(text, prompt)
        
        logger.info("API call successful")
        print(f"result: {result}")
        return result
    except Exception as e:
        logger.exception(f"Error calling API for {analysis_type}")
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
    
def gemini_call_flash(text, prompt):
    logger.info("Calling Gemini Flash API")
    
    system_prompt = """You are a highly experienced analyst who is great at analyzing documents and providing insights."""
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        [system_prompt, text, prompt],
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=4000,
            top_p=0.8,
            top_k=40
        )
    )
    return response.text

def gemini_call(text, prompt):
    logger.info("Calling Gemini API")
    
    system_prompt = """You are a highly experienced legal assistant to the General Counsel of a Fortune 500 company."""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        if(text == ""):
            response = model.generate_content(
                [system_prompt, prompt],
                generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Slightly increased for more natural language while maintaining precision
                max_output_tokens=4000,  # Increased token limit for more comprehensive responses
                top_p=0.8,  # Added for better response quality
                top_k=40,  # Added for better response diversity while maintaining relevance
                )
            )
        else:
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
    logger.info("Calling Claude SONNET API")
    
    system_prompt = """You are a highly experienced General Counsel of a Fortune 500 company with over 20 years of experience in corporate law."""
    
    try:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=4000,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\nDocument:\n{text}"
                }
            ]
        )
        logger.info("Claude API call successful")
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        raise Exception(f"An error occurred while calling Claude API: {e}")
    
def claude_call_haiku(text, prompt):
    logger.info("Calling Claude HAIKU API")
    
    system_prompt = """You are a highly experienced analyst who is great at analyzing documents and providing insights."""
    
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=4000,
        temperature=0.1,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nDocument:\n{text}"
        }]
    )
    print(response.usage.input_tokens)
    print(response.usage.output_tokens)
    return response.content[0].text

def claude_call_cache(text, prompt):
    logger.info("Calling Claude SONNET API with prompt caching")
    
    system_prompt = """You are a highly experienced analyst who is great at analyzing documents and providing insights."""
    
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    response = client.beta.prompt_caching.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=4000,
        temperature=0.1,
        system=[
            {
                "type": "text",
                "text": system_prompt
            },
            {
                "type": "text", 
                "text": text,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    print(response.usage.cache_creation_input_tokens)
    print(response.usage.cache_read_input_tokens)
    print(response.usage.input_tokens)
    print(response.usage.output_tokens)

    return response.content[0].text

def claude_call_explanation(prompt):
    logger.info("Calling Claude SONNET API")

    system_prompt = """You are a highly experienced General Counsel of a Fortune 500 company with over 20 years of experience in corporate law."""
    
    try:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1000,
            temperature=0.9,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        logger.info("Claude API call successful")
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        raise Exception(f"An error occurred while calling Claude API: {e}")
    
def claude_call_opus(text, prompt):
    logger.info("Calling Claude OPUS API")
    
    try:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        response = client.messages.create(
            model="claude-3-opus-latest",
            max_tokens=4000,
            temperature=0.1,
            system="You are a highly experienced General Counsel of a Fortune 500 company with over 20 years of experience in corporate law.",
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\nDocument:\n{text}"
                }
            ]
        )
        logger.info("Claude API call successful")
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        raise Exception(f"An error occurred while calling Claude API: {e}")
    
def analyze_conflicts_and_common_parties(texts: Dict[str, str]) -> str:
    logger.info("Analyzing conflicts and common parties")
    
    prompt = CONFLICT_ANALYSIS_PROMPT

    for filename, content in texts.items():
        prompt += f"\n\nFilename: {filename}\nContent:\n{content}"

    try:
        result = gemini_call("", prompt)
        logger.info("Gemini API call for conflict and common party analysis successful")
        return result
    except Exception as e:
        logger.exception("Error analyzing conflicts and common parties")
        raise Exception(f"An error occurred while analyzing conflicts and common parties: {e}")

def analyze_document_clauses(text: str, party_info: dict = None) -> dict:
    """
    Analyzes document clauses from a specific party's perspective
    """
    if party_info:
        party_name = party_info.get('name', '')
        party_role = party_info.get('role', '')
        print(f"party name: {party_name}")
        prompt = f"""
        Analyze the following legal document from the perspective of {party_name} 
        (acting as {party_role}) and categorize its clauses into three categories:
        
        1. Acceptable Clauses: Terms that are favorable or standard for {party_name}
        2. Risky Clauses: Terms that pose potential risks or need negotiation for {party_name}
        3. Missing Clauses: Important clauses that should be present to protect {party_name}'s interests
        
        Consider the specific role and interests of {party_name} as {party_role} 
        when analyzing each clause.
        
        For each clause identified, provide:
        - Category
        - Clause title/type
        - Relevant text excerpt
        - Explanation of categorization from {party_name}'s perspective
        
        Format the response as a JSON structure and only return the JSON:
        {{
            "acceptable": [
                {{
                    "title": "clause title",
                    "text": "complete clause text exactly as it appears in the document",
                    "explanation": "why acceptable for {party_name}",
                }}
            ],
            "risky": [...],
            "missing": [...]
        }}
        Only return the JSON, no other text.
        """
    else:
        # Your existing prompt for general analysis
        prompt = """
        Analyze the following legal document and categorize its clauses into three categories:
        1. Acceptable Clauses: Standard terms that follow industry best practices
        2. Risky Clauses: Terms that need attention or negotiation
        3. Missing Clauses: Important clauses that should be present but are not
        
        For each clause identified, provide:
        - Category
        - Clause title/type
        - Relevant text excerpt
        - Proper explanation of categorization
        
        Format the response as a JSON structure:
        {{
            "acceptable": [
                {{
                    "title": "clause title",
                    "text": "complete clause text exactly as it appears in the document",
                    "explanation": "why acceptable",
                }}
            ],
            "risky": [...],
            "missing": [...]
        }}
        """
    
    try:
        result = gemini_call(text, prompt)
        print(f"result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in clause analysis: {str(e)}")
        raise

def analyze_document_parties(text: str) -> list:
    """
    Analyzes document content to extract parties involved.
    Returns a list of party names and their roles.
    """
    prompt = """
    Analyze the following legal document and identify all parties involved.
    For each party, provide:
    1. Party name
    2. Role in the document (e.g., Buyer, Seller, Lender, Borrower, etc.)
    
    Return the results in this JSON format:
    {
        "parties": [
            {
                "name": "party name",
                "role": "party role"
            }
        ]
    }

    just the parties and roles, no other text
    """
    
    try:
        result = claude_call_haiku(text, prompt)
        print(f"result of party analysis: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in party analysis: {str(e)}")
        raise

def check_common_parties(parties_by_file):
    """
    Check if there are common parties between documents using Gemini Flash.
    Returns both whether there are common parties and the list of common parties.
    """
    prompt = """
    Analyze the following parties from different documents and identify any common parties.
    
    Document Parties:
    """
    
    # Format the parties data for the prompt
    for filename, parties in parties_by_file.items():
        prompt += f"\n\n{filename}:\n"
        for party in parties:
            prompt += f"- {party['name']} ({party['role']})\n"
    
    prompt += """
    
    Provide your response in the following JSON format:
    {
        "has_common_parties": "Yes/No",
        "common_parties": [
            {
                "name": "party name",
                "roles": [
                    {"role": "role1", "filename": "doc1.pdf"},
                    {"role": "role2", "filename": "doc2.pdf"}
                ]
            }
        ]
    }
    
    Only return the JSON, no other text.
    """
    
    try:
        result = claude_call_haiku("", prompt)
        parsed_result = json.loads(result)
        return {
            'has_common_parties': parsed_result['has_common_parties'].lower() == 'yes',
            'common_parties': parsed_result['common_parties']
        }
    except Exception as e:
        logger.error(f"Error checking common parties: {e}")
        return {
            'has_common_parties': False,
            'common_parties': []
        }

def analyze_conflicts(text, common_parties):
    """
    Analyzes potential conflicts of interest for each common party.
    """
    analyses = {}
    
    for party in common_parties:
        prompt = f"""
        As a legal expert, analyze the potential conflicts of interest and compliance risks for the following party 
        based on their different roles across multiple documents:
        
        Party: {party['name']}
        Roles across documents:
        """
        
        # Add this party's roles to the prompt
        for role_info in party['roles']:
            prompt += f"- {role_info['role']} in {role_info['filename']}\n"
            
        prompt += """
        CITATION REQUIREMENTS:
        1. When referencing document text:
           - Include exact text in [[double brackets]]
           - Add filename after citation in {{curly braces}}
           - Example: [[The party shall be liable]]{{contract1.pdf}}
        
        2. Citation Guidelines:
           - Use exact text as it appears, no paraphrasing
           - Keep citations concise (30-40 characters)
           - Never combine multiple references in one bracket
           - Include filename immediately after each citation
           - Do not use ellipsis, use first part of relevant text
           - No formatting characters in citations
        
        Additional formatting rules for every party:
        Title/Risk Point: Clear description of the specific conflict/breach and its implications. Supporting citation: [[exact text from document]]{{{{filename}}}}
        Never use citation as your analysis, use it as a reference to the specific clause in the document at the end to support your analysis and not as a replacement to analysis content.
Provide your analysis in this structure:
        For the party, ANALYZE ACROSS ALL PROVIDED DOCUMENTS in which party is involved:

        1. OBLIGATION MAPPING:
        - Extract all binding commitments of [TARGET PARTY]:
            * Non-compete restrictions (scope, territory, duration)
            * Confidentiality obligations
            * Exclusivity commitments 
            * Performance requirements
            * Use/disclosure limitations
            * Service/delivery obligations

        2. CONFLICT IDENTIFICATION:
        - Cross-reference obligations to identify:
            * Direct conflicts between commitments
            * Indirect/implied conflicts through performance
            * Timing overlaps creating impossibility
            * Geographic/territorial conflicts
            * Industry/sector conflicts
        - For each conflict:
            * Cite specific clauses in conflict
            * Explain precise nature of incompatibility
            * Identify triggering scenarios/actions
            * Note which agreement was executed first
            * Provide whole analysis, do not replace citations for analysis

        3. BREACH ANALYSIS:
        - For each identified conflict:
            * What specific actions would trigger breach
            * Which agreement would be breached first
            * Domino effect on other obligations
            * Available cure periods
            * Cross-default implications
            * Materiality assessment
            * Provide whole analysis, do not replace citations for analysis
        - Document:
            * Relevant notice requirements
            * Grace periods
            * Cure rights
            * Force majeure applicability
        
        IMPORTANT:
        
        - Give paragraphs in analysis rather that points, provide a rich quality thorough analysis
        - Always place citations at the end of the analysis point
        
        Documents to analyze:
        """
        
        try:
            party_analysis = claude_call_cache(text, prompt)
            analyses[party['name']] = party_analysis
            # Analyze each party individually
            print('*' * 100)
            party_analysis = claude_call_cache(text, prompt)
            print('*' * 100)
            
            # Add to analyses dictionary with party name as key
        except Exception as e:
            logger.error(f"Error analyzing conflicts for party {party['name']}: {str(e)}")
            analyses[party['name']] = f"Error in analysis: {str(e)}"
    
    return analyses

def convert_pdf_to_docx(pdf_file_path):
    """
    Converts PDF to DOCX and returns the content as base64
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_docx:
            # Convert PDF to DOCX
            cv = Converter(pdf_file_path)
            cv.convert(temp_docx.name)
            cv.close()
            
            # Read the converted file
            with open(temp_docx.name, 'rb') as docx_file:
                docx_content = docx_file.read()
                
            # Convert to base64
            base64_content = base64.b64encode(docx_content).decode('utf-8')
            
            return {
                'success': True,
                'content': base64_content,
                'mime_type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        # Cleanup
        if os.path.exists(temp_docx.name):
            os.remove(temp_docx.name)



