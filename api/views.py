import os
import logging
import base64
import gc
import time
import random
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from .utils import (
    RAGPipeline, 
    extract_text_from_file, 
    extract_text_from_zip, 
    perform_analysis as util_perform_analysis, 
    analyze_conflicts_and_common_parties,
    ResourceMonitor
)

logger = logging.getLogger(__name__)

# Initialize RAGPipeline and ResourceMonitor
rag_pipeline = RAGPipeline()
resource_monitor = ResourceMonitor()

@csrf_exempt
@api_view(['GET'])
def health(request):
    return JsonResponse({'status': 'ok'})

def analyze_text(analysis_type: str, text: str) -> dict:
    """
    Analyzes text with memory monitoring and cleanup.
    """
    resource_monitor.log_memory("Starting text analysis")
    
    if not analysis_type or not text:
        logger.warning("Missing analysis_type or text")
        return {'error': 'Missing analysis_type or text'}
    
    try:
        result = util_perform_analysis(analysis_type, text)
        logger.info("Analysis completed successfully")
        return {'success': True, 'result': result}
    except ValueError as ve:
        logger.error(f"Invalid analysis type: {str(ve)}")
        return {'error': str(ve)}
    except Exception as e:
        logger.exception(f"Error performing {analysis_type} analysis")
        return {'error': str(e)}
    finally:
        resource_monitor.force_cleanup()

def process_file_chunk(chunk: bytes, destination) -> None:
    """
    Processes file chunks with proper cleanup.
    """
    try:
        destination.write(chunk)
    finally:
        del chunk
        gc.collect()

def process_single_file(file_path: str, file_extension: str) -> dict:
    """
    Processes a single file with memory management.
    """
    resource_monitor.log_memory("Processing single file")
    response_data = {'success': True}
    extracted_text = None
    
    try:
        extracted_text = extract_text_from_file(file_path, rag_pipeline, file_extension)
        response_data['text'] = extracted_text
        
        if file_path.lower().endswith('.pdf'):
            response_data.update(get_pdf_data(file_path, extracted_text))
        
        return response_data
    finally:
        del extracted_text
        resource_monitor.force_cleanup()

def get_pdf_data(file_path: str, extracted_text: str) -> dict:
    """
    Gets PDF data in chunks with memory management.
    """
    pdf_data = {}
    chunk_size = 8192  # 8KB chunks
    
    try:
        base64_chunks = []
        with open(file_path, 'rb') as pdf_file:
            while True:
                chunk = pdf_file.read(chunk_size)
                if not chunk:
                    break
                base64_chunks.append(base64.b64encode(chunk).decode('utf-8'))
                del chunk
        
        pdf_data['base64'] = ''.join(base64_chunks)
        pdf_data['ocr_text'] = extracted_text
        return pdf_data
    finally:
        del base64_chunks
        gc.collect()

@csrf_exempt
@api_view(['POST'])
def upload_file(request):
    """
    Handles file uploads with memory-efficient processing.
    """
    resource_monitor.log_memory("Starting file upload")
    
    if 'file' not in request.FILES:
        logger.warning("No file uploaded in the request")
        return Response({'error': 'No file uploaded'}, status=400)
    
    file = request.FILES['file']
    file_extension = request.POST.get('file_extension', '')
    logger.info(f"Received file: {file.name} with extension {file_extension}")
    
    # Create media directory if it doesn't exist
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    file_path = os.path.join(settings.MEDIA_ROOT, str(int(time.time()))+ str(random.randint(1, 100)) + file.name)

    extracted_contents = None
    result = None
    try:
        # Write file in chunks
        with open(file_path, 'wb') as destination:
            for chunk in file.chunks(chunk_size=8192):
                process_file_chunk(chunk, destination)
        
        # Process file based on type
        if file.name.lower().endswith('.zip'):
            logger.info("Processing ZIP file")
            extracted_contents = extract_text_from_zip(file_path, rag_pipeline)
            return Response({'success': True, 'files': extracted_contents})
        else:
            logger.info("Processing single file")
            result = process_single_file(file_path, file_extension)
            return Response(result)
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return Response({'error': str(e)}, status=500)
    finally:
        if extracted_contents is not None:
            del extracted_contents
        if result is not None:
            del result
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")
        resource_monitor.force_cleanup()

@api_view(['POST'])
def perform_analysis(request):
    """
    Performs text analysis with memory management.
    """
    resource_monitor.log_memory("Starting analysis request")
    
    text = None
    try:
        analysis_type = request.data.get('analysis_type')
        text = request.data.get('text')
        include_history = request.data.get('include_history', False)
        
        # Parse the input for chat analysis
        if analysis_type == 'ask' and include_history:
            text = f'{text}\n\nPrevious Conversation (last 10 messages):\n{include_history}'
            print(f'[API] 📄 Chat history: {text}')
        
        result = analyze_text(analysis_type, text)
        if 'error' in result:
            return Response(
                result, 
                status=400 if 'Invalid analysis type' in result['error'] else 500
            )
        return Response(result)
    finally:
        # Clean up the text variable
        del text
        resource_monitor.force_cleanup()

@api_view(['POST'])
def perform_conflict_check(request):
    """
    Performs conflict check with memory management.
    """
    resource_monitor.log_memory("Starting conflict check")
    texts = request.data.get('texts')
    
    if not texts or not isinstance(texts, dict) or len(texts) < 2:
        logger.warning("Invalid input for conflict check")
        return Response(
            {'error': 'At least two documents are required for conflict check'}, 
            status=400
        )
    
    try:
        result = analyze_conflicts_and_common_parties(texts)
        return Response({
            'success': True,
            'result': result
        })
    except Exception as e:
        logger.exception("Error performing conflict check")
        return Response({'error': str(e)}, status=500)
    finally:
        # Clean up the texts dictionary
        for key in list(texts.keys()):
            del texts[key]
        del texts
        resource_monitor.force_cleanup()