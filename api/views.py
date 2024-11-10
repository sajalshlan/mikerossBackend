import os
import logging
import base64
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from .utils import RAGPipeline, extract_text_from_file, extract_text_from_zip, perform_analysis as util_perform_analysis, analyze_conflicts_and_common_parties

logger = logging.getLogger(__name__)

# Initialize RAGPipeline
rag_pipeline = RAGPipeline()

@csrf_exempt
@api_view(['GET'])
def health(request):
    return JsonResponse({'status': 'ok'})

def analyze_text(analysis_type, text):
    logger.info(f"Performing {analysis_type} analysis")
    logger.info(f"Text length: {len(text) if text else 0}")

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

@csrf_exempt
@api_view(['POST'])
def upload_file(request):
    logger.info("File upload request received")
    if 'file' not in request.FILES:
        logger.warning("No file uploaded in the request")
        return Response({'error': 'No file uploaded'}, status=400)

    file = request.FILES['file']
    file_extension = request.POST.get('file_extension', '')
    logger.info(f"Received file: {file.name} with extension {file_extension}")
    file_path = os.path.join(settings.MEDIA_ROOT, file.name)

    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    logger.info(f"File saved to {file_path}")

    try:
        if file.name.lower().endswith('.zip'):
            logger.info("Processing ZIP file")
            extracted_contents = extract_text_from_zip(file_path, rag_pipeline)
            return Response({'success': True, 'files': extracted_contents})
        else:
            logger.info("Processing single file")
            extracted_text = extract_text_from_file(file_path, rag_pipeline, file_extension)
            response_data = {
                'success': True, 
                'text': extracted_text
            }
            if file.name.lower().endswith('.pdf'):
                with open(file_path, 'rb') as pdf_file:
                    pdf_content = pdf_file.read()
                    response_data['base64'] = base64.b64encode(pdf_content).decode('utf-8')
                    response_data['ocr_text'] = extracted_text
            return Response(response_data)
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return Response({'error': str(e)}, status=500)
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")

@api_view(['POST'])
def perform_analysis(request):
    logger.info("Analysis request received")
    analysis_type = request.data.get('analysis_type')
    text = request.data.get('text')

    result = analyze_text(analysis_type, text)
    if 'error' in result:
        return Response(result, status=400 if 'Invalid analysis type' in result['error'] else 500)
    return Response(result)

# @api_view(['POST'])
# def perform_conflict_check(request):
#     logger.info("Conflict check request received")
#     texts = request.data.get('texts')

#     if not texts or not isinstance(texts, dict) or len(texts) < 2:
#         logger.warning("Invalid input for conflict check")
#         return Response({'error': 'At least two documents are required for conflict check'}, status=400)

#     # Check if there's at least one common party
#     if not has_common_party(texts):
#         logger.info("No common parties found in the documents")
#         return Response({
#             'success': True,
#             'result': 'No conflicts to check. No common parties found in the documents.'
#         })

#     try:
#         combined_text = "\n\n".join([f"Document: {filename}\n\n{content}" for filename, content in texts.items()])
#         result = analyze_text('conflict', combined_text)
#         if 'error' in result:
#             return Response(result, status=500)
#         return Response(result)
#     except Exception as e:
#         logger.exception("Error performing conflict check")
#         return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def perform_conflict_check(request):
    logger.info("Conflict check request received")
    texts = request.data.get('texts')

    if not texts or not isinstance(texts, dict) or len(texts) < 2:
        logger.warning("Invalid input for conflict check")
        return Response({'error': 'At least two documents are required for conflict check'}, status=400)

    try:
        result = analyze_conflicts_and_common_parties(texts)
        return Response({
            'success': True,
            'result': result
        })
    except Exception as e:
        logger.exception("Error performing conflict check")
        return Response({'error': str(e)}, status=500)

