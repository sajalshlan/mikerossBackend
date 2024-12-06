import os
import logging
import base64
import gc
import time
import random
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.http import JsonResponse
from .utils import (
    RAGPipeline, 
    extract_text_from_file, 
    extract_text_from_zip, 
    perform_analysis as util_perform_analysis, 
    analyze_conflicts_and_common_parties,
    analyze_document_clauses,
    analyze_document_parties,
    ResourceMonitor
)
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import UserSerializer, RegisterSerializer, AcceptTermsSerializer
from .models import Document, User

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
@permission_classes([AllowAny])
def register(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        return Response({
            "user": UserSerializer(user).data,
            "message": "User created successfully"
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_file(request):
    """
    Handles file uploads with memory-efficient processing.
    """
    logger.info(f"Upload request from user: {request.user.username} (is_root: {request.user.is_root})")
    logger.info(f"User organization: {request.user.organization}")
    
    if not request.user.organization and not request.user.is_root:
        logger.warning(f"No organization associated with user {request.user.username}")
        return Response({
            'error': 'No organization associated',
            'details': {
                'is_root': request.user.is_root,
                'has_organization': bool(request.user.organization)
            }
        }, status=400)
    
    file = request.FILES.get('file')
    if not file:
        return Response({'error': 'No file provided'}, status=400)
    
    # Modified organization handling
    if request.user.is_root:
        organization = request.data.get('organization')  # Optional for root
    else:
        organization = request.user.organization  # Required for non-root
        if not organization:
            return Response({'error': 'No organization associated'}, status=400)
    
    document = Document.objects.create(
        title=file.name,
        file=file,
        organization=organization,  # Can be None for root user
        uploaded_by=request.user
    )
    
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

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def perform_analysis(request):
    """
    Performs text analysis with memory management.
    """
    resource_monitor.log_memory("Starting analysis request")
    
    text = None
    try:
        analysis_type = request.data.get('analysis_type')
        text = request.data.get('text')
        ocr_text = request.data.get('ocr_text')
        # Add validation with specific error messages
        if not analysis_type:
            logger.warning("Missing analysis_type")
            return Response({'error': 'Please specify the type of analysis to perform'}, status=400)
            
        if not text and not ocr_text:
            logger.warning("No text or OCR text found to analyze")
            return Response({'error': 'No text found to analyze. Please provide text content or upload a document.'}, status=400)
            
        if analysis_type == 'ocr' and not ocr_text:
            logger.warning("No OCR text found for OCR analysis")
            return Response({'error': 'No OCR text found. Please upload a document for OCR analysis.'}, status=400)
            
        if analysis_type != 'ocr' and not text:
            logger.warning("No normal text found for non-OCR analysis")
            return Response({'error': 'No text content found. Please provide text for analysis.'}, status=400)
            
        include_history = request.data.get('include_history', False)
        
        # Parse the input for chat analysis
        if analysis_type == 'ask' and include_history:
            text = f'{text}\n\nPrevious Conversation (last 10 messages):\n{include_history}'
        result = analyze_text(analysis_type, text or ocr_text)  # Use OCR text if normal text is not available
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

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
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

def get_organization_filter(user):
    """Helper to get organization filter based on user type"""
    if user.is_root:
        return {}  # No filter for root users
    return {'organization': user.organization}

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_profile(request):
    serializer = UserSerializer(request.user)
    return Response(serializer.data)

class CustomTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        logger.info(f"Login attempt with username: {request.data.get('username')}")
        response = super().post(request, *args, **kwargs)
        logger.info(f"Login response status: {response.status_code}")
        return response

@api_view(['GET', 'PATCH'])
@permission_classes([IsAuthenticated])
def accept_terms(request):
    user = request.user
    
    if request.method == 'GET':
        return Response({
            'accepted_terms': user.accepted_terms if hasattr(user, 'accepted_terms') else False
        })
    
    elif request.method == 'PATCH':
        serializer = AcceptTermsSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def explain_text(request):
    """
    Explains a selected portion of text in detail.
    """
    resource_monitor.log_memory("Starting explanation request")
    
    selected_text = None
    context_text = None
    prompt = None
    
    try:
        selected_text = request.data.get('selectedText')
        context_text = request.data.get('contextText')
        
        if not selected_text:
            return Response({'error': 'No text selected for explanation'}, status=400)
            
        prompt = f"""
        You are provided with a document and a section of text from that document.
        Your task is to explain the selected text in more detail being a legal expert. Do not mention about your role.
        
        Document Context:
        {context_text} 
        
        Selected Text to Explain:
        {selected_text}
        
        Provide a to the point and very concise explanation of the selected text keeping in mind the context of the document.
        """
        
        result = util_perform_analysis('explain', prompt)
        return Response(result)
        
    except Exception as e:
        logger.exception("Error generating explanation")
        return Response({'error': str(e)}, status=500)
    finally:
        del selected_text
        del context_text
        del prompt
        resource_monitor.force_cleanup()

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def reply_to_comment(request):
    """
    Generates an AI reply to a comment based on the original comment, its context, and optional instructions.
    """
    resource_monitor.log_memory("Starting reply generation request")
    
    comment = None
    document_content = None
    instructions = None
    replies = None
    replies_context = None
    prompt = None
    
    try:
        comment = request.data.get('comment')
        document_content = request.data.get('documentContent')
        instructions = request.data.get('instructions', '')
        replies = request.data.get('replies', [])
        
        if not comment or not document_content:
            return Response({
                'error': 'Missing required data (comment or document content)'
            }, status=400)
            
        # Format replies for context
        replies_context = ""
        if replies:
            replies_context = "\n\nComment Thread:\n"
            for idx, reply in enumerate(replies, 1):
                replies_context += f"Reply {idx}: {reply['content']}\n"
            
        prompt = f"""
        You are tasked with generating a reply to a comment in a document. Consider the following:
        
        Document Content:
        {document_content}
        
        Original Comment:
        {comment}
        {replies_context}
        
        Instructions for Reply:
        {instructions if instructions else "Maintain the same tone and length as the original comment while considering the context of any replies."}
        
        Please provide a reply that:
        1. Maintains professional tone
        2. Addresses the same core issues
        3. Takes into account the context from any replies
        4. Follows any provided instructions
        5. Is clear and concise
        
        Provide only the reply without any explanations or additional text.
        """
        
        result = util_perform_analysis('explain', prompt)
        return Response({'success': True, 'result': result})
        
    except Exception as e:
        logger.exception("Error generating reply")
        return Response({'error': str(e)}, status=500)
    finally:
        del comment
        del document_content
        del instructions
        del replies
        del replies_context
        del prompt
        resource_monitor.force_cleanup()

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def redraft_comment(request):
    """
    Generates a redraft of the selected text based on the comment context.
    """
    resource_monitor.log_memory("Starting redraft generation request")
    
    comment = None
    document_content = None
    selected_text = None
    instructions = None
    replies = None
    replies_context = None
    prompt = None
    
    try:
        comment = request.data.get('comment')
        document_content = request.data.get('documentContent')
        selected_text = request.data.get('selectedText')
        instructions = request.data.get('instructions', '')
        replies = request.data.get('replies', [])
        
        if not all([comment, document_content, selected_text]):
            return Response({
                'error': 'Missing required data (comment, document content, or selected text)'
            }, status=400)
            
        # Format replies for context
        replies_context = ""
        if replies:
            replies_context = "\n\nComment Thread:\n"
            for idx, reply in enumerate(replies, 1):
                replies_context += f"Reply {idx}: {reply['content']}\n"
            
        prompt = f"""
        You are tasked with redrafting a portion of text from a document based on comments and feedback. Consider the following:
        
        Document Context:
        {document_content}
        
        Original Text to Redraft:
        {selected_text}
        
        Comment on this text:
        {comment}
        {replies_context}
        
        Instructions for Redraft:
        {instructions if instructions else "Improve the text while maintaining the document's style and addressing the feedback in the comments."}
        
        Please provide a redraft that:
        1. Maintains the document's tone and style
        2. Addresses the issues raised in the comments
        3. Improves clarity and precision
        4. Follows any provided instructions
        5. Fits seamlessly into the document context
        
        Provide only the redrafted text without any explanations or additional text.
        """
        
        result = util_perform_analysis('explain', prompt)
        return Response({'success': True, 'result': result})
        
    except Exception as e:
        logger.exception("Error generating redraft")
        return Response({'error': str(e)}, status=500)
    finally:
        del comment
        del document_content
        del selected_text
        del instructions
        del replies
        del replies_context
        del prompt
        resource_monitor.force_cleanup()

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_clauses(request):
    text = None
    result = None
    
    try:
        text = request.data.get('text', '')
        party_info = request.data.get('partyInfo', {})  # Provide empty dict as default
        
        if not text:
            return Response({'error': 'No text provided'}, status=400)
        
        if not party_info:
            logger.warning("No party info provided for clause analysis")
            
        result = analyze_document_clauses(text, party_info)
        return Response({
            'success': True,
            'result': result
        })
    except Exception as e:
        logger.error(f"Error in analyze_clauses: {str(e)}")
        return Response({
            'error': str(e)
        }, status=500)
    finally:
        del text
        del result
        gc.collect()

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_parties(request):
    """
    Analyzes document content to extract parties involved.
    """
    text = None
    try:
        text = request.data.get('text')
        # print(text)
        if not text:
            return Response({'error': 'No text provided'}, status=400)
            
        result = analyze_document_parties(text)
        return Response({
            'success': True,
            'parties': result
        })
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=500)
    finally:
        del text
        gc.collect()