import os
import logging
import base64
import gc
import time
import random
import json
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
    ResourceMonitor,
    claude_call_explanation,
    claude_call_opus,
    check_common_parties,
    analyze_conflicts,
    convert_pdf_to_docx,
    gemini_call
)
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import UserSerializer, RegisterSerializer, AcceptTermsSerializer
import tempfile

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
    
    file_extension = request.POST.get('file_extension', '')
    logger.info(f"Received file: {file.name} with extension {file_extension}")
    
    # Create temporary file path without saving to media directory
    file_path = os.path.join('/tmp', str(int(time.time())) + str(random.randint(1, 100)) + file.name)

    extracted_contents = None
    result = None
    try:
        # Write file in chunks to temporary location
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
        filename = request.data.get('filename')
        ocr_text = request.data.get('ocr_text')

        text = f'Document Filename: {filename}\n\n{text}'
        
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
        referenced_text = request.data.get('referenced_text', False)
        # Parse the input for chat analysis
        if analysis_type == 'ask':
            context_parts = []
            
            if referenced_text:
                print(f"referenced_text: {referenced_text}")
                # If there's referenced text, use it as primary context
                context_parts.extend([
                    f'Selected Text for Reference:\n{referenced_text}',
                    f'Document Context:\n{text}',
                    'Please answer primarily focusing on the referenced text while considering the document context.'
                ])
            else:
                # If no referenced text, use document context and include history if available
                context_parts.append(f'Document Context:\n{text}')
                if include_history:
                    context_parts.append(f'Previous Conversation (last 10 messages):\n{include_history}')
                    context_parts.append('Please provide a response considering both the document context and the conversation history.')
            
            text = '\n\n'.join(context_parts)
        result = analyze_text(analysis_type, text or ocr_text)
        if 'error' in result:
            return Response(
                result, 
                status=400 if 'Invalid analysis type' in result['error'] else 500
            )
        return Response(result)
    finally:
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
        # First get parties for each document
        parties_by_file = {}
        for filename, text in texts.items():
            parties_json = analyze_document_parties(text)
            # Parse the JSON string into Python dict
            parties = json.loads(parties_json)['parties']
            parties_by_file[filename] = parties
        
        print(f"parties_by_file: {parties_by_file}")
        # Check for common parties using gemini flash
        has_common = check_common_parties(parties_by_file)
        logger.info(f"Common parties check result: {has_common}")

        formatted_texts = ""
        for filename, content in texts.items():
            formatted_texts += f"\nDocument: {filename}\n\n{content}\n"
            formatted_texts += "-" * 50 + "\n\n" 

        # print(f"has_common: {has_common}")

        # answer = has_common['common_parties']
        # print(f"answer: {answer}")
        
        # Then perform the regular conflict analysis
        # result = analyze_conflicts_and_common_parties(texts)
        common_parties = has_common['common_parties']
        logger.info(f"Common parties identified: {common_parties}")
        
        # Analyze conflicts for common parties
        if common_parties:
            conflict_analyses = analyze_conflicts(formatted_texts,common_parties)
            logger.info(f"Conflict analyses completed: {conflict_analyses}")
            
            result = {
                'has_common_parties': True,
                'common_parties': common_parties,
                'analyses': conflict_analyses
            }
        else:
            result = {
                'has_common_parties': False,
                'common_parties': [],
                'analyses': {}
            }
        
        logger.info(f"Final result structure: {result}")
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

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def redraft_text(request):
    """
    Endpoint to redraft selected text with optional instructions.
    """
    try:
        selected_text = request.data.get('selectedText')
        document_content = request.data.get('documentContent')
        instructions = request.data.get('instructions', '')

        if not selected_text or not document_content:
            return Response({
                'success': False,
                'error': 'Missing required parameters'
            }, status=400)

        # Create prompt for redrafting
        prompt = f"""
        You are a legal document expert. Your task is to redraft the following text to improve its clarity, 
        precision, and legal effectiveness while maintaining its original intent.

        Document Context:
        {document_content}

        Text to Redraft:
        {selected_text}

        {f"Additional Instructions: {instructions}" if instructions else ""}

        Please provide only the redrafted text without any explanations or additional text.
        Ensure the redrafted version:
        1. Maintains legal accuracy and enforceability
        2. Improves clarity and readability
        3. Uses consistent terminology
        4. Follows standard legal drafting conventions
        """

        # Use the existing perform_analysis utility with a specific mode
        result = util_perform_analysis('explain', prompt)

        return Response({
            'success': True,
            'result': result
        })

    except Exception as e:
        logger.exception("Error in redraft_text endpoint")
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def brainstorm_chat(request):
    """
    Endpoint for brainstorming solutions and ideas about specific clauses.
    """
    resource_monitor.log_memory("Starting brainstorm chat")
    
    message = None
    clause_text = None
    analysis = None
    document_content = None
    prompt = None
    
    try:
        message = request.data.get('message')
        clause_text = request.data.get('clauseText')
        analysis = request.data.get('analysis')
        document_content = request.data.get('documentContent')
            
        prompt = f"""
        You are a legal expert helping to brainstorm and discuss solutions for contract clauses. 
        Consider the following context:

        Document Context:
        {document_content}

        Clause being discussed:
        {clause_text}

        Analysis of the clause:
        {analysis}

        User's message:
        {message}

        Please provide a helpful, detailed response that:
        1. Directly addresses the user's message/question
        2. Considers the specific context of the clause
        3. References relevant legal principles or best practices
        4. Suggests practical solutions or alternatives when appropriate
        5. Maintains a conversational yet professional tone

        Focus on being constructive and solution-oriented while maintaining legal accuracy.
        """
        print(prompt)
        result = claude_call_explanation(prompt)
        return Response({
            'success': True,
            'message': result
        })
        
    except Exception as e:
        logger.exception("Error in brainstorm chat")
        return Response({
            'error': str(e)
        }, status=500)
    finally:
        del message
        del clause_text
        del analysis
        del document_content
        del prompt
        resource_monitor.force_cleanup()

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def preview_pdf_as_docx(request):
    """
    Converts PDF to DOCX for preview purposes
    """
    file = request.FILES.get('file')
    if not file:
        return Response({'error': 'No file provided'}, status=400)
        
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            for chunk in file.chunks():
                temp_pdf.write(chunk)
                
        # Convert to DOCX
        result = convert_pdf_to_docx(temp_pdf.name)
        
        if result['success']:
            return Response(result)
        else:
            return Response({'error': result['error']}, status=500)
    finally:
        if os.path.exists(temp_pdf.name):
            os.remove(temp_pdf.name)

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat(request):
    """
    Handles chat interactions with context awareness and focused responses on referenced text
    """
    try:
        # Extract data from request
        message = request.data.get('message')
        document_context = request.data.get('documentContext', '')
        chat_history = request.data.get('chatHistory', [])
        referenced_text = request.data.get('referencedText', '')
        filename = request.data.get('filename', 'document')  # Default filename if not provided
        
        if not message:
            return Response({
                'error': 'No message provided'
            }, status=400)

        # Format chat history (last 10 messages)
        formatted_history = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in chat_history[-10:]
        ])

        # Create the base prompt structure
        if referenced_text:
            # Focus primarily on the referenced text
            prompt = f"""
            You are a legal AI assistant with expertise in contract analysis and legal document review. Your role is to provide clear, authoritative answers while maintaining accuracy through precise citations.

            
            Selected Text for Primary Focus, resolve the query in this text:
            {referenced_text}
            
            Broader Document Context (for reference only):
            {document_context}
            
            User's Message:
            {message}
            """
        else:
            # Regular full document analysis
            prompt = f"""
             You are a legal AI assistant with expertise in contract analysis and legal document review. Your role is to provide clear, authoritative answers while maintaining accuracy through precise citations.

            
            Document Context:
            {document_context}
            
            Previous Conversation:
            {formatted_history}
            
            User's Message:
            {message}
            """
        prompt += """
        CITATION FORMAT:
        1. When referencing specific clauses or sections, always include the actual text content within [[double brackets]], not the clause numbers.
        Example: "The agreement states [[The party shall be liable for all damages]]"

        2. Add the filename after the citation using {{filename}}:
        Example: "As specified in [[The party shall be liable]]{{Agreement.pdf}}"

        3. For multiple related references, use them separately:
        - Single reference: [[The Seller shall deliver...]]{{Agreement.pdf}}
        - Multiple references: [[The Buyer agrees to pay...]]{{Agreement.pdf}} and [[All disputes shall be...]]{{Agreement.pdf}}

        CITATION GUIDELINES:
        1. Provide thorough analysis in complete sentences and paragraphs
        2. Support your analysis with relevant citations at the end of each point
        3. Citations should follow this format: [[exact text from document]]{{filename}}
        4. Never use citations as replacements for analysis
        5. Structure your response as:
               - Clear explanation/analysis of the point
               - Supporting citation at the end of the point
               
        - Always use the exact text as it appears in the document, do not change it or paraphrase it.
        - Never include formatting characters (**, `, etc.) in citations
        - Keep citations concise (30-40 characters)
        - NEVER combine multiple references within a single bracket like [[amongst: Essilor India Private Limited...and The Persons of the Gupta Family]], use proper formatting and only one exact citation per double bracket.
        - Do not use ellipsis (...), just use the first part of the text
        - Always include the filename after each citation WITHOUT ANY SPACE/GAP


        EXAMPLE RESPONSE:
        The contract includes **important provisions** about liability [[The party shall be liable]]{{Agreement.pdf}} and termination [[Agreement may be terminated]]{{Agreement.pdf}}.
        """

        # Get response using Gemini
        result = gemini_call("",prompt)
        
        return Response({
            'success': True,
            'response': result
        })
        
    except Exception as e:
        logger.exception("Error in chat endpoint")
        return Response({
            'error': str(e)
        }, status=500)