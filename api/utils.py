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

# Add these at the top level with other constants
DOCUMENT_TYPES = [
    "Asset Purchase Agreement",
    "Collaboration Agreement",
    "Confidentiality Agreement",
    "Copyright Assignment Agreement",
    "Escrow Agreement",
    "Franchise Agreement",
    "Indemnification Agreement",
    "Joint Venture Agreement",
    "Lease Agreement",
    "Loan Agreement",
    "Loan Purchase Agreement",
    "Investment Agreement",
    "Share Purchase Agreement",
    "Non-Compete Agreement",
    "Non-Disclosure Agreement (NDA)",
    "Partnership Agreement",
    "Pledge Agreement",
    "Real Estate Agreement to Sell",
    "Real Estate Purchase Agreement",
    "Shareholders' Agreement",
    "Services Agreement",
    "Manufacturing Agreement",
    "Tolling Agreement",
    "Slump Sale Agreement",
    "Patent Assignment Agreement",
    "Technology License Agreement"
]

SHORT_SUMMARY_PROMPTS = {
    "Asset Purchase Agreement": """Generate a summary for this Asset Purchase Agreement focusing on: 1. Clear identification of assets being transferred and any liabilities. 2. Representations and warranties provided by the seller, including guarantees on asset quality. 3. Conditions precedent to closing, including third-party consents. 4. Purchase price structure, including adjustments and payment mechanisms. 5. Indemnification obligations and remedies for breaches. 6. Retained vs. assumed liabilities. 7. Tax implications and allocation responsibilities. 8. Post-closing obligations, including warranties and asset transition.""",
    
    "Collaboration Agreement": """Generate a comprehensive summary of this Collaboration Agreement covering: 1. Scope of collaboration, including roles and responsibilities of each party. 2. Financial contributions, profit-sharing mechanisms, and cost-sharing arrangements. 3. Ownership and usage rights for intellectual property created during the collaboration. 4. Confidentiality and data-sharing obligations. 5. Termination conditions, including scenarios leading to termination and effects. 6. Dispute resolution mechanisms, including arbitration or mediation. 7. Milestones and deliverables with deadlines. 8. Exclusivity clauses, if any. 9. Compliance with laws and regulations relevant to the collaboration.""",
    
    "Confidentiality Agreement": """Summarize this Confidentiality Agreement by detailing: 1. Scope and definition of confidential information. 2. Obligations of receiving and disclosing parties to protect and manage information. 3. Permitted disclosures, including exceptions and obligations during legal proceedings. 4. Duration of confidentiality obligations. 5. Return or destruction of information upon termination. 6. Restrictions on reverse engineering, copying, or misuse. 7. Remedies and penalties for breaches, including liquidated damages. 8. Governing law and jurisdiction for resolving disputes. 9. Provisions for third-party involvement (e.g., subcontractors).""",
    
    "Copyright Assignment Agreement": """Summarize this Copyright Assignment Agreement with focus on: 1. Scope of rights being transferred, including moral rights and associated limitations. 2. Representations and warranties by the assignor regarding ownership and non-infringement. 3. Consideration/payment terms for the transfer. 4. Post-assignment obligations, such as ongoing assistance or consultation. 5. Indemnity clauses covering infringement claims or disputes. 6. Termination rights and associated obligations. 7. Retained rights or restrictions on assigned works. 8. Governing law and dispute resolution mechanisms. 9. Documentation and process requirements for completing the assignment.""",
    
    "Escrow Agreement": """Generate a comprehensive summary of this Escrow Agreement, including: 1. Roles and responsibilities of the escrow agent. 2. Conditions for release of escrowed items, including funds or documents. 3. Specific events triggering default and dispute resolution mechanisms. 4. Liability limitations and indemnification clauses for the escrow agent. 5. Fees and expenses borne by the parties for escrow services. 6. Termination provisions, including conditions under which escrow is released or continued. 7. Governing law and jurisdiction. 8. Interest rights on escrowed funds. 9. Security measures and obligations to ensure safety of escrowed items.""",
    
    "Franchise Agreement": """Summarize this Franchise Agreement focusing on: 1. Franchise fees, royalty structures, and payment terms. 2. Territorial rights, including exclusivity provisions. 3. Obligations of the franchisor to provide training, support, and access to brand resources. 4. Obligations of the franchisee to comply with operational and brand standards. 5. Use of trademarks and intellectual property provided by the franchisor. 6. Advertising and marketing obligations. 7. Termination conditions and renewal options, including associated fees. 8. Transferability and assignment of franchise rights. 9. Compliance with franchising regulations and dispute resolution methods.""",
    
    "Indemnification Agreement": """Generate a detailed summary of this Indemnification Agreement covering: 1. Scope of indemnification, including liabilities covered and exclusions. 2. Trigger events leading to indemnity, such as breach or third-party claims. 3. Obligations of the indemnifying party to defend, reimburse, or settle claims. 4. Liability caps and limitations, if any. 5. Cross-indemnification clauses. 6. Governing law and jurisdiction for resolving disputes. 7. Processes for claiming and substantiating indemnification. 8. Third-party involvement and rights to seek recovery from insurance or subcontractors. 9. Duration of indemnification obligations post-agreement termination.""",
    
    "Joint Venture Agreement": """Generate a comprehensive legal summary of this Joint Venture Agreement by covering: 1. Establishment details, including the formation, registration, and structure of the joint venture entity. 2. Scope of business operations, including objectives and limitations. 3. Share capital structure and conditions for increasing or raising additional capital. 4. Restrictions, procedures, and approvals for transferring shares or ownership interests. 5. Governance structure, including roles of directors, management responsibilities, and decision-making authority. 6. Conduct expectations, compliance policies, and operational standards. 7. Intellectual property ownership, licensing, and usage rights. 8. Non-compete and exclusivity clauses. 9. Deadlock resolution mechanisms, such as buyouts, escalation, or arbitration. 10. Dividend distribution policy, including timing and conditions. 11. Termination provisions, including post-termination obligations. 12. Remedies and penalties for breaches. 13. Liability and indemnification obligations.""",
    
    "Lease Agreement": """Summarize this Lease Agreement focusing on: 1. Rent amount, payment schedule, and conditions for the security deposit. 2. Duration of the lease, including renewal and termination options. 3. Obligations for property maintenance by the tenant and the landlord. 4. Use restrictions, such as residential, commercial, or other specific uses. 5. Provisions for subletting or assignment of lease rights. 6. Conditions for early termination and consequences for breach. 7. Dispute resolution mechanisms and governing law. 8. Inspection and access rights of the landlord. 9. Modifications allowed to the property and related responsibilities.""",
    
    "Loan Agreement": """Summarize this Loan Agreement focusing on: 1. Principal amount, interest rate, and repayment schedule. 2. Prepayment penalties and options. 3. Security or collateral requirements. 4. Events of default and remedies available to the lender. 5. Financial covenants and borrower obligations. 6. Representations and warranties of the borrower. 7. Indemnification and liability limitations. 8. Governing law and jurisdiction. 9. Cross-default clauses and consequences.""",
    
    "Loan Purchase Agreement": """Summarize this Loan Purchase Agreement focusing on: 1. Purchase price and payment terms. 2. Representations and warranties regarding loan quality. 3. Recourse and indemnification provisions in case of defaults. 4. Assignment and transfer restrictions. 5. Post-closing obligations, including notice to borrowers. 6. Conditions precedent for completing the transaction. 7. Governing law and dispute resolution mechanisms.""",
    
    "Investment Agreement": """Summarize this Investment Agreement focusing on: 1. Amount of investment and valuation. 2. Rights attached to the investment, such as equity, debt, or convertible instruments. 3. Investor rights, including board seats, information access, and veto powers. 4. Exit mechanisms, such as IPO or buyback options. 5. Anti-dilution provisions and pre-emption rights. 6. Conditions precedent to the investment. 7. Representations and warranties by the investee. 8. Indemnity provisions for breaches. 9. Drag-along and tag-along rights.""",
    
    "Share Purchase Agreement": """Summarize this Share Purchase Agreement focusing on: 1. Purchase price, payment terms, and adjustments. 2. Representations and warranties provided by the seller. 3. Conditions precedent to closing. 4. Transfer of title and ownership. 5. Post-closing covenants, including warranties and indemnification. 6. Restrictions on future share transfers. 7. Governing law and dispute resolution.""",
    
    "Non-Compete Agreement": """Summarize this Non-Compete Agreement focusing on: 1. Scope of restricted activities and industries. 2. Duration and geographic limits of the restrictions. 3. Consideration or compensation for the non-compete obligations. 4. Enforceability based on jurisdictional laws. 5. Exceptions to restrictions, if any. 6. Remedies and penalties for breaches.""",
    
    "Non-Disclosure Agreement (NDA)": """Summarize this Non-Disclosure Agreement focusing on: 1. Definition and scope of confidential information. 2. Permitted disclosures and obligations of both parties. 3. Duration of confidentiality obligations. 4. Consequences for breach, including damages or termination. 5. Governing law and jurisdiction.""",
    
    "Partnership Agreement": """Summarize this Partnership Agreement focusing on: 1. Contributions (capital, assets, skills) by each partner. 2. Profit and loss sharing ratios. 3. Roles and responsibilities of each partner. 4. Decision-making processes and voting rights. 5. Exit and buyout provisions. 6. Non-compete clauses and dispute resolution mechanisms. 7. Liability and indemnification obligations.""",
    
    "Pledge Agreement": """Summarize this Pledge Agreement focusing on: 1. Identification of pledged assets or collateral. 2. Conditions for enforcement of the pledge. 3. Events of default triggering enforcement. 4. Rights of the pledgee upon default, including sale or retention. 5. Obligations of the pledgor to maintain collateral value. 6. Governing law and impact of bankruptcy.""",
    
    "Real Estate Agreement to Sell": """Summarize this Real Estate Agreement to Sell focusing on: 1. Description of the property, including boundaries and key details. 2. Purchase price and payment schedule. 3. Conditions precedent to sale, such as title clearance. 4. Representations and warranties by the seller. 5. Post-sale obligations and dispute resolution mechanisms.""",
    
    "Real Estate Purchase Agreement": """Summarize this Real Estate Purchase Agreement focusing on: 1. Purchase price and financial terms. 2. Description of the property, including legal and physical characteristics. 3. Title and encumbrance checks. 4. Conditions precedent to closing, such as financing or permits. 5. Dispute resolution and governing law.""",
    
    "Shareholders' Agreement": """Summarize this Shareholders' Agreement focusing on: 1. Voting rights and decision-making processes. 2. Dividend policies and profit-sharing arrangements. 3. Restrictions on share transfers, such as right of first refusal or tag-along rights. 4. Board representation and management roles. 5. Exit provisions and anti-dilution clauses.""",
    
    "Services Agreement": """Summarize this Services Agreement focusing on: 1. Scope and timeline of services to be provided. 2. Payment terms and invoicing procedures. 3. Deliverables and performance standards. 4. Termination conditions and notice requirements. 5. Liability and indemnification provisions. 6. Intellectual property rights for service deliverables.""",
    
    "Manufacturing Agreement": """Summarize this Manufacturing Agreement focusing on: 1. Scope and specifications of manufacturing services. 2. Quality control, testing standards, and inspection rights. 3. Payment terms and cost adjustments. 4. Minimum order quantities and lead times. 5. Intellectual property rights and indemnification clauses. 6. Termination provisions and notice requirements.""",
    
    "Tolling Agreement": """Summarize this Tolling Agreement focusing on: 1. Scope and responsibilities for tolling services. 2. Ownership of raw materials and finished products. 3. Quality control and inspection standards. 4. Payment terms and penalties for delays. 5. Termination conditions and dispute resolution.""",
    
    "Slump Sale Agreement": """Summarize this Slump Sale Agreement focusing on: 1. Transfer of business as a going concern, including assets and liabilities. 2. Valuation of the business and purchase price. 3. Transfer of employees and contractual obligations. 4. Representations and warranties by the seller. 5. Indemnity for liabilities such as taxes and debts.""",
    
    "Patent Assignment Agreement": """Summarize this Patent Assignment Agreement focusing on: 1. Identification of patents being transferred. 2. Consideration/payment terms. 3. Representations and warranties of ownership. 4. Indemnity provisions for infringement claims. 5. Obligations for transferring related documentation.""",
    
    "Technology License Agreement": """Summarize this Technology License Agreement focusing on: 1. Scope of licensed technology, including territorial and exclusivity rights. 2. Payment terms and royalty structure. 3. Restrictions on sublicensing or misuse of the technology. 4. Termination provisions and post-termination obligations. 5. Indemnity and audit rights for royalty calculations."""
}

# Add a GENERAL_SUMMARY_PROMPT constant
GENERAL_SUMMARY_PROMPT = """
Generate a comprehensive summary of this legal document focusing on:

1. Document Type and Purpose
- Identify the primary purpose and nature of the agreement
- Key objectives and intended outcomes

2. Parties Involved
- Identify all parties and their roles
- Key relationships and obligations

3. Key Terms and Conditions
- Main rights and obligations of each party
- Critical deadlines and timelines
- Financial terms and payment obligations
- Performance requirements and standards

4. Risk Allocation
- Liability provisions
- Indemnification obligations
- Insurance requirements
- Warranty and representation commitments

5. Important Clauses
- Termination conditions
- Default scenarios and remedies
- Change or modification provisions
- Assignment and transfer rights
- Dispute resolution mechanisms

6. Special Provisions
- Any unique or notable terms
- Industry-specific requirements
- Regulatory compliance obligations

Present the summary in clear, actionable points that highlight business impact.
Use professional language and maintain a logical flow.
Include specific references to relevant sections where appropriate.
"""


RISK_ANALYSIS_PROMPTS = {

    "Asset Purchase Agreement": """Analyze key risks in this Asset Purchase Agreement focusing on: 1. Hidden liabilities or undisclosed obligations. 2. Gaps in representations and warranties. 3. Inadequate indemnification provisions. 4. Ambiguous asset descriptions or exclusions. 5. Environmental or regulatory compliance risks. 6. Employment and benefit plan obligations. 7. Intellectual property ownership issues. 8. Tax exposure risks. 9. Material contract assignment issues. 10. Post-closing integration risks.""",

    "Collaboration Agreement": """Assess risk factors in this Collaboration Agreement focusing on: 1. Unclear scope definitions leading to scope creep. 2. Inadequate IP protection mechanisms. 3. Ambiguous profit-sharing formulas. 4. Weak confidentiality provisions. 5. Exit strategy complications. 6. Resource commitment ambiguities. 7. Performance measurement uncertainties. 8. Regulatory compliance gaps. 9. Conflict resolution mechanism weaknesses. 10. Third-party liability exposure.""",

    "Confidentiality Agreement": """Evaluate risk exposure in this Confidentiality Agreement focusing on: 1. Ambiguous definition of confidential information. 2. Inadequate protection mechanisms. 3. Weak enforcement provisions. 4. Unclear duration terms. 5. Exception clause loopholes. 6. Information return/destruction verification issues. 7. Employee/contractor compliance risks. 8. Technology and data security gaps. 9. International jurisdiction complications. 10. Trade secret protection adequacy.""",

    "Copyright Assignment Agreement": """Examine risk factors in this Copyright Assignment Agreement focusing on: 1. Incomplete rights transfer provisions. 2. Prior licensing complications. 3. Moral rights issues. 4. Work-for-hire ambiguities. 5. Future rights uncertainty. 6. Third-party infringement risks. 7. Registration and documentation gaps. 8. International protection issues. 9. Derivative works rights. 10. Reversion rights complications.""",

    "Escrow Agreement": """Review risk elements in this Escrow Agreement focusing on: 1. Release condition ambiguities. 2. Escrow agent liability limitations. 3. Asset verification inadequacies. 4. Default trigger uncertainties. 5. Investment return risks. 6. Administrative cost allocation issues. 7. Dispute resolution weaknesses. 8. Security measure adequacy. 9. Regulatory compliance gaps. 10. Bankruptcy implications.""",

    "Franchise Agreement": """Analyze risk exposure in this Franchise Agreement focusing on: 1. Territory definition ambiguities. 2. System standard compliance issues. 3. Training and support inadequacies. 4. Royalty calculation disputes. 5. Brand protection weaknesses. 6. Competition restrictions enforceability. 7. Termination clause fairness. 8. Supply chain vulnerabilities. 9. Customer data protection issues. 10. Regulatory compliance gaps.""",

    "Indemnification Agreement": """Evaluate risk factors in this Indemnification Agreement focusing on: 1. Coverage gaps. 2. Trigger event ambiguities. 3. Cap adequacy issues. 4. Time limitation problems. 5. Defense obligation uncertainties. 6. Insurance coordination complications. 7. Notice requirement issues. 8. Exclusion clause breadth. 9. Cross-indemnification fairness. 10. Enforcement jurisdiction challenges.""",

    "Joint Venture Agreement": """Assess risk elements in this Joint Venture Agreement focusing on: 1. Control and management disputes. 2. Resource commitment ambiguities. 3. Technology transfer risks. 4. Profit sharing formula issues. 5. Exit strategy complications. 6. Intellectual property ownership disputes. 7. Competition concerns. 8. Regulatory compliance gaps. 9. Cultural and operational integration risks. 10. Deadlock resolution adequacy.""",

    "Lease Agreement": """Examine risk factors in this Lease Agreement focusing on: 1. Maintenance responsibility ambiguities. 2. Operating cost allocation issues. 3. Use restriction adequacy. 4. Sublease and assignment risks. 5. Insurance coverage gaps. 6. Environmental compliance issues. 7. Early termination complications. 8. Property modification rights. 9. Security deposit adequacy. 10. Default remedy effectiveness.""",

    "Loan Agreement": """Review risk exposure in this Loan Agreement focusing on: 1. Security interest adequacy. 2. Default trigger ambiguities. 3. Interest rate adjustment issues. 4. Prepayment penalty fairness. 5. Covenant compliance monitoring. 6. Cross-default implications. 7. Collateral valuation risks. 8. Guarantor liability issues. 9. Regulatory compliance gaps. 10. Bankruptcy protection adequacy.""",

    "Loan Purchase Agreement": """Analyze risk factors in this Loan Purchase Agreement focusing on: 1. Portfolio quality issues. 2. Servicing transfer risks. 3. Representation and warranty adequacy. 4. Repurchase obligation triggers. 5. Documentation completeness. 6. Regulatory compliance gaps. 7. Borrower notification issues. 8. Interest rate adjustment risks. 9. Default history implications. 10. Collection practice liability.""",

    "Investment Agreement": """Assess risk elements in this Investment Agreement focusing on: 1. Valuation dispute potential. 2. Control right inadequacies. 3. Exit strategy complications. 4. Anti-dilution protection gaps. 5. Information right limitations. 6. Tag-along/drag-along issues. 7. Board representation risks. 8. Dividend policy disputes. 9. Future financing conflicts. 10. Regulatory compliance issues.""",

    "Share Purchase Agreement": """Evaluate risk exposure in this Share Purchase Agreement focusing on: 1. Price adjustment mechanism issues. 2. Due diligence limitation impacts. 3. Warranty coverage gaps. 4. Earn-out calculation disputes. 5. Tax liability allocation. 6. Employee retention risks. 7. Change of control implications. 8. Third-party consent requirements. 9. Competition law compliance. 10. Post-closing integration issues.""",

    "Non-Compete Agreement": """Review risk factors in this Non-Compete Agreement focusing on: 1. Scope definition ambiguities. 2. Geographic limitation adequacy. 3. Duration reasonableness. 4. Consideration sufficiency. 5. Enforcement jurisdiction issues. 6. Industry definition breadth. 7. Employee restriction fairness. 8. Customer solicitation terms. 9. Trade secret protection adequacy. 10. Reformation clause effectiveness.""",

    "Non-Disclosure Agreement (NDA)": """Analyze risk elements in this NDA focusing on: 1. Information definition gaps. 2. Time limitation adequacy. 3. Exception clause loopholes. 4. Return/destruction verification. 5. Employee/contractor coverage. 6. Residual knowledge issues. 7. Data security requirements. 8. Breach notification provisions. 9. International protection issues. 10. Trade secret preservation.""",

    "Partnership Agreement": """Examine risk exposure in this Partnership Agreement focusing on: 1. Contribution valuation disputes. 2. Profit/loss allocation fairness. 3. Decision-making deadlocks. 4. Exit mechanism adequacy. 5. Partner liability issues. 6. Succession planning gaps. 7. Intellectual property rights. 8. Competition restrictions. 9. Dissolution process clarity. 10. Tax allocation issues.""",

    "Pledge Agreement": """Assess risk factors in this Pledge Agreement focusing on: 1. Collateral description adequacy. 2. Perfection requirement compliance. 3. Default trigger clarity. 4. Valuation mechanism issues. 5. Maintenance obligation specificity. 6. Security interest priority. 7. Enforcement procedure gaps. 8. Release condition clarity. 9. Insurance coverage adequacy. 10. Bankruptcy implications.""",

    "Real Estate Agreement to Sell": """Review risk elements in this Real Estate Agreement focusing on: 1. Title defect issues. 2. Environmental liability exposure. 3. Survey and boundary disputes. 4. Zoning compliance gaps. 5. Financing contingency adequacy. 6. Inspection right limitations. 7. Closing condition issues. 8. Property condition disclosures. 9. Permit transfer problems. 10. Tax assessment implications.""",

    "Real Estate Purchase Agreement": """Evaluate risk factors in this Real Estate Purchase Agreement focusing on: 1. Due diligence limitations. 2. Title insurance gaps. 3. Environmental liability issues. 4. Zoning compliance problems. 5. Property condition disputes. 6. Financing contingency adequacy. 7. Closing delay implications. 8. Tenant right impacts. 9. Service contract assumptions. 10. Tax proration issues.""",

    "Shareholders' Agreement": """Analyze risk exposure in this Shareholders' Agreement focusing on: 1. Control provision effectiveness. 2. Minority protection adequacy. 3. Transfer restriction enforceability. 4. Deadlock resolution mechanisms. 5. Tag-along/drag-along rights. 6. Information access issues. 7. Dividend policy disputes. 8. Exit mechanism practicality. 9. Valuation methodology issues. 10. Regulatory compliance gaps.""",

    "Services Agreement": """Assess risk elements in this Services Agreement focusing on: 1. Scope definition ambiguities. 2. Performance standard issues. 3. Payment term disputes. 4. Termination right adequacy. 5. Liability limitation fairness. 6. Intellectual property ownership. 7. Data protection compliance. 8. Force majeure coverage. 9. Insurance requirement adequacy. 10. Subcontractor liability issues.""",

    "Manufacturing Agreement": """Review risk factors in this Manufacturing Agreement focusing on: 1. Specification clarity issues. 2. Quality standard disputes. 3. Delivery timeline problems. 4. Cost adjustment mechanisms. 5. Inventory liability allocation. 6. Intellectual property protection. 7. Warranty coverage adequacy. 8. Tooling ownership issues. 9. Force majeure implications. 10. Regulatory compliance gaps.""",

    "Tolling Agreement": """Examine risk exposure in this Tolling Agreement focusing on: 1. Processing standard disputes. 2. Material handling liability. 3. Loss allocation issues. 4. Quality control adequacy. 5. Capacity commitment problems. 6. Cost adjustment mechanisms. 7. Equipment maintenance responsibility. 8. Environmental compliance risks. 9. Insurance coverage gaps. 10. Force majeure implications.""",

    "Slump Sale Agreement": """Evaluate risk elements in this Slump Sale Agreement focusing on: 1. Asset identification completeness. 2. Liability transfer issues. 3. Employee transition problems. 4. Tax implication uncertainty. 5. Contract assignment gaps. 6. Regulatory approval requirements. 7. Due diligence limitations. 8. Valuation dispute potential. 9. Warranty coverage adequacy. 10. Post-closing adjustment mechanisms.""",

    "Patent Assignment Agreement": """Analyze risk factors in this Patent Assignment Agreement focusing on: 1. Patent scope definition issues. 2. Prior license complications. 3. Improvement rights allocation. 4. Foreign registration gaps. 5. Enforcement right limitations. 6. Reservation clause ambiguities. 7. Warranty coverage adequacy. 8. Assignment recording requirements. 9. Inventor cooperation issues. 10. Regulatory compliance gaps.""",

    "Technology License Agreement": """Review risk exposure in this Technology License Agreement focusing on: 1. License scope ambiguities. 2. Usage restriction enforceability. 3. Performance requirement issues. 4. Royalty calculation disputes. 5. Source code escrow adequacy. 6. Support obligation clarity. 7. Update/upgrade rights. 8. Patent infringement protection. 9. Termination implications. 10. Regulatory compliance gaps."""
}
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
    """
    Classifies the document type and return the classified document type from the predefined list.
    """
    classification_prompt = """
    You are a legal document classifier. Based on the document provided, classify it into ONE of the following categories:
    
    {}
    
    Respond ONLY with the exact category name from the list above. If the document doesn't match any category exactly, 
    choose the closest match. Provide ONLY the category name, no other text or explanation.
    """.format("\n".join(DOCUMENT_TYPES))
    
    try:
        result = gemini_call(text, classification_prompt)
        classified_type = result.strip()
        if classified_type in DOCUMENT_TYPES:
            return classified_type
        # If response doesn't match exactly, return None to trigger general prompt
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
    
    if analysis_type == 'shortSummary':
        # First classify the document
        doc_type = classify_document(text)
        print(f"classified into {doc_type}")
        logger.info(f"Document classified as: {doc_type}")
        
        if doc_type and doc_type in SHORT_SUMMARY_PROMPTS:
            # Get the specific prompt for this document type
            type_specific_prompt = SHORT_SUMMARY_PROMPTS[doc_type]
            prompt = f"""
            This document has been classified as: {doc_type}
            
            {type_specific_prompt}
            
            Present the summary in clear, actionable points that highlight business impact.
            Use professional language and maintain a logical flow.
            """
        else:
            # Use general summary prompt if classification fails or type not found
            logger.info("Using general summary prompt")
            prompt = GENERAL_SUMMARY_PROMPT
    
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
        doc_type = classify_document(text)
        logger.info(f"Document classified as: {doc_type}")
        
        if doc_type and doc_type in RISK_ANALYSIS_PROMPTS:
            type_specific_prompt = RISK_ANALYSIS_PROMPTS[doc_type]
        else:
            logger.info("Using general risk analysis prompt")
            type_specific_prompt = RISK_ANALYSIS_PROMPTS["Services Agreement"]
        
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
        You are a legal AI assistant. Analyze the provided documents and chat history to answer questions.
        
        The input will be structured as:
        1. Document contents (marked with [1], [2], etc. present at the start of each document)
        2. Previous conversation history (if any)
        3. Current query
        
        Guidelines:
        1. Consider both the documents and chat history for context
        2. If referring to previous messages, be explicit
        3. If the query relates to specific documents, cite them using their numbers [1], [2], etc.
        4. Maintain professional tone
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
        Only use double asterisks for section headers and strictly no other use of asterisks.
        Only provide the draft in your response, do not mention anything else.
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
        return result.strip().lower() == 'yes'
    except Exception as e:
        logger.exception("Error checking for common party")
        return False
    
def gemini_call(text, prompt):
    logger.info("Calling Gemini API")
    
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



