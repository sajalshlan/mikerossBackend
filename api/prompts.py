from typing import Dict

# Document type list
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
GENERAL_SHORT_SUMMARY_PROMPT = """
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

GENERAL_RISK_ANALYSIS_PROMPT = """
Analyze this legal document for risks considering the following aspects:

1. Contractual Framework Risks
- Unclear or ambiguous terms and definitions
- Gaps in essential provisions
- Inconsistent or conflicting clauses
- Enforceability concerns
- Jurisdiction and governing law issues
- Assignment and transfer restrictions
- Amendment and modification provisions
- Force majeure coverage adequacy

2. Performance & Operational Risks
- Delivery and timeline obligations
- Quality and performance standards
- Resource and capability requirements
- Technical specifications compliance
- Reporting and monitoring mechanisms
- Subcontracting and delegation rights
- Integration and implementation challenges
- Operational disruption potential

3. Financial Exposure
- Payment terms and conditions
- Price adjustment mechanisms
- Currency and exchange rate exposure
- Tax implications and liabilities
- Cost allocation and sharing
- Financial security requirements
- Insurance coverage adequacy
- Hidden or contingent costs

4. Compliance & Regulatory
- Industry-specific regulations
- Cross-border compliance requirements
- Licensing and permit obligations
- Environmental regulations
- Data protection and privacy laws
- Anti-corruption and sanctions
- Employment and labor laws
- Reporting obligations

5. Intellectual Property & Confidentiality
- IP ownership and rights allocation
- Technology transfer restrictions
- Confidentiality obligations
- Trade secret protection
- Third-party IP infringement
- Data security requirements
- Usage and exploitation rights
- Post-termination restrictions

6. Liability & Indemnification
- Limitation of liability provisions
- Indemnification scope and exclusions
- Warranty and representation coverage
- Insurance requirements
- Third-party claims exposure
- Product liability risks
- Professional liability exposure
- Cross-indemnification issues

7. Termination & Exit
- Termination rights and triggers
- Notice requirements
- Post-termination obligations
- Transition assistance provisions
- Asset and data return/destruction
- Survival clauses
- Wind-down procedures
- Continuing obligations

8. Strategic & Business
- Market position impact
- Competitive restrictions
- Relationship management
- Reputational exposure
- Growth and expansion limitations
- Strategic alignment risks
- Business model impact
- Stakeholder relationships

9. Dispute Resolution
- Choice of forum
- Arbitration provisions
- Mediation requirements
- Emergency relief options
- Evidence preservation
- Cost allocation
- Enforcement mechanisms
- Cross-border considerations

10. Change Management
- Change control procedures
- Technology evolution impact
- Market condition changes
- Regulatory environment shifts
- Business requirement changes
- Personnel and management changes
- Corporate restructuring impact
- Industry transformation effects

For each risk category:
- Identify specific risks applicable to each party
- Consider both direct and indirect impacts
- Evaluate short-term and long-term implications
- Assess interdependencies between risks
- Consider market and industry context
- Review historical precedents and common pitfalls
- Analyze potential cascading effects
- Consider worst-case scenarios
"""

LONG_SUMMARY_PROMPTS = {
    "Asset Purchase Agreement": """Conduct a comprehensive legal analysis of this Asset Purchase Agreement with detailed attention to:

1. Asset Transfer Structure and Mechanics
   - Detailed inventory and classification of assets being transferred
   - Treatment of tangible vs. intangible assets
   - Assignment mechanisms for contracts, licenses, and permits
   - Treatment of excluded assets and rationale
   - Impact on ongoing operations

2. Purchase Price and Financial Terms
   - Detailed breakdown of consideration structure
   - Working capital adjustments and calculations
   - Earnout provisions and conditions
   - Purchase price allocation implications
   - Tax efficiency analysis

3. Representations and Warranties
   - Scope and materiality qualifiers
   - Knowledge definitions and attribution
   - Survival periods and limitations
   - Special indemnities and carve-outs
   - Relationship to due diligence findings

4. Risk Allocation Mechanisms
   - Indemnification framework and caps
   - Basket types and deductibles
   - Special indemnities and carve-outs
   - Insurance coverage coordination
   - Third-party claim procedures

5. Operational and Integration Considerations
   - Employee transition arrangements
   - Customer and vendor notifications
   - Systems and IT integration requirements
   - Regulatory compliance obligations
   - Post-closing cooperation requirements

6. Conditions Precedent and Closing Mechanics
   - Required third-party consents
   - Regulatory approvals needed
   - Due diligence conditions
   - Bring-down requirements
   - Closing deliverables and timing

7. Tax and Accounting Implications
   - Structure-specific tax consequences
   - Transfer tax responsibilities
   - Tax clearance requirements
   - Accounting treatment considerations
   - Financial statement impacts

8. Post-Closing Obligations
   - Transition services arrangements
   - Non-compete and non-solicit terms
   - Customer relationship management
   - Warranty obligations
   - Record retention requirements""",

    "Collaboration Agreement": """Conduct a comprehensive analysis of this Collaboration Agreement addressing:

1. Scope and Objectives
   - Detailed project scope definition
   - Roles and responsibilities matrix
   - Performance metrics and standards
   - Timeline and milestone structure
   - Resource allocation requirements

2. Governance Structure
   - Project management framework
   - Decision-making hierarchy
   - Steering committee composition
   - Escalation procedures
   - Change management process

3. Financial Framework
   - Cost sharing methodology
   - Revenue allocation model
   - Budget management process
   - Expense approval procedures
   - Financial reporting requirements

4. Intellectual Property Rights
   - Background IP identification
   - Foreground IP ownership
   - Joint IP management
   - Licensing arrangements
   - Commercialization rights

5. Risk Management
   - Liability framework
   - Indemnification obligations
   - Insurance requirements
   - Force majeure provisions
   - Dispute resolution mechanisms

6. Data and Confidentiality
   - Information sharing protocols
   - Data protection measures
   - Third-party disclosure rules
   - Security requirements
   - Post-termination obligations

7. Term and Termination
   - Duration analysis
   - Renewal mechanisms
   - Termination triggers
   - Wind-down procedures
   - Survival provisions""",

    "Confidentiality Agreement": """Perform a detailed analysis of this Confidentiality Agreement focusing on:

1. Information Classification and Scope
   - Definition of confidential information
   - Exclusions and carve-outs
   - Marking requirements
   - Trade secret designation
   - Public domain considerations
   - Derivative information treatment

2. Protection Mechanisms
   - Security protocols
   - Access controls
   - Storage requirements
   - Transmission standards
   - Destruction procedures
   - Breach notification obligations

3. Usage Rights and Restrictions
   - Permitted purposes
   - Internal distribution limits
   - Affiliate sharing rights
   - Contractor obligations
   - Sub-licensing restrictions
   - Residual knowledge rights

4. Compliance Framework
   - Training requirements
   - Audit rights
   - Reporting obligations
   - Documentation standards
   - Certification procedures
   - Return/destruction obligations

5. Term and Survival
   - Duration analysis
   - Extension mechanisms
   - Return/destruction obligations
   - Surviving provisions
   - Ongoing obligations

6. Enforcement and Remedies
   - Breach notification
   - Injunctive relief
   - Damages calculation
   - Jurisdictional considerations
   - Alternative dispute resolution""",

    "Copyright Assignment Agreement": """Conduct an in-depth analysis of this Copyright Assignment Agreement addressing:

1. Rights Transfer Scope
   - Works covered
   - Territory analysis
   - Duration considerations
   - Moral rights treatment
   - Derivative works rights
   - Future works provisions

2. Assignment Mechanics
   - Transfer documentation
   - Registration requirements
   - Recording obligations
   - Notice requirements
   - Third-party consents
   - Execution formalities

3. Warranties and Representations
   - Ownership verification
   - Non-infringement assurance
   - Third-party rights clearance
   - Prior licenses analysis
   - Encumbrance disclosure
   - Chain of title verification

4. Consideration Structure
   - Payment terms
   - Royalty arrangements
   - Tax implications
   - Currency considerations
   - Payment security
   - Adjustment mechanisms

5. Post-Assignment Obligations
   - Further assurance requirements
   - Cooperation duties
   - Maintenance obligations
   - Enforcement support
   - Documentation delivery
   - Ongoing assistance

6. Risk Allocation
   - Indemnification provisions
   - Limitation of liability
   - Insurance requirements
   - Third-party claims
   - Survival periods""",

    "Escrow Agreement": """Analyze this Escrow Agreement with detailed focus on:

1. Escrow Structure and Mechanics
   - Asset identification
   - Deposit mechanisms
   - Verification procedures
   - Access protocols
   - Release conditions
   - Amendment procedures

2. Escrow Agent Obligations
   - Custodial duties
   - Investment authority
   - Reporting requirements
   - Communication protocols
   - Security measures
   - Conflict resolution

3. Release Mechanics
   - Trigger events
   - Notice requirements
   - Verification procedures
   - Dispute handling
   - Partial release provisions
   - Emergency procedures

4. Financial Arrangements
   - Fee structure
   - Cost allocation
   - Interest treatment
   - Tax considerations
   - Payment mechanics
   - Account management

5. Risk Management
   - Liability limitations
   - Indemnification provisions
   - Insurance requirements
   - Force majeure treatment
   - Termination rights
   - Successor agent provisions""",

    "Franchise Agreement": """Perform a comprehensive analysis of this Franchise Agreement addressing:

1. Franchise Grant and Territory
   - Territory definition
   - Exclusivity provisions
   - Reserved rights
   - Term and renewal
   - Development schedule
   - Site selection requirements

2. Financial Structure
   - Initial franchise fees
   - Ongoing royalties
   - Marketing contributions
   - Payment terms
   - Audit rights
   - Default remedies

3. Operational Requirements
   - System standards
   - Quality control
   - Training obligations
   - Supply chain requirements
   - Reporting obligations
   - Compliance monitoring

4. Brand Protection
   - Trademark usage
   - Trade dress compliance
   - Confidentiality obligations
   - Non-compete provisions
   - Social media policies
   - Customer data protection

5. Support Services
   - Initial assistance
   - Ongoing support
   - Training programs
   - Marketing support
   - Technology systems
   - Operations manual

6. Term and Termination
   - Initial term
   - Renewal conditions
   - Termination rights
   - Post-termination obligations
   - De-identification requirements""",

    "Indemnification Agreement": """Conduct a detailed analysis of this Indemnification Agreement focusing on:

1. Scope of Protection
   - Covered claims
   - Exclusions
   - Temporal limitations
   - Geographic restrictions
   - Beneficiary identification
   - Related party coverage

2. Indemnification Mechanics
   - Notice requirements
   - Defense obligations
   - Settlement authority
   - Cooperation duties
   - Cost advancement
   - Recovery rights

3. Financial Limitations
   - Coverage caps
   - Deductibles/retentions
   - Insurance coordination
   - Payment timing
   - Currency considerations
   - Tax treatment

4. Procedural Requirements
   - Claim notification
   - Defense selection
   - Settlement approval
   - Appeals process
   - Documentation requirements
   - Reporting obligations

5. Risk Allocation
   - Primary vs. excess coverage
   - Contribution rights
   - Subrogation rights
   - Set-off rights
   - Recovery allocation""",

    "Joint Venture Agreement": """Perform an in-depth analysis of this Joint Venture Agreement addressing:

1. Structure and Formation
   - Legal form and jurisdiction analysis
   - Capital structure and contributions
   - Registration and regulatory requirements
   - Tax efficiency considerations
   - Implementation timeline and steps
   - Pre-completion obligations

2. Governance Framework
   - Board composition and voting rights
   - Reserved matter provisions
   - Management structure and reporting lines
   - Operational control mechanisms
   - Deadlock resolution procedures
   - Information rights

3. Financial Framework
   - Initial and ongoing capital contributions
   - Working capital management
   - Dividend policy and distribution rules
   - Exit valuation mechanisms
   - Transfer pricing considerations
   - Funding obligations

4. Operational Controls
   - Business plan requirements
   - Budget approval process
   - Performance metrics and KPIs
   - Quality control standards
   - Compliance framework
   - Reporting requirements

5. Intellectual Property Rights
   - Background IP arrangements
   - Foreground IP ownership
   - License terms and restrictions
   - Improvements and modifications
   - Third-party IP implications
   - Technology transfer

6. Competition and Territory
   - Non-compete provisions
   - Market allocation
   - Customer restrictions
   - Geographic limitations
   - Antitrust compliance measures

7. Exit Mechanisms
   - Put and call options
   - Tag-along and drag-along rights
   - IPO provisions
   - Deadlock resolution
   - Termination triggers""",

    "Lease Agreement": """Analyze this Lease Agreement with detailed focus on:

1. Premises and Use
   - Property description
   - Permitted use
   - Common areas
   - Exclusive rights
   - Alterations rights
   - Environmental compliance

2. Financial Terms
   - Base rent structure
   - Operating expenses
   - Utilities allocation
   - Security deposit
   - Insurance obligations
   - Tax responsibilities

3. Maintenance and Operations
   - Landlord obligations
   - Tenant duties
   - Service contracts
   - Emergency procedures
   - Access rights
   - Building rules

4. Term and Renewal
   - Initial term
   - Extension options
   - Early termination rights
   - Holdover provisions
   - Surrender requirements
   - Restoration obligations

5. Assignment and Subletting
   - Consent requirements
   - Change of control
   - Recapture rights
   - Profit sharing
   - Release provisions
   - Assignment fees

6. Default and Remedies
   - Default triggers
   - Cure periods
   - Landlord remedies
   - Tenant remedies
   - Self-help rights
   - Force majeure""",

    "Loan Agreement": """Conduct a comprehensive analysis of this Loan Agreement with focus on:

1. Facility Structure
   - Loan type and purpose
   - Availability period
   - Draw-down conditions
   - Repayment mechanisms
   - Prepayment provisions
   - Interest calculation

2. Security Package
   - Collateral coverage analysis
   - Security documentation requirements
   - Perfection requirements
   - Priority arrangements
   - Enforcement mechanisms
   - Release conditions

3. Financial Covenants
   - Maintenance vs. incurrence tests
   - Definition analysis
   - Testing methodology
   - Cure rights
   - Cross-default implications
   - Reporting requirements

4. Representations and Warranties
   - Scope and materiality
   - Bring-down requirements
   - Knowledge qualifiers
   - Exception schedule analysis
   - Verification requirements
   - Update obligations

5. Events of Default
   - Default triggers
   - Grace periods
   - Cross-default provisions
   - Material adverse change analysis
   - Remedy provisions
   - Acceleration rights

6. Information Rights
   - Financial reporting requirements
   - Compliance certificates
   - Notice obligations
   - Access rights
   - Third-party verification
   - Confidentiality provisions""",

    "Loan Purchase Agreement": """Conduct a detailed analysis of this Loan Purchase Agreement addressing:

1. Portfolio Definition and Scope
   - Loan identification
   - Eligibility criteria
   - Cut-off date mechanics
   - Excluded loans
   - Document delivery
   - Due diligence rights

2. Purchase Terms
   - Price calculation
   - Adjustments mechanism
   - Payment structure
   - Settlement process
   - True sale analysis
   - Transfer mechanics

3. Representations and Warranties
   - Loan level warranties
   - Seller corporate warranties
   - Knowledge qualifiers
   - Survival periods
   - Materiality thresholds
   - Update requirements

4. Risk Allocation
   - Repurchase obligations
   - Indemnification provisions
   - Loss sharing
   - Insurance coordination
   - Servicing liability
   - Enforcement rights

5. Transfer Mechanics
   - Assignment requirements
   - Notice obligations
   - Recording requirements
   - Documentation standards
   - Borrower communication
   - Regulatory compliance""",

    "Investment Agreement": """Analyze this Investment Agreement with focus on:

1. Investment Structure
   - Security type
   - Valuation basis
   - Investment staging
   - Use of proceeds
   - Conditions precedent
   - Closing mechanics

2. Investor Rights
   - Board representation
   - Information rights
   - Approval rights
   - Registration rights
   - Pre-emptive rights
   - Tag-along rights

3. Economic Terms
   - Liquidation preference
   - Dividend rights
   - Anti-dilution protection
   - Conversion rights
   - Exit preferences
   - Pay-to-play provisions

4. Operational Controls
   - Business plan requirements
   - Budget approval
   - Key hire approval
   - Strategic decisions
   - Reporting obligations
   - Compliance requirements

5. Exit Rights
   - IPO obligations
   - Drag-along rights
   - Registration rights
   - Right of first refusal
   - Co-sale rights
   - Put option rights""",

    "Share Purchase Agreement": """Perform a comprehensive analysis of this Share Purchase Agreement addressing:

1. Transaction Structure
   - Share identification
   - Purchase price mechanics
   - Adjustment provisions
   - Escrow arrangements
   - Earn-out structure
   - Payment terms

2. Due Diligence
   - Disclosure requirements
   - Verification process
   - Material contracts
   - Employee matters
   - Regulatory compliance
   - Environmental review

3. Warranties and Indemnities
   - Business warranties
   - Tax warranties
   - Limitation periods
   - Claims procedures
   - Recovery limitations
   - Insurance coordination

4. Closing Mechanics
   - Conditions precedent
   - Closing deliverables
   - Timing requirements
   - Post-closing adjustments
   - Integration planning
   - Regulatory filings

5. Post-Closing Obligations
   - Transition services
   - Non-compete provisions
   - Employee matters
   - Customer relations
   - Tax cooperation
   - Record retention""",

   "Non-Compete Agreement": """Analyze this Non-Compete Agreement with detailed focus on:

1. Restriction Scope and Reasonableness
   - Activity limitations and definitions
   - Geographic bounds and market analysis
   - Time restrictions and industry standards
   - Customer/client restrictions
   - Product/service scope limitations
   - Carve-outs and exceptions

2. Enforcement Mechanisms and Remedies
   - Monitoring rights and procedures
   - Breach definition and materiality
   - Available remedies framework
   - Injunctive relief provisions
   - Damages calculation methodology
   - Attorney fees and costs

3. Consideration and Employment Context
   - Value adequacy analysis
   - Payment structure and timing
   - Employment relationship impact
   - Benefit linkage assessment
   - Tax treatment considerations
   - Continued employment provisions

4. Legal Compliance Analysis
   - Jurisdictional requirements review
   - Industry-specific regulations
   - Public policy considerations
   - Reasonableness assessment
   - Blue pencil provisions
   - Reform clause analysis

5. Integration with Other Agreements
   - Employment agreement coordination
   - Severance agreement interaction
   - Equity plan relationships
   - Assignment provisions
   - Survival terms""",

    "Non-Disclosure Agreement (NDA)": """Conduct a comprehensive analysis of this Non-Disclosure Agreement focusing on:

1. Information Protection Framework
   - Confidential information definition
   - Trade secret designation criteria
   - Exclusions and carve-outs
   - Standard of care requirements
   - Security protocols
   - Data protection compliance

2. Usage Rights and Restrictions
   - Permitted purpose definition
   - Internal distribution controls
   - Third-party disclosure rights
   - Return/destruction obligations
   - Residual knowledge rights
   - Feedback/improvements treatment

3. Term and Duration Analysis
   - Confidentiality period
   - Survival provisions
   - Extension mechanisms
   - Termination rights
   - Post-termination obligations
   - Perpetual obligations

4. Compliance and Monitoring
   - Audit rights framework
   - Record-keeping requirements
   - Breach notification procedures
   - Certification requirements
   - Employee training obligations
   - Subcontractor management

5. Enforcement and Remedies
   - Injunctive relief provisions
   - Damages calculation
   - Breach definition
   - Cure provisions
   - Jurisdictional considerations
   - Alternative dispute resolution""",

    "Partnership Agreement": """Perform an in-depth analysis of this Partnership Agreement addressing:

1. Partnership Structure and Formation
   - Entity classification
   - Capital contributions framework
   - Ownership interests definition
   - Registration requirements
   - Tax treatment election
   - Regulatory compliance

2. Management and Control
   - Decision-making authority
   - Voting rights allocation
   - Partner responsibilities
   - Officers designation
   - Operational control
   - Meeting procedures

3. Financial Framework
   - Profit/loss allocation
   - Distribution methodology
   - Capital account maintenance
   - Tax allocations
   - Additional capital calls
   - Accounting methods

4. Partner Rights and Obligations
   - Fiduciary duties
   - Information rights
   - Participation rights
   - Non-compete obligations
   - Confidentiality requirements
   - Time commitment

5. Transfer and Exit Mechanisms
   - Transfer restrictions
   - Right of first refusal
   - Tag-along/drag-along rights
   - Put/call options
   - Valuation methodology
   - Buy-sell provisions""",

    "Pledge Agreement": """Analyze this Pledge Agreement with comprehensive focus on:

1. Collateral Package Analysis
   - Asset description precision
   - Perfection requirements
   - Control mechanisms
   - Maintenance obligations
   - Valuation methodology
   - Additional collateral provisions

2. Rights and Enforcement
   - Secured party rights
   - Voting rights treatment
   - Dividend/distribution rights
   - Default triggers
   - Enforcement procedures
   - Remedies framework

3. Operational Controls
   - Pledgor covenants
   - Reporting requirements
   - Insurance obligations
   - Corporate action restrictions
   - Information rights
   - Inspection provisions

4. Risk Allocation
   - Representations and warranties
   - Indemnification provisions
   - Expense allocation
   - Tax implications
   - Liability limitations
   - Force majeure treatment

5. Release and Termination
   - Release conditions
   - Partial release mechanics
   - Substitution rights
   - Termination events
   - Documentation requirements
   - Recording obligations""",

    "Real Estate Agreement to Sell": """Conduct a detailed analysis of this Real Estate Agreement to Sell focusing on:

1. Property and Rights Analysis
   - Legal description accuracy
   - Title status review
   - Easement implications
   - Zoning compliance
   - Environmental conditions
   - Property inclusion/exclusion

2. Transaction Structure
   - Purchase price components
   - Payment terms analysis
   - Deposit requirements
   - Escrow arrangements
   - Closing cost allocation
   - Prorations methodology

3. Due Diligence Framework
   - Inspection rights scope
   - Document review process
   - Title examination period
   - Survey requirements
   - Environmental assessment
   - Property condition review

4. Contingencies and Conditions
   - Financing conditions
   - Title contingencies
   - Inspection contingencies
   - Permit/approval requirements
   - Environmental clearance
   - Property condition obligations

5. Closing Process
   - Closing timeline
   - Required deliverables
   - Title insurance requirements
   - Recording procedures
   - Post-closing obligations
   - Default remedies""",

    "Real Estate Purchase Agreement": """Analyze this Real Estate Purchase Agreement with detailed attention to:

1. Transaction Fundamentals
   - Property identification
   - Purchase price structure
   - Payment methodology
   - Deposit handling
   - Closing timeline
   - Extension rights

2. Due Diligence Requirements
   - Physical inspection rights
   - Document review scope
   - Environmental assessment
   - Title examination
   - Survey requirements
   - Zoning compliance

3. Representations and Warranties
   - Title warranties
   - Property condition
   - Environmental status
   - Compliance with laws
   - Litigation disclosure
   - Financial information

4. Risk Allocation
   - Property condition
   - Environmental liability
   - Title defects
   - Casualty/condemnation
   - Regulatory compliance
   - Third-party claims

5. Closing Mechanics
   - Conditions precedent
   - Required deliverables
   - Prorations methodology
   - Transfer tax allocation
   - Recording requirements
   - Post-closing obligations""",

    "Shareholders' Agreement": """Perform a comprehensive analysis of this Shareholders' Agreement addressing:

1. Governance Structure
   - Board composition rules
   - Voting rights framework
   - Reserved matters list
   - Meeting procedures
   - Information rights
   - Management structure

2. Share Transfer Framework
   - Transfer restrictions
   - Right of first refusal
   - Tag-along rights
   - Drag-along rights
   - Put/call options
   - Valuation mechanisms

3. Economic Rights
   - Dividend policy
   - Distribution rights
   - Capital call obligations
   - Share class rights
   - Anti-dilution protection
   - Preemptive rights

4. Exit Mechanisms
   - IPO provisions
   - Strategic sale rights
   - Buy-sell provisions
   - Forced sale rights
   - Redemption rights
   - Valuation procedures

5. Dispute Resolution
   - Deadlock provisions
   - Mediation requirements
   - Arbitration procedures
   - Buyout mechanisms
   - Expert determination
   - Governing law""",

    "Services Agreement": """Analyze this Services Agreement with comprehensive focus on:

1. Service Scope and Standards
   - Service description detail
   - Performance standards
   - Deliverable specifications
   - Quality requirements
   - Timeline commitments
   - Acceptance criteria

2. Commercial Terms
   - Fee structure analysis
   - Payment terms
   - Expense treatment
   - Rate adjustment mechanisms
   - Invoice requirements
   - Late payment consequences

3. Performance Management
   - KPI framework
   - Service levels
   - Reporting requirements
   - Review procedures
   - Remediation process
   - Continuous improvement

4. Risk Allocation
   - Warranties scope
   - Indemnification provisions
   - Limitation of liability
   - Insurance requirements
   - Force majeure terms
   - Termination rights

5. Intellectual Property
   - IP ownership
   - License grants
   - Third-party rights
   - Work product rights
   - Background IP
   - Improvements rights""",

    "Manufacturing Agreement": """Conduct a detailed analysis of this Manufacturing Agreement focusing on:

1. Production Requirements
   - Specifications detail
   - Quality standards
   - Capacity commitments
   - Forecast obligations
   - Order procedures
   - Change management

2. Supply Chain Management
   - Materials sourcing
   - Inventory requirements
   - Storage obligations
   - Logistics arrangements
   - Lead time commitments
   - Contingency planning

3. Quality Control Framework
   - Testing procedures
   - Inspection rights
   - Acceptance criteria
   - Rejection procedures
   - Remediation process
   - Documentation requirements

4. Commercial Structure
   - Pricing mechanisms
   - Cost adjustments
   - Payment terms
   - Volume commitments
   - Tooling rights
   - Exclusivity provisions

5. Risk Management
   - Warranty provisions
   - Product liability
   - Recall procedures
   - Insurance requirements
   - Force majeure
   - Termination rights""",

    "Tolling Agreement": """Analyze this Tolling Agreement with detailed attention to:

1. Processing Framework
   - Service scope
   - Technical specifications
   - Capacity commitments
   - Quality standards
   - By-product handling
   - Process controls

2. Operational Structure
   - Material delivery
   - Processing schedule
   - Storage requirements
   - Testing procedures
   - Maintenance obligations
   - Safety protocols

3. Commercial Terms
   - Fee structure
   - Volume commitments
   - Cost adjustments
   - Payment terms
   - Performance metrics
   - Bonus/penalty provisions

4. Material Management
   - Title retention
   - Loss allocation
   - Yield requirements
   - Scrap handling
   - Storage obligations
   - Transportation responsibilities

5. Risk Allocation
   - Liability framework
   - Insurance requirements
   - Environmental compliance
   - Force majeure
   - Indemnification
   - Termination rights""",

    "Slump Sale Agreement": """Perform a comprehensive analysis of this Slump Sale Agreement addressing:

1. Business Transfer Scope
   - Asset identification
   - Liability assumption
   - Employee transfer
   - Contract assignment
   - Permit/license transfer
   - Excluded items

2. Valuation Framework
   - Purchase price structure
   - Adjustment mechanisms
   - Working capital
   - Debt treatment
   - Tax allocation
   - Payment terms

3. Transition Planning
   - Operational handover
   - Employee integration
   - Systems transfer
   - Customer communication
   - Vendor management
   - Timeline management

4. Regulatory Compliance
   - Required approvals
   - Tax clearances
   - Labor law compliance
   - Environmental requirements
   - Industry regulations
   - Filing obligations

5. Risk Management
   - Representations/warranties
   - Indemnification provisions
   - Insurance requirements
   - Environmental liability
   - Employee claims
   - Third-party consents""",

    "Patent Assignment Agreement": """Analyze this Patent Assignment Agreement with comprehensive focus on:

1. Patent Portfolio Analysis
   - Patent identification
   - Territory coverage
   - Priority rights
   - Family relationships
   - Improvement rights
   - Prosecution status

2. Assignment Scope
   - Rights transferred
   - Reserved rights
   - Prior licenses
   - Future applications
   - Prosecution rights
   - Maintenance obligations

3. Representations/Warranties
   - Title warranty
   - Non-infringement
   - Enforceability
   - Prior agreements
   - Encumbrances
   - Validity status

4. Post-Assignment Obligations
   - Registration requirements
   - Cooperation duties
   - Further assurance
   - Maintenance fees
   - Enforcement support
   - Technology transfer

5. Risk Allocation
   - Indemnification scope
   - Liability limitations
   - Insurance requirements
   - Third-party claims
   - Patent challenges
   - Enforcement costs""",

    "Technology License Agreement": """Conduct a comprehensive analysis of this Technology License Agreement focusing on:

1. License Grant Framework
   - Scope definition
   - Field of use
   - Territory limits
   - Sublicense rights
   - Improvement rights
   - Reserved rights

2. Technical Implementation
   - Technology transfer
   - Integration requirements
   - Documentation standards
   - Support obligations
   - Training requirements
   - Maintenance services

3. IP Protection
   - Ownership provisions
   - Patent prosecution
   - Enforcement rights
   - Infringement defense
   - Trade secret protection
   - Improvements ownership

4. Commercial Terms
   - Fee structure
   - Royalty calculation
   - Payment terms
   - Audit rights
   - Minimum payments
   - Milestone payments

5. Quality Control
   - Performance standards
   - Testing requirements
   - Acceptance criteria
   - Monitoring rights
   - Reporting obligations
   - Compliance verification"""
}

RISK_ANALYSIS_PROMPTS = {
    "Shareholders' Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine board control mechanisms, voting rights, and deadlock provisions
   - Analyze share transfer restrictions, tag-along/drag-along rights enforceability
   - Assess regulatory compliance across jurisdictions

B. Financial & Commercial Risks:
   - Evaluate capital structure, dilution protection, and funding obligations
   - Analyze valuation methodologies and exit mechanisms
   - Examine dividend policies and economic rights distribution

C. Business & Operational Risks:
   - Assess decision-making processes and management control
   - Examine IP rights, confidentiality provisions, and technology transfer
   - Analyze dispute resolution mechanisms and enforceability""",

    "Asset Purchase Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine asset definition completeness and transfer mechanisms
   - Analyze regulatory approvals and compliance requirements
   - Assess employment and benefit plan transfer obligations

B. Financial & Commercial Risks:
   - Evaluate purchase price adjustments and payment terms
   - Analyze working capital mechanisms and earn-out structures
   - Examine tax implications and allocation of liabilities

C. Business & Operational Risks:
   - Assess integration planning and operational continuity
   - Examine customer/supplier contract assignments
   - Analyze transition services and knowledge transfer""",

    "Collaboration Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine IP ownership and licensing provisions
   - Analyze confidentiality and data protection compliance
   - Assess competition law implications

B. Financial & Commercial Risks:
   - Evaluate cost-sharing and revenue allocation mechanisms
   - Analyze resource commitment obligations
   - Examine termination and wind-down costs

C. Business & Operational Risks:
   - Assess project management and control mechanisms
   - Examine quality control and performance standards
   - Analyze dispute resolution and escalation procedures""",

    "Confidentiality Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine scope of confidential information definition
   - Analyze permitted disclosure exceptions
   - Assess data protection compliance requirements

B. Financial & Commercial Risks:
   - Evaluate damages provisions and enforcement costs
   - Analyze audit and monitoring obligations
   - Examine breach notification requirements

C. Business & Operational Risks:
   - Assess information security controls
   - Examine return/destruction procedures
   - Analyze employee/contractor compliance measures""",

    "Escrow Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine escrow agent duties and liability limitations
   - Analyze release conditions and trigger events
   - Assess regulatory compliance requirements

B. Financial & Commercial Risks:
   - Evaluate escrow fund management and interest
   - Analyze fees and cost allocation
   - Examine default scenarios and remedies

C. Business & Operational Risks:
   - Assess verification and reporting procedures
   - Examine dispute resolution mechanisms
   - Analyze security and access controls""",

    "Franchise Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine franchise disclosure compliance
   - Analyze trademark and IP protection
   - Assess territory rights and exclusivity

B. Financial & Commercial Risks:
   - Evaluate royalty calculations and audit rights
   - Analyze advertising fund management
   - Examine supply chain requirements

C. Business & Operational Risks:
   - Assess quality control standards
   - Examine training and support obligations
   - Analyze termination and post-term obligations""",

    "Investment Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine securities law compliance
   - Analyze investor rights and protections
   - Assess corporate governance requirements

B. Financial & Commercial Risks:
   - Evaluate valuation and anti-dilution provisions
   - Analyze liquidation preferences and exit rights
   - Examine future financing obligations

C. Business & Operational Risks:
   - Assess board control and veto rights
   - Examine information rights and reporting
   - Analyze strategic alignment and conflicts""",

    "Joint Venture Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine structure and governance compliance
   - Analyze competition law implications
   - Assess cross-border regulations

B. Financial & Commercial Risks:
   - Evaluate capital contributions and funding
   - Analyze profit sharing and distribution
   - Examine exit and termination costs

C. Business & Operational Risks:
   - Assess management control and deadlock
   - Examine IP rights and technology transfer
   - Analyze operational integration challenges""",

    "Lease Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine property use and zoning compliance
   - Analyze maintenance obligations
   - Assess environmental regulations

B. Financial & Commercial Risks:
   - Evaluate rent escalation and operating costs
   - Analyze security deposit provisions
   - Examine insurance requirements

C. Business & Operational Risks:
   - Assess property access and alterations
   - Examine sublease and assignment rights
   - Analyze default remedies and termination""",

    "Loan Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine security interest perfection
   - Analyze regulatory compliance requirements
   - Assess cross-default provisions

B. Financial & Commercial Risks:
   - Evaluate interest rates and payment terms
   - Analyze financial covenants and testing
   - Examine collateral valuation methods

C. Business & Operational Risks:
   - Assess reporting requirements
   - Examine default triggers and remedies
   - Analyze operational restrictions""",

    "Technology License Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine IP ownership and protection
   - Analyze export control compliance
   - Assess data protection requirements

B. Financial & Commercial Risks:
   - Evaluate royalty calculations and audits
   - Analyze maintenance and support costs
   - Examine termination implications

C. Business & Operational Risks:
   - Assess implementation and integration
   - Examine performance requirements
   - Analyze security and updates""",

    "Manufacturing Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine product liability exposure
   - Analyze quality control standards
   - Assess regulatory compliance

B. Financial & Commercial Risks:
   - Evaluate pricing and cost adjustments
   - Analyze minimum purchase requirements
   - Examine tooling and equipment costs

C. Business & Operational Risks:
   - Assess supply chain dependencies
   - Examine capacity commitments
   - Analyze quality control procedures""",

    "Services Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine service level requirements
   - Analyze data protection compliance
   - Assess regulatory requirements

B. Financial & Commercial Risks:
   - Evaluate pricing and payment terms
   - Analyze liability limitations
   - Examine termination costs

C. Business & Operational Risks:
   - Assess performance monitoring
   - Examine staff and resource commitments
   - Analyze business continuity plans""",

   "Copyright Assignment Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine scope of rights being transferred
   - Analyze moral rights and jurisdictional variations
   - Assess registration and recordation requirements

B. Financial & Commercial Risks:
   - Evaluate payment terms and royalty obligations
   - Analyze future exploitation rights
   - Examine warranties and indemnification

C. Business & Operational Risks:
   - Assess delivery and documentation requirements
   - Examine quality control provisions
   - Analyze future cooperation obligations""",

    "Indemnification Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine scope of indemnified claims
   - Analyze defense and settlement rights
   - Assess insurance coordination

B. Financial & Commercial Risks:
   - Evaluate caps and baskets
   - Analyze survival periods
   - Examine recovery procedures

C. Business & Operational Risks:
   - Assess notice requirements
   - Examine claim management procedures
   - Analyze dispute resolution mechanisms""",

    "Loan Purchase Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine loan documentation completeness
   - Analyze assignability restrictions
   - Assess regulatory compliance transfer

B. Financial & Commercial Risks:
   - Evaluate pricing and payment mechanisms
   - Analyze representations about loan quality
   - Examine servicing transfer obligations

C. Business & Operational Risks:
   - Assess borrower notification requirements
   - Examine document delivery procedures
   - Analyze servicing transition risks""",

    "Share Purchase Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine title and authority to sell
   - Analyze regulatory approvals
   - Assess warranties and representations

B. Financial & Commercial Risks:
   - Evaluate price adjustment mechanisms
   - Analyze earn-out structures
   - Examine working capital requirements

C. Business & Operational Risks:
   - Assess business continuity risks
   - Examine employee retention issues
   - Analyze integration planning""",

    "Non-Compete Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine scope and duration enforceability
   - Analyze jurisdictional variations
   - Assess consideration adequacy

B. Financial & Commercial Risks:
   - Evaluate compensation arrangements
   - Analyze monitoring costs
   - Examine damage calculations

C. Business & Operational Risks:
   - Assess market impact
   - Examine compliance monitoring
   - Analyze enforcement mechanisms""",

    "Non-Disclosure Agreement (NDA)": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine scope of protected information
   - Analyze permitted disclosure exceptions
   - Assess data protection compliance

B. Financial & Commercial Risks:
   - Evaluate breach remedies
   - Analyze monitoring costs
   - Examine enforcement mechanisms

C. Business & Operational Risks:
   - Assess information handling procedures
   - Examine return/destruction requirements
   - Analyze compliance tracking""",

    "Partnership Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine partner liability provisions
   - Analyze governance structure
   - Assess regulatory compliance

B. Financial & Commercial Risks:
   - Evaluate capital contribution obligations
   - Analyze profit/loss allocation
   - Examine distribution policies

C. Business & Operational Risks:
   - Assess management responsibilities
   - Examine decision-making processes
   - Analyze exit mechanisms""",

    "Pledge Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine perfection requirements
   - Analyze enforcement procedures
   - Assess regulatory restrictions

B. Financial & Commercial Risks:
   - Evaluate collateral valuation
   - Analyze maintenance requirements
   - Examine default triggers

C. Business & Operational Risks:
   - Assess monitoring procedures
   - Examine voting rights
   - Analyze disposition procedures""",

    "Real Estate Agreement to Sell": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine title issues
   - Analyze environmental compliance
   - Assess zoning requirements

B. Financial & Commercial Risks:
   - Evaluate price adjustment mechanisms
   - Analyze financing contingencies
   - Examine closing cost allocation

C. Business & Operational Risks:
   - Assess due diligence procedures
   - Examine property condition issues
   - Analyze closing requirements""",

    "Real Estate Purchase Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine title and survey issues
   - Analyze environmental compliance
   - Assess land use restrictions

B. Financial & Commercial Risks:
   - Evaluate purchase price adjustments
   - Analyze deposit provisions
   - Examine closing cost allocation

C. Business & Operational Risks:
   - Assess property condition
   - Examine tenant issues
   - Analyze service contracts""",

    "Slump Sale Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine transfer requirements
   - Analyze employee obligations
   - Assess tax implications

B. Financial & Commercial Risks:
   - Evaluate valuation mechanisms
   - Analyze working capital adjustments
   - Examine liability transfer

C. Business & Operational Risks:
   - Assess business continuity
   - Examine contract assignments
   - Analyze integration planning""",

    "Patent Assignment Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine patent validity and scope
   - Analyze recordation requirements
   - Assess international rights

B. Financial & Commercial Risks:
   - Evaluate payment terms
   - Analyze future royalties
   - Examine improvement rights

C. Business & Operational Risks:
   - Assess technology transfer
   - Examine documentation requirements
   - Analyze ongoing obligations""",

    "Tolling Agreement": """Analyze risk exposure in three key areas:

A. Legal & Regulatory Risks:
   - Examine product liability
   - Analyze quality standards
   - Assess regulatory compliance

B. Financial & Commercial Risks:
   - Evaluate processing fees
   - Analyze raw material costs
   - Examine yield requirements

C. Business & Operational Risks:
   - Assess production scheduling
   - Examine quality control
   - Analyze capacity commitments"""
}

# Analysis prompts
CONFLICT_ANALYSIS_PROMPT = """
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

GENERAL_LONG_SUMMARY_PROMPT = """
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

DRAFT_PROMPT = """
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

ASK_PROMPT = '''
You are a legal AI assistant with expertise in contract analysis and legal document review. Your role is to provide clear, authoritative answers while maintaining accuracy through precise citations.

CITATION FORMAT:
1. When referencing specific clauses or sections, always include the actual text content within [[double brackets]], not the clause numbers.
   Example: "The agreement states [[The party shall be liable for all damages]]"

2. Add the filename after the citation using {{filename}}:
   Example: "As specified in [[The party shall be liable]]{{Agreement.pdf}}"

3. For multiple related references, use them separately:
   - Single reference: [[The Seller shall deliver...]]{{Agreement.pdf}}
   - Multiple references: [[The Buyer agrees to pay...]]{{Agreement.pdf}} and [[All disputes shall be...]]{{Agreement.pdf}}

CITATION GUIDELINES:
- Always use the exact text as it appears in the document, do not change it or paraphrase it.
- Never include formatting characters (**, `, etc.) in citations
- Keep citations concise (30-40 characters)
- Never combine multiple references within a single bracket
- Do not use ellipsis (...), just use the first part of the text
- Always include the filename after each citation

EXAMPLE RESPONSE:
The contract includes **important provisions** about liability [[The party shall be liable]]{{Agreement.pdf}} and termination [[Agreement may be terminated]]{{Agreement.pdf}}.

RESPONSE GUIDELINES:
1. Begin with a direct answer
2. Support claims with exact citations
3. Structure complex responses clearly
4. Maintain professional tone

Now analyze the provided context and address the query.
'''
