import fitz  # PyMuPDF
import base64
import io
import json
import logging
import asyncio
import aiohttp
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import re
from pathlib import Path
import os
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kyc_verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    AADHAAR = "aadhaar"
    PAN = "pan"
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    VOTER_ID = "voter_id"
    FORM = "form"

class VerificationStatus(Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"
    ERROR = "ERROR"

@dataclass
class ExtractedInfo:
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    document_number: Optional[str] = None
    document_type: Optional[str] = None
    father_name: Optional[str] = None
    mother_name: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class VerificationResult:
    status: VerificationStatus
    confidence_score: float
    mismatched_attributes: List[str]
    missing_attributes: List[str]
    extracted_documents: Dict[str, ExtractedInfo]
    processing_time: float
    errors: List[str]
    document_hash: str

class DocumentProcessor:
    """Handles PDF to image conversion with optimization"""
    
    @staticmethod
    def pdf_to_base64_images(pdf_path: str, quality: int = 85, max_size: Tuple[int, int] = (1024, 1024)) -> List[str]:
        """Convert PDF pages to base64 encoded images"""
        try:
            doc = fitz.open(pdf_path)
            base64_images = []
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                # Higher DPI for better OCR accuracy
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # Convert to PIL Image
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Optimize image size
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                image_data = io.BytesIO()
                image.save(image_data, format='PNG', optimize=True, quality=quality)
                image_data.seek(0)
                base64_encoded = base64.b64encode(image_data.getvalue()).decode('utf-8')
                base64_images.append(base64_encoded)
            
            doc.close()
            logger.info(f"Successfully processed {len(base64_images)} pages from {pdf_path}")
            return base64_images
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    @staticmethod
    def calculate_document_hash(pdf_path: str) -> str:
        """Calculate hash for document integrity"""
        with open(pdf_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

class LLMClient:
    """Enhanced LLM client with retry logic and error handling"""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-haiku-20240307"):
        # Note: In production, use environment variables for API keys
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = 3
        
    async def extract_information(self, images: List[str], prompt: str) -> str:
        """Extract information from images using LLM"""
        # This is a placeholder for the actual API call
        # In production, you would implement the actual Anthropic API call here
        
        # Simulated extraction based on the document type
        if "aadhaar" in prompt.lower():
            return self._simulate_aadhaar_extraction()
        elif "pan" in prompt.lower():
            return self._simulate_pan_extraction()
        else:
            return self._simulate_form_extraction()
    
    def _simulate_aadhaar_extraction(self) -> str:
        return json.dumps({
            "name": "MANOJ KUMAR",
            "date_of_birth": "01-01-1988",
            "gender": "Male",
            "aadhaar_number": "XXXXXXXX1701",
            "address": "Bundel Singh,House No 02,Fortune soumya Heritage Nehar k,Huzur Bhopal,Shiv mandir,Huzur Bhopal,Madhya Pradesh,462039"
        })
    
    def _simulate_pan_extraction(self) -> str:
        return json.dumps({
            "name": "MANOJ KUMAR",
            "pan_number": "BGQPK4512E",
            "date_of_birth": "01/01/1988",
            "gender": "MALE"
        })
    
    def _simulate_form_extraction(self) -> str:
        return json.dumps({
            "name": "MANOJ KUMAR",
            "mother_name": "Bundel Singh",
            "father_name": "Bundel Singh",
            "date_of_birth": "01/01/1988",
            "gender": "Male",
            "address": "Bundel Singh,House No 02,Fortune soumya Heritage Nehar k,Huzur Bhopal,Shiv mandir,Huzur,Trilanga,Bhopal,Madhya Pradesh",
            "email": "MANOJKUMAR241351@GMAIL.COM",
            "phone": "9244101683",
            "pan_number": "BGQPK4512E"
        })

class DataValidator:
    """Validates and cross-references extracted data"""
    
    @staticmethod
    def normalize_date(date_str: str) -> str:
        """Normalize date formats"""
        if not date_str:
            return ""
        
        # Handle different date formats
        patterns = [
            r'(\d{2})[/-](\d{2})[/-](\d{4})',  # DD/MM/YYYY or DD-MM-YYYY
            r'(\d{2})[/-](\d{2})[/-](\d{2})',   # DD/MM/YY or DD-MM-YY
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                day, month, year = match.groups()
                if len(year) == 2:
                    year = "19" + year if int(year) > 50 else "20" + year
                return f"{day}/{month}/{year}"
        
        return date_str
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize name for comparison"""
        if not name:
            return ""
        return re.sub(r'\s+', ' ', name.upper().strip())
    
    @staticmethod
    def validate_pan(pan: str) -> bool:
        """Validate PAN format"""
        if not pan:
            return False
        pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
        return bool(re.match(pattern, pan))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number"""
        if not phone:
            return False
        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)
        return len(digits) >= 10
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        if not email:
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

class KYCVerificationAgent:
    """Main KYC Verification Agent"""
    
    def __init__(self, api_key: str):
        self.llm_client = LLMClient(api_key)
        self.document_processor = DocumentProcessor()
        self.validator = DataValidator()
        
    async def process_kyc_document(self, pdf_path: str) -> VerificationResult:
        """Process a single KYC document"""
        start_time = datetime.now()
        
        try:
            # Calculate document hash
            doc_hash = self.document_processor.calculate_document_hash(pdf_path)
            
            # Convert PDF to images
            images = self.document_processor.pdf_to_base64_images(pdf_path)
            
            # Extract information from different document types
            extracted_docs = await self._extract_all_document_types(images)
            
            # Validate and cross-reference
            validation_result = self._validate_extracted_data(extracted_docs)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VerificationResult(
                status=validation_result['status'],
                confidence_score=validation_result['confidence_score'],
                mismatched_attributes=validation_result['mismatched_attributes'],
                missing_attributes=validation_result['missing_attributes'],
                extracted_documents=extracted_docs,
                processing_time=processing_time,
                errors=validation_result.get('errors', []),
                document_hash=doc_hash
            )
            
        except Exception as e:
            logger.error(f"Error processing KYC document {pdf_path}: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VerificationResult(
                status=VerificationStatus.ERROR,
                confidence_score=0.0,
                mismatched_attributes=[],
                missing_attributes=[],
                extracted_documents={},
                processing_time=processing_time,
                errors=[str(e)],
                document_hash=""
            )
    
    async def _extract_all_document_types(self, images: List[str]) -> Dict[str, ExtractedInfo]:
        """Extract information for all document types"""
        extraction_tasks = [
            self._extract_form_data(images),
            self._extract_aadhaar_data(images),
            self._extract_pan_data(images)
        ]
        
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        extracted_docs = {}
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                doc_type = ['form', 'aadhaar', 'pan'][i]
                extracted_docs[doc_type] = result
        
        return extracted_docs
    
    async def _extract_form_data(self, images: List[str]) -> ExtractedInfo:
        """Extract KYC form data"""
        prompt = """
        Extract the following information from the KYC application form:
        - Full name
        - Date of birth
        - Gender
        - Father's/Spouse's name
        - Mother's name
        - Address
        - Phone number
        - Email
        - PAN number
        
        Return the data in JSON format with exact field names.
        """
        
        response = await self.llm_client.extract_information(images, prompt)
        data = json.loads(response)
        
        return ExtractedInfo(
            name=data.get('name'),
            date_of_birth=data.get('date_of_birth'),
            gender=data.get('gender'),
            address=data.get('address'),
            phone_number=data.get('phone'),
            email=data.get('email'),
            document_number=data.get('pan_number'),
            document_type="KYC_FORM",
            father_name=data.get('father_name'),
            mother_name=data.get('mother_name')
        )
    
    async def _extract_aadhaar_data(self, images: List[str]) -> ExtractedInfo:
        """Extract Aadhaar data"""
        prompt = """
        Extract the following information from the Aadhaar document:
        - Name
        - Date of birth
        - Gender
        - Address
        - Aadhaar number (last 4 digits)
        
        Return the data in JSON format.
        """
        
        response = await self.llm_client.extract_information(images, prompt)
        data = json.loads(response)
        
        return ExtractedInfo(
            name=data.get('name'),
            date_of_birth=data.get('date_of_birth'),
            gender=data.get('gender'),
            address=data.get('address'),
            document_number=data.get('aadhaar_number'),
            document_type="AADHAAR"
        )
    
    async def _extract_pan_data(self, images: List[str]) -> ExtractedInfo:
        """Extract PAN card data"""
        prompt = """
        Extract the following information from the PAN card:
        - Name
        - PAN number
        - Date of birth
        - Gender
        
        Return the data in JSON format.
        """
        
        response = await self.llm_client.extract_information(images, prompt)
        data = json.loads(response)
        
        return ExtractedInfo(
            name=data.get('name'),
            date_of_birth=data.get('date_of_birth'),
            gender=data.get('gender'),
            document_number=data.get('pan_number'),
            document_type="PAN"
        )
    
    def _validate_extracted_data(self, extracted_docs: Dict[str, ExtractedInfo]) -> Dict[str, Any]:
        """Validate and cross-reference extracted data"""
        if not extracted_docs:
            return {
                'status': VerificationStatus.ERROR,
                'confidence_score': 0.0,
                'mismatched_attributes': [],
                'missing_attributes': [],
                'errors': ['No data extracted']
            }
        
        mismatched_attributes = []
        missing_attributes = []
        confidence_scores = []
        
        # Get reference document (usually form)
        ref_doc = extracted_docs.get('form') or list(extracted_docs.values())[0]
        
        # Cross-validate common fields
        common_fields = ['name', 'date_of_birth', 'gender']
        
        for field in common_fields:
            ref_value = getattr(ref_doc, field, None)
            if not ref_value:
                missing_attributes.append(field)
                continue
                
            # Normalize reference value
            if field == 'name':
                ref_value = self.validator.normalize_name(ref_value)
            elif field == 'date_of_birth':
                ref_value = self.validator.normalize_date(ref_value)
            
            field_confidence = []
            
            for doc_type, doc_info in extracted_docs.items():
                if doc_type == 'form':
                    continue
                    
                doc_value = getattr(doc_info, field, None)
                if not doc_value:
                    continue
                
                # Normalize document value
                if field == 'name':
                    doc_value = self.validator.normalize_name(doc_value)
                elif field == 'date_of_birth':
                    doc_value = self.validator.normalize_date(doc_value)
                
                # Compare values
                if ref_value == doc_value:
                    field_confidence.append(1.0)
                else:
                    field_confidence.append(0.0)
                    mismatched_attributes.append(f"{field} mismatch between form and {doc_type}")
            
            if field_confidence:
                confidence_scores.append(sum(field_confidence) / len(field_confidence))
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Determine status
        if mismatched_attributes:
            status = VerificationStatus.REJECTED
        elif missing_attributes and overall_confidence < 0.7:
            status = VerificationStatus.PENDING
        else:
            status = VerificationStatus.ACCEPTED
        
        return {
            'status': status,
            'confidence_score': overall_confidence,
            'mismatched_attributes': mismatched_attributes,
            'missing_attributes': missing_attributes,
            'errors': []
        }
    
    async def process_batch_documents(self, pdf_paths: List[str]) -> List[VerificationResult]:
        """Process multiple KYC documents concurrently"""
        tasks = [self.process_kyc_document(pdf_path) for pdf_path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing document {pdf_paths[i]}: {str(result)}")
                processed_results.append(VerificationResult(
                    status=VerificationStatus.ERROR,
                    confidence_score=0.0,
                    mismatched_attributes=[],
                    missing_attributes=[],
                    extracted_documents={},
                    processing_time=0.0,
                    errors=[str(result)],
                    document_hash=""
                ))
            else:
                processed_results.append(result)
        
        return processed_results

class KYCReportGenerator:
    """Generate detailed verification reports"""
    
    @staticmethod
    def generate_report(results: List[VerificationResult]) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        total_documents = len(results)
        accepted = sum(1 for r in results if r.status == VerificationStatus.ACCEPTED)
        rejected = sum(1 for r in results if r.status == VerificationStatus.REJECTED)
        pending = sum(1 for r in results if r.status == VerificationStatus.PENDING)
        errors = sum(1 for r in results if r.status == VerificationStatus.ERROR)
        
        avg_processing_time = sum(r.processing_time for r in results) / total_documents if total_documents > 0 else 0
        avg_confidence = sum(r.confidence_score for r in results) / total_documents if total_documents > 0 else 0
        
        return {
            'summary': {
                'total_documents': total_documents,
                'accepted': accepted,
                'rejected': rejected,
                'pending': pending,
                'errors': errors,
                'success_rate': (accepted / total_documents * 100) if total_documents > 0 else 0,
                'average_processing_time': avg_processing_time,
                'average_confidence_score': avg_confidence
            },
            'detailed_results': [
                {
                    'status': result.status.value,
                    'confidence_score': result.confidence_score,
                    'mismatched_attributes': result.mismatched_attributes,
                    'missing_attributes': result.missing_attributes,
                    'processing_time': result.processing_time,
                    'errors': result.errors,
                    'document_hash': result.document_hash
                }
                for result in results
            ]
        }

# Example usage and testing
async def main():
    """Main function to demonstrate the KYC verification system"""
    
    # Initialize the KYC agent
    # Note: Replace with your actual API key
    api_key = "your-anthropic-api-key-here"
    kyc_agent = KYCVerificationAgent(api_key)
    
    # Example PDF paths
    pdf_paths = [
        "sample1.pdf",
        # Add more PDF paths as needed
    ]
    
    try:
        logger.info("Starting KYC verification process...")
        
        # Process documents
        results = await kyc_agent.process_batch_documents(pdf_paths)
        
        # Generate report
        report_generator = KYCReportGenerator()
        report = report_generator.generate_report(results)
        
        # Print results
        print("\n" + "="*50)
        print("KYC VERIFICATION REPORT")
        print("="*50)
        print(f"Total Documents: {report['summary']['total_documents']}")
        print(f"Accepted: {report['summary']['accepted']}")
        print(f"Rejected: {report['summary']['rejected']}")
        print(f"Pending: {report['summary']['pending']}")
        print(f"Errors: {report['summary']['errors']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Average Processing Time: {report['summary']['average_processing_time']:.2f}s")
        print(f"Average Confidence Score: {report['summary']['average_confidence_score']:.2f}")
        
        # Save detailed report
        with open('kyc_verification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("KYC verification completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())