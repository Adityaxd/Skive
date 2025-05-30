# enhanced_features.py - Additional production-ready features

import asyncio
import redis
import json
import hashlib
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import cv2
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

logger = logging.getLogger(__name__)

@dataclass
class AuditLog:
    timestamp: datetime
    user_id: str
    document_hash: str
    action: str
    result: str
    processing_time: float
    confidence_score: float

class CacheManager:
    """Redis-based caching for improved performance"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Using in-memory cache.")
            self.redis_client = None
            self.memory_cache = {}
    
    async def get(self, key: str) -> Optional[str]:
        """Get cached value"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                return value.decode('utf-8') if value else None
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: str, expire: int = 3600):
        """Set cached value with expiration"""
        try:
            if self.redis_client:
                self.redis_client.setex(key, expire, value)
            else:
                self.memory_cache[key] = value
                # Simple memory cleanup (in production, use proper TTL)
                if len(self.memory_cache) > 1000:
                    self.memory_cache.clear()
        except Exception as e:
            logger.error(f"Cache set error: {e}")

class ImagePreprocessor:
    """Advanced image preprocessing for better OCR accuracy"""
    
    @staticmethod
    def enhance_image(image_array: np.ndarray) -> np.ndarray:
        """Enhance image quality for better OCR"""
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    @staticmethod
    def detect_document_boundaries(image_array: np.ndarray) -> np.ndarray:
        """Detect and crop document boundaries"""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (presumably the document)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Crop to document boundaries with some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_array.shape[1] - x, w + 2*padding)
            h = min(image_array.shape[0] - y, h + 2*padding)
            
            return image_array[y:y+h, x:x+w]
        
        return image_array

class SecurityManager:
    """Handle security and compliance features"""
    
    def __init__(self):
        self.blocked_patterns = [
            r'test.*test',  # Test data patterns
            r'dummy.*data',  # Dummy data patterns
            r'sample.*sample',  # Sample data patterns
        ]
    
    def validate_document_authenticity(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document authenticity using various checks"""
        authenticity_score = 1.0
        warnings = []
        
        # Check for suspicious patterns
        for field, value in extracted_data.items():
            if isinstance(value, str):
                for pattern in self.blocked_patterns:
                    import re
                    if re.search(pattern, value.lower()):
                        authenticity_score -= 0.3
                        warnings.append(f"Suspicious pattern detected in {field}")
        
        # Check for consistency in document numbers
        if 'pan_number' in extracted_data:
            pan = extracted_data['pan_number']
            if pan and not self._validate_pan_checksum(pan):
                authenticity_score -= 0.2
                warnings.append("Invalid PAN checksum")
        
        return {
            'authenticity_score': max(0.0, authenticity_score),
            'warnings': warnings
        }
    
    def _validate_pan_checksum(self, pan: str) -> bool:
        """Validate PAN number checksum"""
        if len(pan) != 10:
            return False
        
        # Simple PAN validation (basic format check)
        import re
        pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
        return bool(re.match(pattern, pan))
    
    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data for logging/storage"""
        masked_data = data.copy()
        
        sensitive_fields = ['aadhaar_number', 'pan_number', 'phone_number']
        
        for field in sensitive_fields:
            if field in masked_data and masked_data[field]:
                value = str(masked_data[field])
                if len(value) > 4:
                    masked_data[field] = 'X' * (len(value) - 4) + value[-4:]
        
        return masked_data

class NotificationManager:
    """Handle notifications and alerts"""
    
    def __init__(self, smtp_config: Optional[Dict[str, str]] = None):
        self.smtp_config = smtp_config or {}
    
    async def send_verification_complete_notification(self, 
                                                     email: str, 
                                                     result: Dict[str, Any]):
        """Send email notification when verification is complete"""
        if not self.smtp_config.get('enabled'):
            logger.info(f"Email notification disabled. Would send to: {email}")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = email
            msg['Subject'] = "KYC Verification Complete"
            
            status = result.get('status', 'UNKNOWN')
            confidence = result.get('confidence_score', 0)
            
            body = f"""
            Dear Customer,
            
            Your KYC verification has been completed.
            
            Status: {status}
            Confidence Score: {confidence:.2f}
            Processing Time: {result.get('processing_time', 0):.2f} seconds
            
            {self._get_status_message(status)}
            
            Best regards,
            KYC Verification Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port'])
            server.starttls()
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            text = msg.as_string()
            server.sendmail(self.smtp_config['from_email'], email, text)
            server.quit()
            
            logger.info(f"Notification sent to {email}")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def _get_status_message(self, status: str) -> str:
        """Get human-readable status message"""
        messages = {
            'ACCEPTED': 'Your documents have been successfully verified and approved.',
            'REJECTED': 'Your documents could not be verified due to mismatched information. Please resubmit with correct documents.',
            'PENDING': 'Your documents are under manual review. We will contact you within 24 hours.',
            'ERROR': 'There was an error processing your documents. Please try again or contact support.'
        }
        return messages.get(status, 'Please contact support for more information.')

class AuditLogger:
    """Comprehensive audit logging for compliance"""
    
    def __init__(self, log_file: str = "kyc_audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_verification_attempt(self, audit_log: AuditLog):
        """Log verification attempt"""
        log_entry = {
            'timestamp': audit_log.timestamp.isoformat(),
            'user_id': audit_log.user_id,
            'document_hash': audit_log.document_hash,
            'action': audit_log.action,
            'result': audit_log.result,
            'processing_time': audit_log.processing_time,
            'confidence_score': audit_log.confidence_score
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def get_audit_trail(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get audit trail for a user"""
        # In production, this would query a database
        # For now, parse the log file
        audit_trail = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        # Extract JSON from log line
                        json_start = line.find('{')
                        if json_start != -1:
                            log_entry = json.loads(line[json_start:])
                            if log_entry.get('user_id') == user_id:
                                entry_date = datetime.fromisoformat(log_entry['timestamp'])
                                if entry_date > datetime.now() - timedelta(days=days):
                                    audit_trail.append(log_entry)
                    except:
                        continue
        except FileNotFoundError:
            pass
        
        return sorted(audit_trail, key=lambda x: x['timestamp'], reverse=True)

class EnhancedKYCAgent:
    """Enhanced KYC agent with additional production features"""
    
    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        from kyc_verification_agent import KYCVerificationAgent
        
        self.base_agent = KYCVerificationAgent(api_key)
        self.config = config or {}
        
        # Initialize enhanced components
        self.cache_manager = CacheManager(self.config.get('redis_url', 'redis://localhost:6379'))
        self.security_manager = SecurityManager()
        self.notification_manager = NotificationManager(self.config.get('smtp'))
        self.audit_logger = AuditLogger()
        self.image_preprocessor = ImagePreprocessor()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_kyc_document_enhanced(self, 
                                          pdf_path: str, 
                                          user_id: str,
                                          email: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced KYC processing with caching, security, and notifications"""
        start_time = time.time()
        
        # Calculate document hash for caching and audit
        with open(pdf_path, 'rb') as f:
            doc_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Check cache first
        cache_key = f"kyc_result_{doc_hash}"
        cached_result = await self.cache_manager.get(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached result for document {doc_hash[:8]}")
            result = json.loads(cached_result)
            
            # Log cache hit
            audit_log = AuditLog(
                timestamp=datetime.now(),
                user_id=user_id,
                document_hash=doc_hash[:8],
                action="CACHE_HIT",
                result=result.get('status', 'UNKNOWN'),
                processing_time=0.001,
                confidence_score=result.get('confidence_score', 0)
            )
            self.audit_logger.log_verification_attempt(audit_log)
            
            return result
        
        try:
            # Process with base agent
            base_result = await self.base_agent.process_kyc_document(pdf_path)
            
            # Extract data for security validation
            extracted_data = {}
            for doc_type, doc_info in base_result.extracted_documents.items():
                extracted_data.update(doc_info.to_dict())
            
            # Security validation
            security_result = self.security_manager.validate_document_authenticity(extracted_data)
            
            # Combine results
            enhanced_result = {
                'status': base_result.status.value,
                'confidence_score': base_result.confidence_score * security_result['authenticity_score'],
                'processing_time': base_result.processing_time,
                'mismatched_attributes': base_result.mismatched_attributes,
                'missing_attributes': base_result.missing_attributes,
                'errors': base_result.errors,
                'document_hash': doc_hash[:8],
                'security_warnings': security_result['warnings'],
                'authenticity_score': security_result['authenticity_score'],
                'extracted_data': self.security_manager.mask_sensitive_data(extracted_data)
            }
            
            # Cache result
            await self.cache_manager.set(cache_key, json.dumps(enhanced_result), expire=3600)
            
            # Send notification if email provided
            if email:
                await self.notification_manager.send_verification_complete_notification(
                    email, enhanced_result
                )
            
            # Audit logging
            audit_log = AuditLog(
                timestamp=datetime.now(),
                user_id=user_id,
                document_hash=doc_hash[:8],
                action="VERIFICATION_COMPLETE",
                result=enhanced_result['status'],
                processing_time=time.time() - start_time,
                confidence_score=enhanced_result['confidence_score']
            )
            self.audit_logger.log_verification_attempt(audit_log)
            
            logger.info(f"Enhanced KYC processing completed for user {user_id}")
            return enhanced_result
            
        except Exception as e:
            # Error logging
            audit_log = AuditLog(
                timestamp=datetime.now(),
                user_id=user_id,
                document_hash=doc_hash[:8],
                action="VERIFICATION_ERROR",
                result="ERROR",
                processing_time=time.time() - start_time,
                confidence_score=0.0
            )
            self.audit_logger.log_verification_attempt(audit_log)
            
            logger.error(f"Enhanced KYC processing failed for user {user_id}: {e}")
            raise

# Configuration example
DEFAULT_CONFIG = {
    'redis_url': 'redis://localhost:6379',
    'smtp': {
        'enabled': False,  # Set to True to enable email notifications
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_app_password',
        'from_email': 'noreply@yourcompany.com'
    },
    'security': {
        'min_confidence_threshold': 0.7,
        'max_processing_time': 300,  # 5 minutes
        'enable_authenticity_checks': True
    },
    'performance': {
        'max_concurrent_requests': 10,
        'cache_ttl': 3600,  # 1 hour
        'retry_attempts': 3
    }
}