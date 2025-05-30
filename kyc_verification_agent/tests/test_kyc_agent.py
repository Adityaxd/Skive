# tests/test_kyc_agent.py - Comprehensive test suite

import pytest
import asyncio
import tempfile
import json
import os
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Import modules to test
from kyc_verification_agent import (
    KYCVerificationAgent, 
    DocumentProcessor, 
    DataValidator, 
    VerificationStatus,
    ExtractedInfo
)
from enhanced_features import (
    EnhancedKYCAgent,
    SecurityManager,
    CacheManager,
    NotificationManager
)

class TestDocumentProcessor:
    """Test document processing functionality"""
    
    def test_pdf_to_base64_conversion(self):
        """Test PDF to base64 image conversion"""
        # Create a mock PDF file for testing
        processor = DocumentProcessor()
        
        # This would require a real PDF file for integration testing
        # For unit testing, we'll mock the behavior
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_doc.page_count = 2
            mock_page = Mock()
            mock_pix = Mock()
            mock_pix.width = 800
            mock_pix.height = 600
            mock_pix.samples = b'fake_image_data'
            mock_page.get_pixmap.return_value = mock_pix
            mock_doc.load_page.return_value = mock_page
            mock_fitz.return_value = mock_doc
            
            with patch('PIL.Image.frombytes') as mock_image:
                mock_img = Mock()
                mock_img.size = (800, 600)
                mock_img.save = Mock()
                mock_image.return_value = mock_img
                
                # Test the conversion
                result = processor.pdf_to_base64_images('test.pdf')
                assert len(result) == 2
                assert all(isinstance(img, str) for img in result)
    
    def test_document_hash_calculation(self):
        """Test document hash calculation"""
        processor = DocumentProcessor()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'test content')
            tmp_path = tmp.name
        
        try:
            hash1 = processor.calculate_document_hash(tmp_path)
            hash2 = processor.calculate_document_hash(tmp_path)
            
            # Hash should be consistent
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hash length
        finally:
            os.unlink(tmp_path)

class TestDataValidator:
    """Test data validation functionality"""
    
    def test_date_normalization(self):
        """Test date format normalization"""
        validator = DataValidator()
        
        # Test various date formats
        assert validator.normalize_date("01/01/1990") == "01/01/1990"
        assert validator.normalize_date("01-01-1990") == "01/01/1990"
        assert validator.normalize_date("01/01/90") == "01/01/1990"
        assert validator.normalize_date("invalid") == "invalid"
    
    def test_name_normalization(self):
        """Test name normalization"""
        validator = DataValidator()
        
        assert validator.normalize_name("  john  doe  ") == "JOHN DOE"
        assert validator.normalize_name("JANE SMITH") == "JANE SMITH"
        assert validator.normalize_name("") == ""
    
    def test_pan_validation(self):
        """Test PAN number validation"""
        validator = DataValidator()
        
        assert validator.validate_pan("ABCDE1234F") == True
        assert validator.validate_pan("INVALID123") == False
        assert validator.validate_pan("") == False
        assert validator.validate_pan("ABCDE12345") == False  # Too long
    
    def test_phone_validation(self):
        """Test phone number validation"""
        validator = DataValidator()
        
        assert validator.validate_phone("9876543210") == True
        assert validator.validate_phone("+91 9876543210") == True
        assert validator.validate_phone("123") == False
        assert validator.validate_phone("") == False
    
    def test_email_validation(self):
        """Test email validation"""
        validator = DataValidator()
        
        assert validator.validate_email("test@example.com") == True
        assert validator.validate_email("user.name+tag@domain.co.uk") == True
        assert validator.validate_email("invalid.email") == False
        assert validator.validate_email("") == False

class TestSecurityManager:
    """Test security features"""
    
    def test_document_authenticity_validation(self):
        """Test document authenticity checks"""
        security = SecurityManager()
        
        # Test with suspicious data
        suspicious_data = {
            'name': 'test test',
            'pan': 'ABCDE1234F'
        }
        result = security.validate_document_authenticity(suspicious_data)
        assert result['authenticity_score'] < 1.0
        assert len(result['warnings']) > 0
        
        # Test with normal data
        normal_data = {
            'name': 'John Smith',
            'pan': 'ABCDE1234F'
        }
        result = security.validate_document_authenticity(normal_data)
        assert result['authenticity_score'] > 0.5
    
    def test_sensitive_data_masking(self):
        """Test sensitive data masking"""
        security = SecurityManager()
        
        data = {
            'name': 'John Smith',
            'aadhaar_number': '123456789012',
            'pan_number': 'ABCDE1234F',
            'phone_number': '9876543210'
        }
        
        masked = security.mask_sensitive_data(data)
        
        assert masked['name'] == 'John Smith'  # Not sensitive
        assert masked['aadhaar_number'] == 'XXXXXXXX9012'
        assert masked['pan_number'] == 'XXXXXX234F'
        assert masked['phone_number'] == 'XXXXXX3210'

@pytest.mark.asyncio
class TestKYCAgent:
    """Test main KYC agent functionality"""
    
    async def test_kyc_processing_flow(self):
        """Test complete KYC processing flow"""
        # Mock the LLM client
        with patch('kyc_verification_agent.LLMClient') as mock_llm:
            mock_instance = AsyncMock()
            mock_instance.extract_information.return_value = json.dumps({
                'name': 'John Smith',
                'date_of_birth': '01/01/1990',
                'gender': 'Male'
            })
            mock_llm.return_value = mock_instance
            
            agent = KYCVerificationAgent('fake_api_key')
            
            # Mock document processor
            with patch.object(agent.document_processor, 'pdf_to_base64_images') as mock_convert:
                mock_convert.return_value = ['fake_base64_image']
                
                with patch.object(agent.document_processor, 'calculate_document_hash') as mock_hash:
                    mock_hash.return_value = 'fake_hash'
                    
                    # Create a temporary PDF file
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                        tmp.write(b'fake pdf content')
                        tmp_path = tmp.name
                    
                    try:
                        result = await agent.process_kyc_document(tmp_path)
                        
                        assert result.status in [VerificationStatus.ACCEPTED, VerificationStatus.REJECTED, VerificationStatus.PENDING]
                        assert isinstance(result.confidence_score, float)
                        assert result.processing_time > 0
                        assert result.document_hash == 'fake_hash'
                        
                    finally:
                        os.unlink(tmp_path)
    
    async def test_batch_processing(self):
        """Test batch document processing"""
        with patch('kyc_verification_agent.LLMClient') as mock_llm:
            mock_instance = AsyncMock()
            mock_instance.extract_information.return_value = json.dumps({
                'name': 'John Smith',
                'date_of_birth': '01/01/1990'
            })
            mock_llm.return_value = mock_instance
            
            agent = KYCVerificationAgent('fake_api_key')
            
            # Mock the process_kyc_document method
            with patch.object(agent, 'process_kyc_document') as mock_process:
                from kyc_verification_agent import VerificationResult
                mock_result = VerificationResult(
                    status=VerificationStatus.ACCEPTED,
                    confidence_score=0.95,
                    mismatched_attributes=[],
                    missing_attributes=[],
                    extracted_documents={},
                    processing_time=1.0,
                    errors=[],
                    document_hash='hash123'
                )
                mock_process.return_value = mock_result
                
                results = await agent.process_batch_documents(['file1.pdf', 'file2.pdf'])
                
                assert len(results) == 2
                assert all(isinstance(r, VerificationResult) for r in results)
                assert mock_process.call_count == 2

@pytest.mark.asyncio
class TestEnhancedKYCAgent:
    """Test enhanced KYC agent with additional features"""
    
    async def test_caching_functionality(self):
        """Test caching functionality"""
        config = {'redis_url': 'redis://fake'}
        
        with patch('enhanced_features.CacheManager') as mock_cache_manager:
            mock_cache = AsyncMock()
            mock_cache.get.return_value = None  # Cache miss
            mock_cache_manager.return_value = mock_cache
            
            with patch('enhanced_features.KYCVerificationAgent') as mock_base_agent:
                enhanced_agent = EnhancedKYCAgent('fake_api_key', config)
                
                # Test would require more complex mocking for full functionality
                assert enhanced_agent.cache_manager == mock_cache

class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.performance
    def test_processing_time_under_threshold(self):
        """Test that processing time is under acceptable threshold"""
        import time
        
        start_time = time.time()
        
        # Simulate processing
        validator = DataValidator()
        for i in range(1000):
            validator.normalize_name(f"Test Name {i}")
        
        processing_time = time.time() - start_time
        
        # Should process 1000 normalizations under 1 second
        assert processing_time < 1.0
    
    @pytest.mark.performance
    async def test_concurrent_processing(self):
        """Test concurrent processing capability"""
        import asyncio
        
        async def mock_process():
            await asyncio.sleep(0.1)  # Simulate processing
            return "processed"
        
        # Test processing 10 documents concurrently
        start_time = asyncio.get_event_loop().time()
        
        tasks = [mock_process() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        
        # Should complete all 10 in roughly the time of 1 (due to concurrency)
        assert len(results) == 10
        assert (end_time - start_time) < 0.2  # Allow some overhead

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.integration
    async def test_full_workflow_integration(self):
        """Test the complete workflow from upload to verification"""
        # This would test the entire flow in a real environment
        # with actual API calls, file processing, etc.
        pass
    
    @pytest.mark.integration
    def test_api_endpoints(self):
        """Test API endpoints"""
        # This would test the FastAPI endpoints
        from fastapi.testclient import TestClient
        from web_interface import app
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

# Fixtures for test data
@pytest.fixture
def sample_extracted_info():
    """Sample extracted information for testing"""
    return ExtractedInfo(
        name="John Smith",
        date_of_birth="01/01/1990",
        gender="Male",
        address="123 Main St, City, State",
        phone_number="9876543210",
        email="john@example.com",
        document_number="ABCDE1234F",
        document_type="PAN"
    )

@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(b'%PDF-1.4\nfake pdf content\n%%EOF')
        yield tmp.name
    os.unlink(tmp.name)

# Performance benchmark
def test_benchmark_data_validation():
    """Benchmark data validation operations"""
    import time
    
    validator = DataValidator()
    
    # Benchmark name normalization
    start = time.time()
    for i in range(10000):
        validator.normalize_name(f"  Test Name {i}  ")
    name_time = time.time() - start
    
    # Benchmark PAN validation
    start = time.time()
    for i in range(10000):
        validator.validate_pan("ABCDE1234F")
    pan_time = time.time() - start
    
    print(f"Name normalization: {name_time:.4f}s for 10k operations")
    print(f"PAN validation: {pan_time:.4f}s for 10k operations")
    
    # Ensure performance is acceptable
    assert name_time < 1.0
    assert pan_time < 0.5

# Conftest.py content for pytest configuration
conftest_content = '''
# conftest.py - Pytest configuration

import pytest
import asyncio
import os
from unittest.mock import patch

# Configure pytest-asyncio
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    os.environ['ANTHROPIC_API_KEY'] = 'test_key'
    
    yield
    
    # Cleanup
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client"""
    with patch('anthropic.Anthropic') as mock:
        yield mock

# Markers for different test types
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "security: mark test as security test")

# Custom test collection
def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location"""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
'''

if __name__ == "__main__":
    # Run tests
    pytest.main(["-v", "--cov=kyc_verification_agent", "--cov-report=html"])