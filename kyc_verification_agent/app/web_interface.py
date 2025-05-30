# web_interface.py - FastAPI backend + Streamlit frontend

import streamlit as st
import requests
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from typing import List
import threading
import time
from pathlib import Path

# Import our KYC agent
from kyc_verification_agent import KYCVerificationAgent, KYCReportGenerator, VerificationResult

# FastAPI Backend
app = FastAPI(title="KYC Verification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for processing status
processing_status = {}
kyc_agent = None

@app.on_event("startup")
async def startup_event():
    global kyc_agent
    # Initialize KYC agent - replace with your API key
    api_key = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
    kyc_agent = KYCVerificationAgent(api_key)

@app.post("/api/upload-kyc-documents")
async def upload_kyc_documents(files: List[UploadFile] = File(...)):
    """Upload and process KYC documents"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    job_id = f"job_{int(time.time())}"
    processing_status[job_id] = {
        "status": "processing",
        "progress": 0,
        "total_files": len(files),
        "completed_files": 0,
        "results": []
    }
    
    # Save uploaded files temporarily
    temp_files = []
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        temp_files.append(temp_file.name)
    
    # Process files in background
    background_task = asyncio.create_task(process_files_background(job_id, temp_files))
    
    return {"job_id": job_id, "message": "Processing started", "total_files": len(files)}

async def process_files_background(job_id: str, file_paths: List[str]):
    """Process files in background"""
    try:
        results = await kyc_agent.process_batch_documents(file_paths)
        
        # Generate report
        report_generator = KYCReportGenerator()
        report = report_generator.generate_report(results)
        
        processing_status[job_id] = {
            "status": "completed",
            "progress": 100,
            "total_files": len(file_paths),
            "completed_files": len(file_paths),
            "results": report
        }
        
    except Exception as e:
        processing_status[job_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e)
        }
    finally:
        # Cleanup temp files
        for file_path in file_paths:
            try:
                os.unlink(file_path)
            except:
                pass

@app.get("/api/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_status[job_id]

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Streamlit Frontend
def run_streamlit():
    """Run Streamlit frontend"""
    
    st.set_page_config(
        page_title="KYC Verification Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Multi-Modal KYC Verification Agent")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 System Statistics")
    
    # Check backend health
    try:
        response = requests.get("http://localhost:8000/api/health")
        if response.status_code == 200:
            st.sidebar.success("✅ Backend Online")
        else:
            st.sidebar.error("❌ Backend Offline")
    except:
        st.sidebar.error("❌ Backend Offline")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Upload KYC Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload KYC documents (PDFs only)"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files uploaded")
            
            # Show file details
            with st.expander("📋 File Details"):
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size} bytes)")
            
            if st.button("🚀 Start Verification", type="primary"):
                # Upload files to backend
                files_data = []
                for file in uploaded_files:
                    files_data.append(('files', file))
                
                with st.spinner("🔄 Uploading and processing documents..."):
                    response = requests.post(
                        "http://localhost:8000/api/upload-kyc-documents",
                        files=files_data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        job_id = result['job_id']
                        st.session_state.job_id = job_id
                        st.success(f"✅ Processing started! Job ID: {job_id}")
                        st.rerun()
                    else:
                        st.error("❌ Error uploading files")
    
    with col2:
        st.header("⚙️ Processing Options")
        
        quality = st.slider("Image Quality", 50, 100, 85, help="Higher quality = better accuracy, slower processing")
        max_size = st.selectbox("Max Image Size", [(512, 512), (1024, 1024), (2048, 2048)], index=1)
        
        st.markdown("### 🎯 Verification Features")
        st.markdown("""
        - **Multi-document cross-verification**
        - **OCR + LLM hybrid extraction**
        - **Real-time processing status**
        - **Detailed mismatch reporting**
        - **Batch processing support**
        - **Document integrity checks**
        """)
    
    # Processing status
    if hasattr(st.session_state, 'job_id'):
        st.markdown("---")
        st.header("📊 Processing Status")
        
        # Get job status
        response = requests.get(f"http://localhost:8000/api/job-status/{st.session_state.job_id}")
        
        if response.status_code == 200:
            status_data = response.json()
            
            if status_data['status'] == 'processing':
                progress = status_data.get('progress', 0)
                st.progress(progress / 100)
                st.info(f"🔄 Processing... {status_data['completed_files']}/{status_data['total_files']} files completed")
                time.sleep(1)
                st.rerun()
                
            elif status_data['status'] == 'completed':
                st.success("✅ Processing completed!")
                
                # Display results
                results = status_data['results']
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Documents", results['summary']['total_documents'])
                with col2:
                    st.metric("Accepted", results['summary']['accepted'], 
                             delta=f"{results['summary']['success_rate']:.1f}%")
                with col3:
                    st.metric("Rejected", results['summary']['rejected'])
                with col4:
                    st.metric("Avg Time", f"{results['summary']['average_processing_time']:.2f}s")
                
                # Detailed results
                st.subheader("📋 Detailed Results")
                
                for i, result in enumerate(results['detailed_results']):
                    with st.expander(f"Document {i+1} - Status: {result['status']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Confidence Score:** {result['confidence_score']:.2f}")
                            st.write(f"**Processing Time:** {result['processing_time']:.2f}s")
                            
                            if result['mismatched_attributes']:
                                st.error("❌ Mismatched Attributes:")
                                for attr in result['mismatched_attributes']:
                                    st.write(f"• {attr}")
                        
                        with col2:
                            if result['missing_attributes']:
                                st.warning("⚠️ Missing Attributes:")
                                for attr in result['missing_attributes']:
                                    st.write(f"• {attr}")
                            
                            if result['errors']:
                                st.error("🚨 Errors:")
                                for error in result['errors']:
                                    st.write(f"• {error}")
                
                # Download report
                if st.button("📥 Download Report"):
                    st.download_button(
                        label="Download JSON Report",
                        data=json.dumps(results, indent=2),
                        file_name=f"kyc_report_{st.session_state.job_id}.json",
                        mime="application/json"
                    )
                
            elif status_data['status'] == 'error':
                st.error(f"❌ Processing failed: {status_data.get('error', 'Unknown error')}")
        
        else:
            st.error("❌ Could not fetch job status")

def run_backend():
    """Run FastAPI backend"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "backend":
        # Run backend
        run_backend()
    else:
        # Run Streamlit frontend
        run_streamlit()