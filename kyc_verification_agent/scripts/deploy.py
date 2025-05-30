#!/usr/bin/env python3
# Windows-compatible deployment script
import subprocess
import sys
import os
import time

def run_command(command, shell=True):
    try:
        result = subprocess.run(command, shell=shell, check=True, capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("Starting KYC Verification Agent...")
    
    # Check if virtual environment exists
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", "venv"], shell=False)
    
    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    print("Installing dependencies...")
    run_command(f"{pip_cmd} install -r requirements.txt")
    
    # Start services
    print("Starting backend...")
    backend_process = subprocess.Popen([python_cmd, "app/web_interface.py", "backend"])
    
    time.sleep(3)  # Wait for backend to start
    
    print("Starting frontend...")
    frontend_process = subprocess.Popen([python_cmd, "-m", "streamlit", "run", "app/web_interface.py", "--server.port=8501"])
    
    print("Services started!")
    print("Frontend: http://localhost:8501")
    print("Backend: http://localhost:8000")
    print("Press Ctrl+C to stop...")
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("Stopping services...")
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main()