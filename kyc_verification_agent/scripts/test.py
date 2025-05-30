#!/usr/bin/env python3
# Windows-compatible test runner
import subprocess
import sys
import os

def main():
    print("Running tests...")
    
    if sys.platform == "win32":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    # Run tests
    subprocess.run([python_cmd, "-m", "pytest", "tests/", "-v", "--cov=app", "--cov-report=html"])
    
    print("Tests completed!")

if __name__ == "__main__":
    main()