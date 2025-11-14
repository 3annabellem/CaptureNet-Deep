#!/usr/bin/env python3
"""
Simple Dashboard Launcher

This script tries multiple methods to launch the CaptureNet-Deep dashboard
and provides helpful error messages if something goes wrong.
"""

import os
import sys
import subprocess
import shutil

def find_python():
    """Find Python executable using multiple methods"""
    
    # Method 1: Current Python (if we're running in Python)
    if hasattr(sys, 'executable') and sys.executable:
        return sys.executable
    
    # Method 2: Check common Windows locations
    common_paths = [
        r'C:\Users\annab\AppData\Local\Programs\Python\Python312\python.exe',
        r'C:\Users\annab\AppData\Local\Programs\Python\Python311\python.exe',
        r'C:\Python312\python.exe',
        r'C:\Python311\python.exe',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # Method 3: Check PATH
    python_exe = shutil.which('python')
    if python_exe:
        return python_exe
        
    python3_exe = shutil.which('python3')
    if python3_exe:
        return python3_exe
    
    return None

def main():
    print("CaptureNet-Deep Dashboard Launcher")
    print("=" * 40)
    
    # Find Python
    python_exe = find_python()
    if not python_exe:
        print("Error: Could not find Python executable")
        print("\nTroubleshooting:")
        print("1. Make sure Python is installed")
        print("2. Add Python to your PATH, or")
        print("3. Edit this script to point to your Python installation")
        return 1
    
    print(f"Found Python: {python_exe}")
    
    # Check if dashboard exists
    dashboard_script = os.path.join(os.path.dirname(__file__), 'dashboard', 'start_screen.py')
    if not os.path.exists(dashboard_script):
        print(f"Error: Dashboard script not found at {dashboard_script}")
        print("\nMake sure you're running this from the CaptureNet-Deep directory")
        return 1
    
    print(f"Found dashboard: {dashboard_script}")
    
    # Try to launch
    print("\nLaunching dashboard...")
    try:
        result = subprocess.run([python_exe, dashboard_script], 
                              cwd=os.path.dirname(__file__))
        return result.returncode
    except KeyboardInterrupt:
        print("\nDashboard closed by user")
        return 0
    except Exception as e:
        print(f"\nError launching dashboard: {e}")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check PyQt5: pip install PyQt5 pyqtgraph")
        print("3. Try running directly: python dashboard/start_screen.py")
        return 1

if __name__ == '__main__':
    sys.exit(main())