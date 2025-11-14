#!/usr/bin/env python3
"""
CaptureNet-Deep Dashboard Launcher Script

Simple launcher for the CaptureNet-Deep GUI dashboard.
"""

import sys
import subprocess
import os

def main():
    """Launch the dashboard or provide helpful information"""
    
    # Check if we're in the right directory
    if not os.path.exists('dashboard') or not os.path.exists('src'):
        print("Error: Please run this script from the CaptureNet-Deep root directory")
        print("   Expected structure: CaptureNet-Deep/dashboard/ and CaptureNet-Deep/src/")
        return 1
    
    print("Launching CaptureNet-Deep Dashboard...")
    print("   - Real-time 512-channel nanopore visualization")
    print("   - Automated capture phase detection")
    print("   - Interactive analysis interface")
    print()
    
    try:
        # Try to launch the dashboard directly
        dashboard_path = os.path.join(os.getcwd(), 'dashboard', 'start_screen.py')
        result = subprocess.run([sys.executable, dashboard_path], cwd=os.getcwd())
        return result.returncode
    except KeyboardInterrupt:
        print("\nDashboard closed by user")
        return 0
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("\nTroubleshooting:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Check PyQt5 installation: pip install PyQt5 pyqtgraph")
        print("   3. Generate demo data: python data/generate_demo_data.py --fast5-only")
        return 1

if __name__ == '__main__':
    sys.exit(main())