#!/usr/bin/env python3
"""
CaptureNet-Deep Dashboard Launcher

Usage:
    python -m dashboard
    
Or directly:
    python dashboard/main.py
"""

import sys
import os

# Add the parent directory to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication

try:
    # Try relative import first
    from .start_screen import StartupScreen
except ImportError:
    # Fall back to direct import
    from start_screen import StartupScreen

def main():
    """Launch the CaptureNet-Deep Dashboard"""
    app = QApplication(sys.argv)
    window = StartupScreen(app)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()