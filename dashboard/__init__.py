"""
CaptureNet-Deep Dashboard

A PyQt5-based graphical interface for analyzing nanopore protein sequencing data
using the CaptureNet-Deep neural network model.
"""

from .start_screen import StartupScreen
from .import_fast5 import ImportFast5

__version__ = "1.0.0"
__all__ = ["StartupScreen", "ImportFast5"]