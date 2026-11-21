"""
Data loading module for UD_Hindi-HDTB
"""

# Try both possible names
try:
    from .loader_paper import HUTBLoaderPaper
    __all__ = ['HUTBLoaderPaper']
except ImportError:
    try:
        from .loader import HUTBLoader as HUTBLoaderPaper
        __all__ = ['HUTBLoaderPaper']
    except ImportError:
        # If neither exists, create a helpful error
        print("Warning: Could not import HUTBLoader. Please ensure loader_paper.py or loader.py exists.")
        HUTBLoaderPaper = None
        __all__ = []
