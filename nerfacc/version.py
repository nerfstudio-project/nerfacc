"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
try:
    from importlib.metadata import version
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    from importlib_metadata import version

__version__ = version("nerfacc")
