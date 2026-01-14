"""
Data Provider Module for AlphaVantage Algo Trading Tools.

This module provides an abstract interface for data providers, enabling
seamless switching between data sources (AlphaVantage API, KDB+, etc.).
"""

from .base import DataProvider, PROVIDER_REGISTRY, get_provider, register_provider
from .alphavantage import AlphaVantageProvider

__all__ = [
    'DataProvider',
    'PROVIDER_REGISTRY',
    'get_provider',
    'register_provider',
    'AlphaVantageProvider',
]
