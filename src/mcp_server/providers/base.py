"""
Abstract Base Class for Data Providers.

This module defines the interface that all data providers must implement,
enabling database-agnostic algorithmic trading tools.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
import pandas as pd

logger = logging.getLogger(__name__)

# Provider Registry for dynamic provider lookup
PROVIDER_REGISTRY: Dict[str, Type['DataProvider']] = {}


def register_provider(name: str):
    """Decorator to register a data provider class."""
    def wrapper(cls: Type['DataProvider']) -> Type['DataProvider']:
        PROVIDER_REGISTRY[name] = cls
        logger.info(f"Registered data provider: {name}")
        return cls
    return wrapper


def get_provider(name: str, **kwargs) -> 'DataProvider':
    """Get an instance of a registered data provider."""
    cls = PROVIDER_REGISTRY.get(name)
    if not cls:
        available = list(PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")
    return cls(**kwargs)


class DataProvider(ABC):
    """
    Abstract base class for data providers.

    All data providers must implement this interface to ensure compatibility
    with the algo trading tools. This enables seamless switching between
    data sources like AlphaVantage API, KDB+, PostgreSQL, etc.

    Standard Data Schemas:

    OHLCV DataFrame Columns:
    - timestamp: datetime - Candle timestamp
    - symbol: str - Ticker symbol
    - open: float - Opening price
    - high: float - High price
    - low: float - Low price
    - close: float - Closing price
    - volume: int - Trading volume

    News Sentiment DataFrame Columns:
    - timestamp: datetime - Article publication time
    - symbol: str - Related ticker
    - title: str - Article headline
    - summary: str - Article summary
    - source: str - News source
    - sentiment_score: float - Sentiment (-1 to 1)
    - sentiment_label: str - Bearish/Neutral/Bullish
    - relevance_score: float - Symbol relevance (0 to 1)
    """

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "60min",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_size: str = "compact"
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'MSFT')
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            output_size: 'compact' (100 data points) or 'full' (20+ years)

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume

        SQL Equivalent:
            SELECT timestamp, symbol, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = '{symbol}'
              AND timestamp >= '{start_time}'
              AND timestamp <= '{end_time}'
            ORDER BY timestamp ASC
        """
        pass

    @abstractmethod
    async def get_fx_ohlcv(
        self,
        from_currency: str,
        to_currency: str,
        interval: str = "60min",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_size: str = "compact"
    ) -> pd.DataFrame:
        """
        Get FX (Foreign Exchange) OHLCV data.

        Args:
            from_currency: Source currency code (e.g., 'EUR', 'GBP')
            to_currency: Target currency code (e.g., 'USD', 'JPY')
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            output_size: 'compact' or 'full'

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close
            Note: FX data typically doesn't include volume

        SQL Equivalent:
            SELECT timestamp, symbol, open, high, low, close
            FROM fx_ohlcv
            WHERE symbol = '{from_currency}/{to_currency}'
              AND timestamp >= '{start_time}'
              AND timestamp <= '{end_time}'
            ORDER BY timestamp ASC
        """
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with keys: symbol, open, high, low, price, volume,
            latest_trading_day, previous_close, change, change_percent

        SQL Equivalent:
            SELECT * FROM quotes WHERE symbol = '{symbol}'
            ORDER BY timestamp DESC LIMIT 1
        """
        pass

    @abstractmethod
    async def get_fx_rate(
        self,
        from_currency: str,
        to_currency: str
    ) -> Dict[str, Any]:
        """
        Get the current exchange rate between two currencies.

        Args:
            from_currency: Source currency code
            to_currency: Target currency code

        Returns:
            Dictionary with exchange rate information
        """
        pass

    @abstractmethod
    async def get_news_sentiment(
        self,
        symbols: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        sort: str = "LATEST",
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Get news sentiment data.

        Args:
            symbols: List of ticker symbols to filter by
            topics: List of topics (e.g., 'technology', 'finance')
            start_time: Start of time range
            end_time: End of time range
            sort: Sort order ('LATEST', 'EARLIEST', 'RELEVANCE')
            limit: Maximum number of results (up to 1000)

        Returns:
            DataFrame with news sentiment data

        SQL Equivalent:
            SELECT timestamp, symbol, title, summary, source,
                   sentiment_score, sentiment_label, relevance_score
            FROM news_sentiment
            WHERE symbol IN ({symbols})
              AND timestamp >= '{start_time}'
              AND timestamp <= '{end_time}'
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        pass

    @abstractmethod
    async def get_daily_ohlcv(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_size: str = "compact",
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Get daily OHLCV data.

        Args:
            symbol: Ticker symbol
            start_time: Start of time range
            end_time: End of time range
            output_size: 'compact' (100 days) or 'full' (20+ years)
            adjusted: Whether to use adjusted close prices

        Returns:
            DataFrame with daily OHLCV data
        """
        pass

    async def get_technical_indicator(
        self,
        function: str,
        symbol: str,
        interval: str = "60min",
        time_period: int = 20,
        series_type: str = "close",
        **kwargs
    ) -> pd.DataFrame:
        """
        Get pre-computed technical indicator from the data source.

        This is an optional method - not all providers may support
        pre-computed indicators. Falls back to computing locally.

        Args:
            function: Indicator function name (e.g., 'SMA', 'RSI')
            symbol: Ticker symbol
            interval: Time interval
            time_period: Lookback period
            series_type: Price series to use ('open', 'high', 'low', 'close')
            **kwargs: Additional indicator-specific parameters

        Returns:
            DataFrame with indicator values
        """
        raise NotImplementedError(
            f"Provider {self.__class__.__name__} does not support "
            "pre-computed technical indicators"
        )

    def validate_interval(self, interval: str) -> bool:
        """Validate that the interval is supported."""
        valid_intervals = ['1min', '5min', '15min', '30min', '60min']
        return interval in valid_intervals

    def validate_symbol(self, symbol: str) -> bool:
        """Basic symbol validation."""
        if not symbol:
            return False
        # Allow alphanumeric symbols with optional dots (e.g., BRK.A)
        return all(c.isalnum() or c in './-' for c in symbol)


class DataProviderError(Exception):
    """Base exception for data provider errors."""
    pass


class RateLimitError(DataProviderError):
    """Raised when API rate limit is exceeded."""
    pass


class InvalidSymbolError(DataProviderError):
    """Raised when an invalid symbol is requested."""
    pass


class DataNotFoundError(DataProviderError):
    """Raised when requested data is not available."""
    pass
