"""
AlphaVantage Data Provider Implementation.

This module implements the DataProvider interface for the AlphaVantage API,
with built-in rate limiting (75 calls/minute) and response caching.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional
import aiohttp
import pandas as pd

from .base import (
    DataProvider,
    DataProviderError,
    RateLimitError,
    InvalidSymbolError,
    DataNotFoundError,
    register_provider
)

logger = logging.getLogger(__name__)

# AlphaVantage API Configuration
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_API_KEY = "MSJBKZQFTPOYB2PI"


class RateLimiter:
    """
    Rate limiter for API calls.

    Implements a sliding window rate limiter to ensure we stay
    within the 75 calls/minute limit.
    """

    def __init__(self, max_calls: int = 75, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make an API call, waiting if necessary."""
        async with self._lock:
            now = time.time()
            # Remove calls outside the current window
            self.calls = [t for t in self.calls if now - t < self.period]

            if len(self.calls) >= self.max_calls:
                # Calculate wait time
                oldest_call = min(self.calls)
                sleep_time = self.period - (now - oldest_call) + 0.1
                logger.warning(
                    f"Rate limit reached. Waiting {sleep_time:.1f}s "
                    f"({len(self.calls)}/{self.max_calls} calls in window)"
                )
                await asyncio.sleep(sleep_time)
                # Clean up again after sleeping
                now = time.time()
                self.calls = [t for t in self.calls if now - t < self.period]

            self.calls.append(time.time())

    @property
    def remaining(self) -> int:
        """Get remaining calls in the current window."""
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        return max(0, self.max_calls - len(self.calls))


class ResponseCache:
    """
    Simple in-memory cache for API responses.

    Caches responses to reduce API calls for repeated requests.
    Default TTL is 5 minutes for real-time data, 1 hour for historical.
    """

    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value if it exists and hasn't expired."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                logger.debug(f"Cache hit: {key[:50]}...")
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Cache a value with optional custom TTL."""
        expiry = time.time() + (ttl or self.default_ttl)
        self._cache[key] = (value, expiry)
        logger.debug(f"Cached: {key[:50]}... (TTL: {ttl or self.default_ttl}s)")

    def clear(self):
        """Clear all cached values."""
        self._cache.clear()


# Global rate limiter and cache instances
_rate_limiter = RateLimiter(max_calls=75, period=60.0)
_response_cache = ResponseCache(default_ttl=300)


@register_provider("alphavantage")
class AlphaVantageProvider(DataProvider):
    """
    AlphaVantage API Data Provider.

    Provides access to stock, forex, and news sentiment data through
    the AlphaVantage API with built-in rate limiting and caching.

    Usage:
        provider = AlphaVantageProvider(api_key="YOUR_KEY")
        data = await provider.get_ohlcv("AAPL", interval="60min")
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEFAULT_API_KEY
        self.base_url = ALPHAVANTAGE_BASE_URL
        self.rate_limiter = _rate_limiter
        self.cache = _response_cache

    async def _make_request(
        self,
        params: Dict[str, Any],
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make an API request with rate limiting and caching.

        Args:
            params: API query parameters
            cache_key: Optional cache key for the response
            cache_ttl: Optional TTL for cached response

        Returns:
            API response as dictionary
        """
        # Check cache first
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Add API key to params
        params['apikey'] = self.api_key

        # Acquire rate limit permission
        await self.rate_limiter.acquire()

        # Make the request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    self.base_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        raise DataProviderError(
                            f"API request failed with status {response.status}"
                        )

                    data = await response.json()

                    # Check for API errors
                    if 'Error Message' in data:
                        raise InvalidSymbolError(data['Error Message'])

                    if 'Note' in data:
                        # Rate limit message from AlphaVantage
                        raise RateLimitError(data['Note'])

                    if 'Information' in data:
                        # Usually indicates rate limit or subscription issue
                        logger.warning(f"API Info: {data['Information']}")

                    # Cache the response
                    if cache_key:
                        self.cache.set(cache_key, data, cache_ttl)

                    return data

            except asyncio.TimeoutError:
                raise DataProviderError("API request timed out")
            except aiohttp.ClientError as e:
                raise DataProviderError(f"Network error: {str(e)}")

    def _parse_ohlcv(
        self,
        data: Dict[str, Any],
        symbol: str,
        time_series_key: str
    ) -> pd.DataFrame:
        """Parse OHLCV data from API response into DataFrame."""
        if time_series_key not in data:
            available_keys = list(data.keys())
            raise DataNotFoundError(
                f"No time series data found. Available keys: {available_keys}"
            )

        time_series = data[time_series_key]
        records = []

        for timestamp_str, values in time_series.items():
            record = {
                'timestamp': pd.to_datetime(timestamp_str),
                'symbol': symbol,
                'open': float(values.get('1. open', 0)),
                'high': float(values.get('2. high', 0)),
                'low': float(values.get('3. low', 0)),
                'close': float(values.get('4. close', 0)),
                'volume': int(float(values.get('5. volume', 0))) if '5. volume' in values else 0
            }
            records.append(record)

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _parse_fx_ohlcv(
        self,
        data: Dict[str, Any],
        symbol: str,
        time_series_key: str
    ) -> pd.DataFrame:
        """Parse FX OHLCV data from API response into DataFrame."""
        if time_series_key not in data:
            available_keys = list(data.keys())
            raise DataNotFoundError(
                f"No FX time series data found. Available keys: {available_keys}"
            )

        time_series = data[time_series_key]
        records = []

        for timestamp_str, values in time_series.items():
            record = {
                'timestamp': pd.to_datetime(timestamp_str),
                'symbol': symbol,
                'open': float(values.get('1. open', 0)),
                'high': float(values.get('2. high', 0)),
                'low': float(values.get('3. low', 0)),
                'close': float(values.get('4. close', 0))
            }
            records.append(record)

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "60min",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_size: str = "compact"
    ) -> pd.DataFrame:
        """
        Get intraday OHLCV data for a stock symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            output_size: 'compact' (100 points) or 'full' (full history)

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume

        Example:
            >>> data = await provider.get_ohlcv('AAPL', interval='60min')
            >>> print(data.columns)
            Index(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        """
        if not self.validate_symbol(symbol):
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")

        if not self.validate_interval(interval):
            raise DataProviderError(
                f"Invalid interval: {interval}. "
                "Valid: 1min, 5min, 15min, 30min, 60min"
            )

        # Build cache key
        cache_key = f"ohlcv:{symbol}:{interval}:{output_size}"

        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol.upper(),
            'interval': interval,
            'outputsize': output_size,
            'adjusted': 'true'
        }

        data = await self._make_request(
            params,
            cache_key=cache_key,
            cache_ttl=300  # 5 minutes for intraday
        )

        # Parse the response
        time_series_key = f'Time Series ({interval})'
        df = self._parse_ohlcv(data, symbol.upper(), time_series_key)

        # Filter by time range if specified
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]

        return df

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
        Get intraday FX OHLCV data.

        Args:
            from_currency: Source currency code (e.g., 'EUR')
            to_currency: Target currency code (e.g., 'USD')
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            start_time: Start of time range
            end_time: End of time range
            output_size: 'compact' or 'full'

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close

        Example:
            >>> data = await provider.get_fx_ohlcv('EUR', 'USD', interval='60min')
        """
        symbol = f"{from_currency.upper()}/{to_currency.upper()}"
        cache_key = f"fx_ohlcv:{symbol}:{interval}:{output_size}"

        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': from_currency.upper(),
            'to_symbol': to_currency.upper(),
            'interval': interval,
            'outputsize': output_size
        }

        data = await self._make_request(
            params,
            cache_key=cache_key,
            cache_ttl=300
        )

        time_series_key = f'Time Series FX (Intraday)'
        df = self._parse_fx_ohlcv(data, symbol, time_series_key)

        # Filter by time range if specified
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]

        return df

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest quote for a stock symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with quote information

        Example:
            >>> quote = await provider.get_quote('AAPL')
            >>> print(quote['price'])
        """
        if not self.validate_symbol(symbol):
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")

        cache_key = f"quote:{symbol}"

        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol.upper()
        }

        data = await self._make_request(
            params,
            cache_key=cache_key,
            cache_ttl=60  # 1 minute for quotes
        )

        if 'Global Quote' not in data or not data['Global Quote']:
            raise DataNotFoundError(f"No quote data for symbol: {symbol}")

        quote = data['Global Quote']
        return {
            'symbol': quote.get('01. symbol', symbol),
            'open': float(quote.get('02. open', 0)),
            'high': float(quote.get('03. high', 0)),
            'low': float(quote.get('04. low', 0)),
            'price': float(quote.get('05. price', 0)),
            'volume': int(float(quote.get('06. volume', 0))),
            'latest_trading_day': quote.get('07. latest trading day', ''),
            'previous_close': float(quote.get('08. previous close', 0)),
            'change': float(quote.get('09. change', 0)),
            'change_percent': quote.get('10. change percent', '0%')
        }

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
        cache_key = f"fx_rate:{from_currency}:{to_currency}"

        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': from_currency.upper(),
            'to_currency': to_currency.upper()
        }

        data = await self._make_request(
            params,
            cache_key=cache_key,
            cache_ttl=60
        )

        if 'Realtime Currency Exchange Rate' not in data:
            raise DataNotFoundError(
                f"No exchange rate for {from_currency}/{to_currency}"
            )

        rate = data['Realtime Currency Exchange Rate']
        return {
            'from_currency': rate.get('1. From_Currency Code', from_currency),
            'to_currency': rate.get('3. To_Currency Code', to_currency),
            'exchange_rate': float(rate.get('5. Exchange Rate', 0)),
            'last_refreshed': rate.get('6. Last Refreshed', ''),
            'timezone': rate.get('7. Time Zone', 'UTC'),
            'bid_price': float(rate.get('8. Bid Price', 0)),
            'ask_price': float(rate.get('9. Ask Price', 0))
        }

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
            topics: List of topics (e.g., 'technology', 'earnings')
            start_time: Start of time range
            end_time: End of time range
            sort: Sort order ('LATEST', 'EARLIEST', 'RELEVANCE')
            limit: Maximum number of results (up to 1000)

        Returns:
            DataFrame with news sentiment data

        Example:
            >>> sentiment = await provider.get_news_sentiment(['AAPL', 'MSFT'])
        """
        params = {
            'function': 'NEWS_SENTIMENT',
            'sort': sort,
            'limit': min(limit, 1000)
        }

        if symbols:
            params['tickers'] = ','.join([s.upper() for s in symbols])

        if topics:
            params['topics'] = ','.join(topics)

        if start_time:
            params['time_from'] = start_time.strftime('%Y%m%dT%H%M')

        if end_time:
            params['time_to'] = end_time.strftime('%Y%m%dT%H%M')

        # Build cache key
        cache_parts = [
            'news',
            params.get('tickers', 'all'),
            params.get('topics', 'all'),
            sort,
            str(limit)
        ]
        cache_key = ':'.join(cache_parts)

        data = await self._make_request(
            params,
            cache_key=cache_key,
            cache_ttl=300  # 5 minutes
        )

        if 'feed' not in data:
            logger.warning("No news feed in response")
            return pd.DataFrame()

        records = []
        for article in data['feed']:
            # Extract ticker sentiment
            ticker_sentiments = article.get('ticker_sentiment', [])

            if symbols and ticker_sentiments:
                # Filter to requested symbols
                for ts in ticker_sentiments:
                    ticker = ts.get('ticker', '')
                    if ticker.upper() in [s.upper() for s in symbols]:
                        record = {
                            'timestamp': pd.to_datetime(
                                article.get('time_published', ''),
                                format='%Y%m%dT%H%M%S'
                            ),
                            'symbol': ticker,
                            'title': article.get('title', ''),
                            'summary': article.get('summary', ''),
                            'source': article.get('source', ''),
                            'url': article.get('url', ''),
                            'sentiment_score': float(ts.get('ticker_sentiment_score', 0)),
                            'sentiment_label': ts.get('ticker_sentiment_label', 'Neutral'),
                            'relevance_score': float(ts.get('relevance_score', 0)),
                            'overall_sentiment_score': float(
                                article.get('overall_sentiment_score', 0)
                            ),
                            'overall_sentiment_label': article.get(
                                'overall_sentiment_label', 'Neutral'
                            )
                        }
                        records.append(record)
            else:
                # No symbol filter, use overall sentiment
                record = {
                    'timestamp': pd.to_datetime(
                        article.get('time_published', ''),
                        format='%Y%m%dT%H%M%S'
                    ),
                    'symbol': symbols[0] if symbols else 'MARKET',
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', ''),
                    'sentiment_score': float(
                        article.get('overall_sentiment_score', 0)
                    ),
                    'sentiment_label': article.get(
                        'overall_sentiment_label', 'Neutral'
                    ),
                    'relevance_score': 1.0,
                    'overall_sentiment_score': float(
                        article.get('overall_sentiment_score', 0)
                    ),
                    'overall_sentiment_label': article.get(
                        'overall_sentiment_label', 'Neutral'
                    )
                }
                records.append(record)

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

        return df

    async def get_daily_ohlcv(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_size: str = "compact",
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Get daily OHLCV data for a stock symbol.

        Args:
            symbol: Stock ticker symbol
            start_time: Start of time range
            end_time: End of time range
            output_size: 'compact' (100 days) or 'full' (20+ years)
            adjusted: Whether to use adjusted prices

        Returns:
            DataFrame with daily OHLCV data
        """
        if not self.validate_symbol(symbol):
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")

        function = 'TIME_SERIES_DAILY_ADJUSTED' if adjusted else 'TIME_SERIES_DAILY'
        cache_key = f"daily:{symbol}:{output_size}:{adjusted}"

        params = {
            'function': function,
            'symbol': symbol.upper(),
            'outputsize': output_size
        }

        data = await self._make_request(
            params,
            cache_key=cache_key,
            cache_ttl=3600  # 1 hour for daily data
        )

        time_series_key = 'Time Series (Daily)'
        time_series = data.get(time_series_key, {})

        records = []
        for timestamp_str, values in time_series.items():
            record = {
                'timestamp': pd.to_datetime(timestamp_str),
                'symbol': symbol.upper(),
                'open': float(values.get('1. open', 0)),
                'high': float(values.get('2. high', 0)),
                'low': float(values.get('3. low', 0)),
                'close': float(values.get('4. close', 0)),
                'volume': int(float(values.get('6. volume', values.get('5. volume', 0))))
            }
            if adjusted and '5. adjusted close' in values:
                record['adjusted_close'] = float(values.get('5. adjusted close', 0))
                record['dividend'] = float(values.get('7. dividend amount', 0))
                record['split_coefficient'] = float(values.get('8. split coefficient', 1))
            records.append(record)

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Filter by time range if specified
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]

        return df

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
        Get pre-computed technical indicator from AlphaVantage.

        AlphaVantage provides pre-computed indicators which can be more
        efficient than computing locally for large datasets.

        Args:
            function: Indicator name (e.g., 'SMA', 'RSI', 'MACD')
            symbol: Stock ticker symbol
            interval: Time interval
            time_period: Lookback period
            series_type: Price series ('open', 'high', 'low', 'close')
            **kwargs: Additional parameters specific to the indicator

        Returns:
            DataFrame with indicator values
        """
        cache_key = f"indicator:{function}:{symbol}:{interval}:{time_period}"

        params = {
            'function': function.upper(),
            'symbol': symbol.upper(),
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type,
            **kwargs
        }

        data = await self._make_request(
            params,
            cache_key=cache_key,
            cache_ttl=300
        )

        # Find the technical analysis key
        ta_key = None
        for key in data.keys():
            if 'Technical Analysis' in key:
                ta_key = key
                break

        if not ta_key:
            raise DataNotFoundError(
                f"No technical indicator data found for {function}"
            )

        records = []
        for timestamp_str, values in data[ta_key].items():
            record = {
                'timestamp': pd.to_datetime(timestamp_str),
                'symbol': symbol.upper()
            }
            # Add all indicator values
            for key, value in values.items():
                record[key.lower()] = float(value)
            records.append(record)

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    @property
    def rate_limit_remaining(self) -> int:
        """Get remaining API calls in current rate limit window."""
        return self.rate_limiter.remaining
