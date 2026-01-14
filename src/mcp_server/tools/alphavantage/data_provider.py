"""
Data Provider Wrapper for AlphaVantage Tools.

This module provides a unified interface for fetching data from different sources.
The data fetching is abstracted so that tools can work with any backend
(AlphaVantage API, KDB+, or other databases).

IMPORTANT: This module is designed to be easily swappable with other data sources.
To use a different data source, implement the same interface and update the
get_data_provider() function.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
import aiohttp
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Configuration
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHAVANTAGE_API_KEY = "MSJBKZQFTPOYB2PI"
RATE_LIMIT_CALLS = 75
RATE_LIMIT_PERIOD = 60.0


class RateLimiter:
    """Rate limiter for API calls (75 calls/minute)."""

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.calls = []
            cls._instance.max_calls = RATE_LIMIT_CALLS
            cls._instance.period = RATE_LIMIT_PERIOD
        return cls._instance

    async def acquire(self):
        """Acquire permission to make an API call."""
        async with self._lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]

            if len(self.calls) >= self.max_calls:
                oldest = min(self.calls)
                sleep_time = self.period - (now - oldest) + 0.1
                logger.warning(f"Rate limit: waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                now = time.time()
                self.calls = [t for t in self.calls if now - t < self.period]

            self.calls.append(time.time())

    @property
    def remaining(self) -> int:
        """Remaining calls in current window."""
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        return max(0, self.max_calls - len(self.calls))


class ResponseCache:
    """Simple cache for API responses."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
            cls._instance.default_ttl = 300
        return cls._instance

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int = None):
        self._cache[key] = (value, time.time() + (ttl or self.default_ttl))

    def clear(self):
        self._cache.clear()


# Global instances
_rate_limiter = RateLimiter()
_cache = ResponseCache()


async def fetch_alphavantage(
    params: Dict[str, Any],
    cache_key: str = None,
    cache_ttl: int = 300
) -> Dict[str, Any]:
    """
    Fetch data from AlphaVantage API with rate limiting and caching.

    Args:
        params: API query parameters
        cache_key: Optional cache key
        cache_ttl: Cache TTL in seconds

    Returns:
        API response as dictionary
    """
    if cache_key:
        cached = _cache.get(cache_key)
        if cached:
            return cached

    params['apikey'] = ALPHAVANTAGE_API_KEY
    await _rate_limiter.acquire()

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                ALPHAVANTAGE_BASE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                data = await response.json()

                if 'Error Message' in data:
                    raise ValueError(data['Error Message'])
                if 'Note' in data:
                    raise ValueError(f"Rate limit: {data['Note']}")

                if cache_key:
                    _cache.set(cache_key, data, cache_ttl)

                return data
        except asyncio.TimeoutError:
            raise ValueError("API request timed out")
        except aiohttp.ClientError as e:
            raise ValueError(f"Network error: {e}")


async def get_ohlcv(
    symbol: str,
    interval: str = "60min",
    output_size: str = "compact",
    data_source: str = "alphavantage"
) -> pd.DataFrame:
    """
    Get OHLCV data for equities.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        interval: Time interval ('1min', '5min', '15min', '30min', '60min')
        output_size: 'compact' (100 points) or 'full'
        data_source: Data source ('alphavantage' or future: 'kdb')

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume

    SQL Equivalent:
        SELECT timestamp, symbol, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = '{symbol}'
        ORDER BY timestamp ASC
    """
    cache_key = f"ohlcv:{symbol}:{interval}:{output_size}"

    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol.upper(),
        'interval': interval,
        'outputsize': output_size,
        'adjusted': 'true'
    }

    data = await fetch_alphavantage(params, cache_key, 300)

    time_series_key = f'Time Series ({interval})'
    if time_series_key not in data:
        raise ValueError(f"No data for {symbol}. Keys: {list(data.keys())}")

    records = []
    for ts_str, values in data[time_series_key].items():
        records.append({
            'timestamp': pd.to_datetime(ts_str),
            'symbol': symbol.upper(),
            'open': float(values.get('1. open', 0)),
            'high': float(values.get('2. high', 0)),
            'low': float(values.get('3. low', 0)),
            'close': float(values.get('4. close', 0)),
            'volume': int(float(values.get('5. volume', 0)))
        })

    df = pd.DataFrame(records)
    return df.sort_values('timestamp').reset_index(drop=True)


async def get_fx_ohlcv(
    from_currency: str,
    to_currency: str,
    interval: str = "60min",
    output_size: str = "compact"
) -> pd.DataFrame:
    """
    Get OHLCV data for forex pairs.

    Args:
        from_currency: Source currency (e.g., 'EUR')
        to_currency: Target currency (e.g., 'USD')
        interval: Time interval
        output_size: 'compact' or 'full'

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close

    SQL Equivalent:
        SELECT timestamp, symbol, open, high, low, close
        FROM fx_ohlcv
        WHERE symbol = '{from_currency}/{to_currency}'
        ORDER BY timestamp ASC
    """
    symbol = f"{from_currency.upper()}/{to_currency.upper()}"
    cache_key = f"fx:{symbol}:{interval}:{output_size}"

    params = {
        'function': 'FX_INTRADAY',
        'from_symbol': from_currency.upper(),
        'to_symbol': to_currency.upper(),
        'interval': interval,
        'outputsize': output_size
    }

    data = await fetch_alphavantage(params, cache_key, 300)

    time_series_key = 'Time Series FX (Intraday)'
    if time_series_key not in data:
        raise ValueError(f"No FX data for {symbol}")

    records = []
    for ts_str, values in data[time_series_key].items():
        records.append({
            'timestamp': pd.to_datetime(ts_str),
            'symbol': symbol,
            'open': float(values.get('1. open', 0)),
            'high': float(values.get('2. high', 0)),
            'low': float(values.get('3. low', 0)),
            'close': float(values.get('4. close', 0))
        })

    df = pd.DataFrame(records)
    return df.sort_values('timestamp').reset_index(drop=True)


async def get_daily_ohlcv(
    symbol: str,
    output_size: str = "compact"
) -> pd.DataFrame:
    """
    Get daily OHLCV data for equities.

    Args:
        symbol: Stock ticker symbol
        output_size: 'compact' (100 days) or 'full' (20+ years)

    Returns:
        DataFrame with daily OHLCV data
    """
    cache_key = f"daily:{symbol}:{output_size}"

    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol.upper(),
        'outputsize': output_size
    }

    data = await fetch_alphavantage(params, cache_key, 3600)

    time_series_key = 'Time Series (Daily)'
    if time_series_key not in data:
        raise ValueError(f"No daily data for {symbol}")

    records = []
    for ts_str, values in data[time_series_key].items():
        records.append({
            'timestamp': pd.to_datetime(ts_str),
            'symbol': symbol.upper(),
            'open': float(values.get('1. open', 0)),
            'high': float(values.get('2. high', 0)),
            'low': float(values.get('3. low', 0)),
            'close': float(values.get('4. close', 0)),
            'adjusted_close': float(values.get('5. adjusted close', 0)),
            'volume': int(float(values.get('6. volume', 0)))
        })

    df = pd.DataFrame(records)
    return df.sort_values('timestamp').reset_index(drop=True)


async def get_quote(symbol: str) -> Dict[str, Any]:
    """
    Get latest quote for a symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with quote information
    """
    cache_key = f"quote:{symbol}"

    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol.upper()
    }

    data = await fetch_alphavantage(params, cache_key, 60)

    if 'Global Quote' not in data or not data['Global Quote']:
        raise ValueError(f"No quote for {symbol}")

    q = data['Global Quote']
    return {
        'symbol': q.get('01. symbol', symbol),
        'open': float(q.get('02. open', 0)),
        'high': float(q.get('03. high', 0)),
        'low': float(q.get('04. low', 0)),
        'price': float(q.get('05. price', 0)),
        'volume': int(float(q.get('06. volume', 0))),
        'previous_close': float(q.get('08. previous close', 0)),
        'change': float(q.get('09. change', 0)),
        'change_percent': q.get('10. change percent', '0%')
    }


async def get_news_sentiment(
    symbols: List[str] = None,
    topics: List[str] = None,
    limit: int = 50
) -> pd.DataFrame:
    """
    Get news sentiment data.

    Args:
        symbols: List of ticker symbols to filter
        topics: List of topics to filter
        limit: Maximum results (up to 1000)

    Returns:
        DataFrame with sentiment data

    SQL Equivalent:
        SELECT timestamp, symbol, title, sentiment_score, sentiment_label, relevance_score
        FROM news_sentiment
        WHERE symbol IN ({symbols})
        ORDER BY timestamp DESC
        LIMIT {limit}
    """
    params = {'function': 'NEWS_SENTIMENT', 'limit': min(limit, 1000)}

    if symbols:
        params['tickers'] = ','.join([s.upper() for s in symbols])
    if topics:
        params['topics'] = ','.join(topics)

    cache_key = f"news:{params.get('tickers', 'all')}:{limit}"
    data = await fetch_alphavantage(params, cache_key, 300)

    if 'feed' not in data:
        return pd.DataFrame()

    records = []
    for article in data['feed']:
        ts = article.get('ticker_sentiment', [])

        if symbols and ts:
            for t in ts:
                ticker = t.get('ticker', '')
                if ticker.upper() in [s.upper() for s in symbols]:
                    records.append({
                        'timestamp': pd.to_datetime(
                            article.get('time_published', ''),
                            format='%Y%m%dT%H%M%S'
                        ),
                        'symbol': ticker,
                        'title': article.get('title', ''),
                        'source': article.get('source', ''),
                        'sentiment_score': float(t.get('ticker_sentiment_score', 0)),
                        'sentiment_label': t.get('ticker_sentiment_label', 'Neutral'),
                        'relevance_score': float(t.get('relevance_score', 0))
                    })
        else:
            records.append({
                'timestamp': pd.to_datetime(
                    article.get('time_published', ''),
                    format='%Y%m%dT%H%M%S'
                ),
                'symbol': symbols[0] if symbols else 'MARKET',
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'sentiment_score': float(article.get('overall_sentiment_score', 0)),
                'sentiment_label': article.get('overall_sentiment_label', 'Neutral'),
                'relevance_score': 1.0
            })

    if records:
        df = pd.DataFrame(records)
        return df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    return pd.DataFrame()


async def get_fx_rate(from_currency: str, to_currency: str) -> Dict[str, Any]:
    """
    Get current FX exchange rate.

    Args:
        from_currency: Source currency
        to_currency: Target currency

    Returns:
        Dictionary with exchange rate info
    """
    cache_key = f"fx_rate:{from_currency}:{to_currency}"

    params = {
        'function': 'CURRENCY_EXCHANGE_RATE',
        'from_currency': from_currency.upper(),
        'to_currency': to_currency.upper()
    }

    data = await fetch_alphavantage(params, cache_key, 60)

    if 'Realtime Currency Exchange Rate' not in data:
        raise ValueError(f"No rate for {from_currency}/{to_currency}")

    r = data['Realtime Currency Exchange Rate']
    return {
        'from_currency': r.get('1. From_Currency Code', from_currency),
        'to_currency': r.get('3. To_Currency Code', to_currency),
        'exchange_rate': float(r.get('5. Exchange Rate', 0)),
        'last_refreshed': r.get('6. Last Refreshed', ''),
        'bid_price': float(r.get('8. Bid Price', 0)) if r.get('8. Bid Price') else None,
        'ask_price': float(r.get('9. Ask Price', 0)) if r.get('9. Ask Price') else None
    }


def get_rate_limit_status() -> Dict[str, Any]:
    """Get current rate limit status."""
    return {
        'remaining': _rate_limiter.remaining,
        'max_calls': RATE_LIMIT_CALLS,
        'period_seconds': RATE_LIMIT_PERIOD
    }
