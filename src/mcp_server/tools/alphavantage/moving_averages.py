"""
Moving Average Tools for Algorithmic Trading.

This module provides moving average calculations:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- WMA (Weighted Moving Average)
- DEMA (Double Exponential Moving Average)
- TEMA (Triple Exponential Moving Average)

All tools support both equities and forex data.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

from .data_provider import get_ohlcv, get_fx_ohlcv

logger = logging.getLogger(__name__)


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_wma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return data.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def calculate_dema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Double Exponential Moving Average."""
    ema1 = calculate_ema(data, period)
    ema2 = calculate_ema(ema1, period)
    return 2 * ema1 - ema2


def calculate_tema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Triple Exponential Moving Average."""
    ema1 = calculate_ema(data, period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


async def av_sma_impl(
    symbol: str,
    period: int = 20,
    interval: str = "60min",
    price_type: str = "close",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Simple Moving Average (SMA).

    The SMA is the unweighted mean of the previous n data points.
    It's commonly used to identify trend direction and potential support/resistance levels.

    Formula: SMA = (P1 + P2 + ... + Pn) / n

    Required Data Columns:
        - close (or specified price_type): Price data
        - timestamp: Time of the candle

    SQL Equivalent:
        SELECT timestamp, close,
               AVG(close) OVER (ORDER BY timestamp ROWS BETWEEN {period-1} PRECEDING AND CURRENT ROW) as sma
        FROM ohlcv
        WHERE symbol = '{symbol}'
        ORDER BY timestamp ASC

    Args:
        symbol: Stock ticker (e.g., 'AAPL') or ignored if is_fx=True
        period: Number of periods for averaging (default: 20)
        interval: Time interval ('1min', '5min', '15min', '30min', '60min')
        price_type: Price to use ('open', 'high', 'low', 'close')
        is_fx: Whether this is a forex pair
        from_currency: Source currency for FX (e.g., 'EUR')
        to_currency: Target currency for FX (e.g., 'USD')

    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - symbol: The ticker symbol
        - period: SMA period used
        - current_sma: Latest SMA value
        - current_price: Latest price
        - trend: 'bullish' if price > SMA, 'bearish' otherwise
        - data: List of {timestamp, price, sma} records
        - metadata: Column requirements for database integration
    """
    try:
        # Get data based on asset type
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty:
            return {"status": "error", "message": f"No data available for {symbol}"}

        if len(df) < period:
            return {
                "status": "error",
                "message": f"Insufficient data. Need {period} points, got {len(df)}"
            }

        # Calculate SMA
        df['sma'] = calculate_sma(df[price_type], period)

        # Get results
        result_df = df[['timestamp', price_type, 'sma']].dropna()
        current_price = result_df.iloc[-1][price_type]
        current_sma = result_df.iloc[-1]['sma']
        trend = "bullish" if current_price > current_sma else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "price_type": price_type,
            "current_price": round(current_price, 4),
            "current_sma": round(current_sma, 4),
            "trend": trend,
            "price_vs_sma_pct": round((current_price / current_sma - 1) * 100, 2),
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": [price_type, "timestamp"],
                "sql_template": f"SELECT timestamp, {price_type} FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"SMA calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_ema_impl(
    symbol: str,
    period: int = 20,
    interval: str = "60min",
    price_type: str = "close",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Exponential Moving Average (EMA).

    The EMA gives more weight to recent prices, making it more responsive
    to new information than the SMA.

    Formula: EMA = Price * k + EMA(prev) * (1 - k), where k = 2 / (period + 1)

    Required Data Columns:
        - close (or specified price_type): Price data
        - timestamp: Time of the candle

    SQL Equivalent:
        -- EMA requires recursive calculation or window function
        -- This is a simplified representation
        SELECT timestamp, close, ema_{period}
        FROM technical_indicators
        WHERE symbol = '{symbol}'

    Args:
        symbol: Stock ticker or ignored if is_fx=True
        period: Number of periods for EMA (default: 20)
        interval: Time interval
        price_type: Price to use ('open', 'high', 'low', 'close')
        is_fx: Whether this is a forex pair
        from_currency: Source currency for FX
        to_currency: Target currency for FX

    Returns:
        Dictionary with EMA values and trend analysis
    """
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < period:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['ema'] = calculate_ema(df[price_type], period)
        result_df = df[['timestamp', price_type, 'ema']].dropna()

        current_price = result_df.iloc[-1][price_type]
        current_ema = result_df.iloc[-1]['ema']
        prev_ema = result_df.iloc[-2]['ema'] if len(result_df) > 1 else current_ema
        trend = "bullish" if current_price > current_ema else "bearish"
        ema_slope = "rising" if current_ema > prev_ema else "falling"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "price_type": price_type,
            "current_price": round(current_price, 4),
            "current_ema": round(current_ema, 4),
            "trend": trend,
            "ema_slope": ema_slope,
            "price_vs_ema_pct": round((current_price / current_ema - 1) * 100, 2),
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": [price_type, "timestamp"],
                "sql_template": f"SELECT timestamp, {price_type} FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"EMA calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_wma_impl(
    symbol: str,
    period: int = 20,
    interval: str = "60min",
    price_type: str = "close",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Weighted Moving Average (WMA).

    The WMA assigns a heavier weighting to more current data points
    since they are more relevant than data points in the distant past.

    Formula: WMA = (P1*1 + P2*2 + ... + Pn*n) / (1 + 2 + ... + n)

    Required Data Columns:
        - close (or specified price_type): Price data
        - timestamp: Time of the candle

    Args:
        symbol: Stock ticker or ignored if is_fx=True
        period: Number of periods for WMA (default: 20)
        interval: Time interval
        price_type: Price to use
        is_fx: Whether this is a forex pair
        from_currency: Source currency for FX
        to_currency: Target currency for FX

    Returns:
        Dictionary with WMA values and trend analysis
    """
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < period:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['wma'] = calculate_wma(df[price_type], period)
        result_df = df[['timestamp', price_type, 'wma']].dropna()

        current_price = result_df.iloc[-1][price_type]
        current_wma = result_df.iloc[-1]['wma']
        trend = "bullish" if current_price > current_wma else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_price": round(current_price, 4),
            "current_wma": round(current_wma, 4),
            "trend": trend,
            "price_vs_wma_pct": round((current_price / current_wma - 1) * 100, 2),
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": [price_type, "timestamp"],
                "sql_template": f"SELECT timestamp, {price_type} FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"WMA calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_dema_impl(
    symbol: str,
    period: int = 20,
    interval: str = "60min",
    price_type: str = "close",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Double Exponential Moving Average (DEMA).

    DEMA is designed to reduce the lag inherent in traditional moving averages.
    It's calculated by combining two EMAs to create a more responsive indicator.

    Formula: DEMA = 2 * EMA(n) - EMA(EMA(n))

    Required Data Columns:
        - close (or specified price_type): Price data
        - timestamp: Time of the candle

    Args:
        symbol: Stock ticker or ignored if is_fx=True
        period: Number of periods for DEMA (default: 20)
        interval: Time interval
        price_type: Price to use
        is_fx: Whether this is a forex pair
        from_currency: Source currency for FX
        to_currency: Target currency for FX

    Returns:
        Dictionary with DEMA values and trend analysis
    """
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < period * 2:
            return {"status": "error", "message": f"Insufficient data for {symbol}. DEMA needs {period*2} points"}

        df['dema'] = calculate_dema(df[price_type], period)
        df['ema'] = calculate_ema(df[price_type], period)
        result_df = df[['timestamp', price_type, 'ema', 'dema']].dropna()

        current_price = result_df.iloc[-1][price_type]
        current_dema = result_df.iloc[-1]['dema']
        current_ema = result_df.iloc[-1]['ema']
        trend = "bullish" if current_price > current_dema else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_price": round(current_price, 4),
            "current_dema": round(current_dema, 4),
            "current_ema": round(current_ema, 4),
            "trend": trend,
            "lag_reduction": round(abs(current_price - current_dema) - abs(current_price - current_ema), 4),
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": [price_type, "timestamp"],
                "sql_template": f"SELECT timestamp, {price_type} FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"DEMA calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_tema_impl(
    symbol: str,
    period: int = 20,
    interval: str = "60min",
    price_type: str = "close",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Triple Exponential Moving Average (TEMA).

    TEMA is designed to smooth price fluctuations with even less lag than DEMA.
    It uses three EMAs to achieve this smoothing effect.

    Formula: TEMA = 3 * EMA(n) - 3 * EMA(EMA(n)) + EMA(EMA(EMA(n)))

    Required Data Columns:
        - close (or specified price_type): Price data
        - timestamp: Time of the candle

    Args:
        symbol: Stock ticker or ignored if is_fx=True
        period: Number of periods for TEMA (default: 20)
        interval: Time interval
        price_type: Price to use
        is_fx: Whether this is a forex pair
        from_currency: Source currency for FX
        to_currency: Target currency for FX

    Returns:
        Dictionary with TEMA values and trend analysis
    """
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < period * 3:
            return {"status": "error", "message": f"Insufficient data for {symbol}. TEMA needs {period*3} points"}

        df['tema'] = calculate_tema(df[price_type], period)
        df['ema'] = calculate_ema(df[price_type], period)
        df['dema'] = calculate_dema(df[price_type], period)
        result_df = df[['timestamp', price_type, 'ema', 'dema', 'tema']].dropna()

        current_price = result_df.iloc[-1][price_type]
        current_tema = result_df.iloc[-1]['tema']
        trend = "bullish" if current_price > current_tema else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_price": round(current_price, 4),
            "current_tema": round(current_tema, 4),
            "current_dema": round(result_df.iloc[-1]['dema'], 4),
            "current_ema": round(result_df.iloc[-1]['ema'], 4),
            "trend": trend,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": [price_type, "timestamp"],
                "sql_template": f"SELECT timestamp, {price_type} FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"TEMA calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_ma_compare_impl(
    symbol: str,
    periods: List[int] = None,
    interval: str = "60min",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Compare multiple moving averages for a symbol.

    This tool calculates and compares SMA, EMA, WMA, DEMA, and TEMA
    for given periods, useful for identifying trends and crossovers.

    Args:
        symbol: Stock ticker or ignored if is_fx=True
        periods: List of periods to calculate (default: [10, 20, 50])
        interval: Time interval
        is_fx: Whether this is a forex pair
        from_currency: Source currency for FX
        to_currency: Target currency for FX

    Returns:
        Dictionary with all moving average comparisons
    """
    periods = periods or [10, 20, 50]

    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty:
            return {"status": "error", "message": f"No data for {symbol}"}

        max_period = max(periods) * 3  # For TEMA
        if len(df) < max_period:
            return {"status": "error", "message": f"Insufficient data. Need {max_period} points"}

        results = {}
        current_price = df['close'].iloc[-1]

        for period in periods:
            results[f'sma_{period}'] = round(calculate_sma(df['close'], period).iloc[-1], 4)
            results[f'ema_{period}'] = round(calculate_ema(df['close'], period).iloc[-1], 4)
            results[f'wma_{period}'] = round(calculate_wma(df['close'], period).iloc[-1], 4)
            if len(df) >= period * 2:
                results[f'dema_{period}'] = round(calculate_dema(df['close'], period).iloc[-1], 4)
            if len(df) >= period * 3:
                results[f'tema_{period}'] = round(calculate_tema(df['close'], period).iloc[-1], 4)

        # Determine overall trend
        short_ma = results.get(f'ema_{min(periods)}', current_price)
        long_ma = results.get(f'ema_{max(periods)}', current_price)
        overall_trend = "bullish" if short_ma > long_ma else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "current_price": round(current_price, 4),
            "periods_analyzed": periods,
            "moving_averages": results,
            "overall_trend": overall_trend,
            "short_term_ma": short_ma,
            "long_term_ma": long_ma,
            "ma_spread_pct": round((short_ma / long_ma - 1) * 100, 2),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"MA comparison error: {e}")
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server: FastMCP) -> List[str]:
    """Register all moving average tools with the MCP server."""

    @mcp_server.tool()
    async def av_sma(
        symbol: str,
        period: int = 20,
        interval: str = "60min",
        price_type: str = "close",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Simple Moving Average (SMA) for a stock or forex pair.

        The SMA is the most basic moving average, calculated by taking the
        arithmetic mean of prices over a specified period. It's widely used
        to identify trend direction and potential support/resistance levels.

        USAGE:
        - For stocks: av_sma(symbol="AAPL", period=20)
        - For forex: av_sma(is_fx=True, from_currency="EUR", to_currency="USD", period=20)

        TRADING SIGNALS:
        - Price above SMA: Bullish trend
        - Price below SMA: Bearish trend
        - Price crossing above SMA: Potential buy signal
        - Price crossing below SMA: Potential sell signal

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'MSFT', 'GOOGL')
            period: Number of periods for SMA calculation (default: 20)
            interval: Time interval - '1min', '5min', '15min', '30min', '60min'
            price_type: Price to use - 'open', 'high', 'low', 'close'
            is_fx: Set to True for forex pairs
            from_currency: Source currency for FX (e.g., 'EUR', 'GBP')
            to_currency: Target currency for FX (e.g., 'USD', 'JPY')

        Returns:
            Dictionary with SMA values, trend analysis, and data points

        Examples:
            >>> av_sma(symbol="AAPL", period=20)
            >>> av_sma(symbol="MSFT", period=50, interval="30min")
            >>> av_sma(is_fx=True, from_currency="EUR", to_currency="USD", period=20)
        """
        return await av_sma_impl(
            symbol, period, interval, price_type, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_ema(
        symbol: str,
        period: int = 20,
        interval: str = "60min",
        price_type: str = "close",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Exponential Moving Average (EMA) for a stock or forex pair.

        The EMA gives more weight to recent prices, making it more responsive
        to new information than the SMA. It's preferred for short-term trading.

        USAGE:
        - For stocks: av_ema(symbol="AAPL", period=12)
        - For forex: av_ema(is_fx=True, from_currency="EUR", to_currency="USD")

        TRADING SIGNALS:
        - EMA slope up + price above EMA: Strong bullish
        - EMA slope down + price below EMA: Strong bearish
        - Common periods: 12, 26 (for MACD), 9, 21, 50, 200

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            period: Number of periods for EMA (default: 20)
            interval: Time interval
            price_type: Price to use
            is_fx: Set to True for forex pairs
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with EMA values, trend, and slope analysis

        Examples:
            >>> av_ema(symbol="TSLA", period=12)
            >>> av_ema(symbol="GOOGL", period=26, interval="15min")
        """
        return await av_ema_impl(
            symbol, period, interval, price_type, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_wma(
        symbol: str,
        period: int = 20,
        interval: str = "60min",
        price_type: str = "close",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Weighted Moving Average (WMA) for a stock or forex pair.

        The WMA assigns linearly increasing weights to more recent prices,
        providing a balance between SMA and EMA responsiveness.

        Args:
            symbol: Stock ticker
            period: Number of periods for WMA (default: 20)
            interval: Time interval
            price_type: Price to use
            is_fx: Set to True for forex pairs
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with WMA values and trend analysis
        """
        return await av_wma_impl(
            symbol, period, interval, price_type, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_dema(
        symbol: str,
        period: int = 20,
        interval: str = "60min",
        price_type: str = "close",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Double Exponential Moving Average (DEMA).

        DEMA reduces the lag of traditional moving averages by using
        a combination of two EMAs. It's more responsive to price changes.

        Formula: DEMA = 2 * EMA(n) - EMA(EMA(n))

        Args:
            symbol: Stock ticker
            period: Number of periods (default: 20)
            interval: Time interval
            price_type: Price to use
            is_fx: Set to True for forex
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with DEMA values and lag reduction analysis
        """
        return await av_dema_impl(
            symbol, period, interval, price_type, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_tema(
        symbol: str,
        period: int = 20,
        interval: str = "60min",
        price_type: str = "close",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Triple Exponential Moving Average (TEMA).

        TEMA provides the least lag of all moving averages by using
        three EMAs. It's excellent for identifying trend changes quickly.

        Formula: TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

        Args:
            symbol: Stock ticker
            period: Number of periods (default: 20)
            interval: Time interval
            price_type: Price to use
            is_fx: Set to True for forex
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with TEMA values and comparison to EMA/DEMA
        """
        return await av_tema_impl(
            symbol, period, interval, price_type, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_ma_compare(
        symbol: str,
        periods: str = "10,20,50",
        interval: str = "60min",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Compare all types of moving averages for multiple periods.

        This tool calculates SMA, EMA, WMA, DEMA, and TEMA for multiple
        periods, providing a comprehensive moving average analysis.

        USAGE:
        - Identify trend direction across timeframes
        - Spot potential crossover signals
        - Compare MA responsiveness

        Args:
            symbol: Stock ticker
            periods: Comma-separated periods (default: "10,20,50")
            interval: Time interval
            is_fx: Set to True for forex
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with all MA types for all periods

        Example:
            >>> av_ma_compare(symbol="AAPL", periods="9,21,50,200")
        """
        period_list = [int(p.strip()) for p in periods.split(',')]
        return await av_ma_compare_impl(
            symbol, period_list, interval, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    return [
        'av_sma',
        'av_ema',
        'av_wma',
        'av_dema',
        'av_tema',
        'av_ma_compare'
    ]
