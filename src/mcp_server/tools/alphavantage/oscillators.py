"""
Oscillator Tools for Algorithmic Trading.

This module provides oscillator indicators:
- CCI (Commodity Channel Index)
- Williams %R
- Ultimate Oscillator
- TRIX
"""

import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

from .data_provider import get_ohlcv, get_fx_ohlcv
from .moving_averages import calculate_ema

logger = logging.getLogger(__name__)


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad)
    return cci


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    return williams_r


def calculate_ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                                   period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
    """Calculate Ultimate Oscillator."""
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([high, close.shift(1)], axis=1).max(axis=1) - pd.concat([low, close.shift(1)], axis=1).min(axis=1)

    avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
    avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
    avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()

    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
    return uo


def calculate_trix(close: pd.Series, period: int = 15) -> pd.Series:
    """Calculate TRIX (Triple Exponential Average)."""
    ema1 = calculate_ema(close, period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)
    trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    return trix


async def av_cci_impl(
    symbol: str, period: int = 20, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate Commodity Channel Index."""
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

        df['cci'] = calculate_cci(df['high'], df['low'], df['close'], period)
        result_df = df[['timestamp', 'close', 'cci']].dropna()
        current = result_df.iloc[-1]

        if current['cci'] > 100:
            signal = "overbought"
        elif current['cci'] < -100:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_cci": round(current['cci'], 2),
            "signal": signal,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_willr_impl(
    symbol: str, period: int = 14, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate Williams %R."""
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

        df['willr'] = calculate_williams_r(df['high'], df['low'], df['close'], period)
        result_df = df[['timestamp', 'close', 'willr']].dropna()
        current = result_df.iloc[-1]

        if current['willr'] > -20:
            signal = "overbought"
        elif current['willr'] < -80:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_willr": round(current['willr'], 2),
            "signal": signal,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_ultosc_impl(
    symbol: str, period1: int = 7, period2: int = 14, period3: int = 28, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate Ultimate Oscillator."""
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < period3 + 1:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['ultosc'] = calculate_ultimate_oscillator(df['high'], df['low'], df['close'], period1, period2, period3)
        result_df = df[['timestamp', 'close', 'ultosc']].dropna()
        current = result_df.iloc[-1]

        if current['ultosc'] > 70:
            signal = "overbought"
        elif current['ultosc'] < 30:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"period1": period1, "period2": period2, "period3": period3},
            "interval": interval,
            "current_ultosc": round(current['ultosc'], 2),
            "signal": signal,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_trix_impl(
    symbol: str, period: int = 15, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate TRIX."""
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < period * 3:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['trix'] = calculate_trix(df['close'], period)
        df['trix_signal'] = calculate_ema(df['trix'], 9)
        result_df = df[['timestamp', 'close', 'trix', 'trix_signal']].dropna()
        current = result_df.iloc[-1]

        trend = "bullish" if current['trix'] > 0 else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_trix": round(current['trix'], 4),
            "trix_signal": round(current['trix_signal'], 4),
            "trend": trend,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server: FastMCP) -> List[str]:
    """Register all oscillator tools."""

    @mcp_server.tool()
    async def av_cci(
        symbol: str, period: int = 20, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Commodity Channel Index (CCI).

        CCI measures the current price level relative to an average price level.
        High positive readings indicate overbought, low negative readings oversold.

        SIGNALS:
        - CCI > 100: Overbought
        - CCI < -100: Oversold
        - CCI zero crossover: Trend change
        """
        return await av_cci_impl(symbol, period, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_willr(
        symbol: str, period: int = 14, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Williams %R.

        Williams %R is a momentum indicator ranging from 0 to -100.
        It shows where the close is relative to the high-low range.

        SIGNALS:
        - %R > -20: Overbought
        - %R < -80: Oversold
        """
        return await av_willr_impl(symbol, period, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_ultosc(
        symbol: str, period1: int = 7, period2: int = 14, period3: int = 28, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Ultimate Oscillator.

        Ultimate Oscillator uses three timeframes to reduce volatility
        and false signals common in single-timeframe oscillators.

        SIGNALS:
        - UO > 70: Overbought
        - UO < 30: Oversold
        - Divergence with price: Strong reversal signal
        """
        return await av_ultosc_impl(symbol, period1, period2, period3, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_trix(
        symbol: str, period: int = 15, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate TRIX (Triple Exponential Average).

        TRIX is a momentum oscillator that shows the percent rate of change
        of a triple exponentially smoothed moving average.

        SIGNALS:
        - TRIX > 0: Bullish momentum
        - TRIX < 0: Bearish momentum
        - TRIX crossing signal line: Entry/exit signals
        """
        return await av_trix_impl(symbol, period, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    return ['av_cci', 'av_willr', 'av_ultosc', 'av_trix']
