"""
Volume Indicator Tools for Algorithmic Trading.

This module provides volume-based indicators:
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- A/D Line (Accumulation/Distribution)
- MFI (Money Flow Index)
- ADOSC (Chaikin A/D Oscillator)
"""

import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

from .data_provider import get_ohlcv
from .moving_averages import calculate_ema

logger = logging.getLogger(__name__)


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]

    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return obv


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()
    return cumulative_tp_vol / cumulative_vol


def calculate_ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Accumulation/Distribution Line."""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    ad = (clv * volume).cumsum()
    return ad


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    positive_flow = pd.Series(0, index=close.index, dtype=float)
    negative_flow = pd.Series(0, index=close.index, dtype=float)

    for i in range(1, len(close)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = raw_money_flow.iloc[i]
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            negative_flow.iloc[i] = raw_money_flow.iloc[i]

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    money_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi


def calculate_adosc(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                    fast_period: int = 3, slow_period: int = 10) -> pd.Series:
    """Calculate Chaikin A/D Oscillator."""
    ad = calculate_ad_line(high, low, close, volume)
    fast_ema = calculate_ema(ad, fast_period)
    slow_ema = calculate_ema(ad, slow_period)
    return fast_ema - slow_ema


async def av_obv_impl(symbol: str, interval: str = "60min") -> Dict[str, Any]:
    """Calculate On-Balance Volume."""
    try:
        df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < 2:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['obv'] = calculate_obv(df['close'], df['volume'])
        df['obv_ema'] = calculate_ema(df['obv'], 20)

        result_df = df[['timestamp', 'close', 'volume', 'obv', 'obv_ema']].dropna()
        current = result_df.iloc[-1]
        prev = result_df.iloc[-2] if len(result_df) > 1 else current

        obv_trend = "rising" if current['obv'] > prev['obv'] else "falling"
        price_trend = "rising" if current['close'] > prev['close'] else "falling"

        divergence = None
        if obv_trend != price_trend:
            if obv_trend == "rising" and price_trend == "falling":
                divergence = "bullish_divergence"
            else:
                divergence = "bearish_divergence"

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "current_obv": int(current['obv']),
            "obv_ema_20": int(current['obv_ema']),
            "obv_trend": obv_trend,
            "price_trend": price_trend,
            "divergence": divergence,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["close", "volume", "timestamp"],
                "sql_template": "SELECT timestamp, close, volume FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_vwap_impl(symbol: str, interval: str = "60min") -> Dict[str, Any]:
    """Calculate Volume Weighted Average Price."""
    try:
        df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < 2:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])

        result_df = df[['timestamp', 'close', 'volume', 'vwap']].dropna()
        current = result_df.iloc[-1]

        position = "above" if current['close'] > current['vwap'] else "below"
        bias = "bullish" if position == "above" else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "current_price": round(current['close'], 4),
            "current_vwap": round(current['vwap'], 4),
            "price_vs_vwap_pct": round((current['close'] / current['vwap'] - 1) * 100, 2),
            "position": position,
            "bias": bias,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "volume", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close, volume FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_ad_impl(symbol: str, interval: str = "60min") -> Dict[str, Any]:
    """Calculate Accumulation/Distribution Line."""
    try:
        df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < 2:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['ad'] = calculate_ad_line(df['high'], df['low'], df['close'], df['volume'])
        df['ad_ema'] = calculate_ema(df['ad'], 20)

        result_df = df[['timestamp', 'close', 'ad', 'ad_ema']].dropna()
        current = result_df.iloc[-1]
        prev = result_df.iloc[-2] if len(result_df) > 1 else current

        ad_trend = "accumulation" if current['ad'] > prev['ad'] else "distribution"

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "current_ad": round(current['ad'], 0),
            "ad_ema_20": round(current['ad_ema'], 0),
            "ad_trend": ad_trend,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "volume", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close, volume FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_mfi_impl(symbol: str, period: int = 14, interval: str = "60min") -> Dict[str, Any]:
    """Calculate Money Flow Index."""
    try:
        df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < period + 1:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['mfi'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'], period)

        result_df = df[['timestamp', 'close', 'volume', 'mfi']].dropna()
        current = result_df.iloc[-1]

        if current['mfi'] >= 80:
            signal = "overbought"
        elif current['mfi'] <= 20:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_mfi": round(current['mfi'], 2),
            "signal": signal,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "volume", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close, volume FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_adosc_impl(symbol: str, fast_period: int = 3, slow_period: int = 10, interval: str = "60min") -> Dict[str, Any]:
    """Calculate Chaikin A/D Oscillator."""
    try:
        df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < slow_period + 1:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['adosc'] = calculate_adosc(df['high'], df['low'], df['close'], df['volume'], fast_period, slow_period)

        result_df = df[['timestamp', 'close', 'adosc']].dropna()
        current = result_df.iloc[-1]

        trend = "bullish" if current['adosc'] > 0 else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"fast_period": fast_period, "slow_period": slow_period},
            "interval": interval,
            "current_adosc": round(current['adosc'], 0),
            "trend": trend,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "volume", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close, volume FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server: FastMCP) -> List[str]:
    """Register all volume indicator tools."""

    @mcp_server.tool()
    async def av_obv(symbol: str, interval: str = "60min") -> Dict[str, Any]:
        """
        Calculate On-Balance Volume (OBV).

        OBV relates volume to price change. It's a cumulative indicator
        that adds volume on up days and subtracts on down days.

        SIGNALS:
        - Rising OBV: Buying pressure
        - Falling OBV: Selling pressure
        - OBV divergence from price: Potential reversal
        """
        return await av_obv_impl(symbol, interval)

    @mcp_server.tool()
    async def av_vwap(symbol: str, interval: str = "60min") -> Dict[str, Any]:
        """
        Calculate Volume Weighted Average Price (VWAP).

        VWAP is the average price weighted by volume, commonly used
        as a benchmark for institutional trading.

        TRADING:
        - Price above VWAP: Bullish bias
        - Price below VWAP: Bearish bias
        - VWAP acts as dynamic support/resistance
        """
        return await av_vwap_impl(symbol, interval)

    @mcp_server.tool()
    async def av_ad(symbol: str, interval: str = "60min") -> Dict[str, Any]:
        """
        Calculate Accumulation/Distribution Line.

        A/D Line measures cumulative money flow based on the close's
        position within the day's range.

        SIGNALS:
        - Rising A/D: Accumulation (buying)
        - Falling A/D: Distribution (selling)
        - A/D divergence: Trend reversal warning
        """
        return await av_ad_impl(symbol, interval)

    @mcp_server.tool()
    async def av_mfi(symbol: str, period: int = 14, interval: str = "60min") -> Dict[str, Any]:
        """
        Calculate Money Flow Index (MFI).

        MFI is a volume-weighted RSI. It measures buying/selling
        pressure using both price and volume.

        SIGNALS:
        - MFI > 80: Overbought
        - MFI < 20: Oversold
        - MFI divergence: Reversal signal
        """
        return await av_mfi_impl(symbol, period, interval)

    @mcp_server.tool()
    async def av_adosc(symbol: str, fast_period: int = 3, slow_period: int = 10, interval: str = "60min") -> Dict[str, Any]:
        """
        Calculate Chaikin A/D Oscillator.

        ADOSC is the difference between fast and slow EMAs of the
        A/D Line. It measures momentum of accumulation/distribution.

        SIGNALS:
        - Positive ADOSC: Buying pressure increasing
        - Negative ADOSC: Selling pressure increasing
        - Zero crossover: Momentum shift
        """
        return await av_adosc_impl(symbol, fast_period, slow_period, interval)

    return ['av_obv', 'av_vwap', 'av_ad', 'av_mfi', 'av_adosc']
