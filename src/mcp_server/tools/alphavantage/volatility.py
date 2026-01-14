"""
Volatility Indicator Tools for Algorithmic Trading.

This module provides volatility-based indicators:
- Bollinger Bands
- ATR (Average True Range)
- NATR (Normalized ATR)
- Keltner Channels
- Standard Deviation
"""

import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

from .data_provider import get_ohlcv, get_fx_ohlcv
from .moving_averages import calculate_sma, calculate_ema

logger = logging.getLogger(__name__)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands."""
    middle = calculate_sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    bandwidth = (upper - lower) / middle * 100
    percent_b = (close - lower) / (upper - lower)
    return upper, middle, lower, bandwidth, percent_b


def calculate_keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, multiplier: float = 2.0) -> tuple:
    """Calculate Keltner Channels."""
    middle = calculate_ema(close, period)
    atr = calculate_atr(high, low, close, period)
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)
    return upper, middle, lower


async def av_bbands_impl(
    symbol: str, period: int = 20, std_dev: float = 2.0, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate Bollinger Bands."""
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

        df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bandwidth'], df['percent_b'] = \
            calculate_bollinger_bands(df['close'], period, std_dev)

        result_df = df[['timestamp', 'close', 'bb_upper', 'bb_middle', 'bb_lower', 'bandwidth', 'percent_b']].dropna()
        current = result_df.iloc[-1]

        # Determine position relative to bands
        if current['close'] >= current['bb_upper']:
            position = "above_upper"
            signal = "overbought"
        elif current['close'] <= current['bb_lower']:
            position = "below_lower"
            signal = "oversold"
        else:
            position = "inside_bands"
            signal = "neutral"

        # Squeeze detection
        avg_bandwidth = result_df['bandwidth'].tail(50).mean()
        squeeze = current['bandwidth'] < avg_bandwidth * 0.75

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"period": period, "std_dev": std_dev},
            "interval": interval,
            "current_price": round(current['close'], 4),
            "upper_band": round(current['bb_upper'], 4),
            "middle_band": round(current['bb_middle'], 4),
            "lower_band": round(current['bb_lower'], 4),
            "bandwidth": round(current['bandwidth'], 2),
            "percent_b": round(current['percent_b'], 2),
            "position": position,
            "signal": signal,
            "squeeze_detected": squeeze,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_atr_impl(
    symbol: str, period: int = 14, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate Average True Range."""
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < period + 1:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period)
        df['natr'] = (df['atr'] / df['close']) * 100

        result_df = df[['timestamp', 'close', 'atr', 'natr']].dropna()
        current = result_df.iloc[-1]

        # Volatility classification
        avg_atr = result_df['atr'].mean()
        if current['atr'] > avg_atr * 1.5:
            volatility = "high"
        elif current['atr'] < avg_atr * 0.5:
            volatility = "low"
        else:
            volatility = "normal"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_atr": round(current['atr'], 4),
            "normalized_atr": round(current['natr'], 2),
            "average_atr": round(avg_atr, 4),
            "volatility": volatility,
            "suggested_stop_loss_distance": round(current['atr'] * 2, 4),
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_keltner_impl(
    symbol: str, period: int = 20, multiplier: float = 2.0, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate Keltner Channels."""
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

        df['kc_upper'], df['kc_middle'], df['kc_lower'] = calculate_keltner_channels(
            df['high'], df['low'], df['close'], period, multiplier
        )
        result_df = df[['timestamp', 'close', 'kc_upper', 'kc_middle', 'kc_lower']].dropna()
        current = result_df.iloc[-1]

        if current['close'] >= current['kc_upper']:
            position = "above_upper"
            signal = "overbought"
        elif current['close'] <= current['kc_lower']:
            position = "below_lower"
            signal = "oversold"
        else:
            position = "inside_channels"
            signal = "neutral"

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"period": period, "multiplier": multiplier},
            "interval": interval,
            "current_price": round(current['close'], 4),
            "upper_channel": round(current['kc_upper'], 4),
            "middle_channel": round(current['kc_middle'], 4),
            "lower_channel": round(current['kc_lower'], 4),
            "position": position,
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


async def av_stddev_impl(
    symbol: str, period: int = 20, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate Standard Deviation."""
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

        df['stddev'] = df['close'].rolling(window=period).std()
        df['stddev_pct'] = (df['stddev'] / df['close']) * 100

        result_df = df[['timestamp', 'close', 'stddev', 'stddev_pct']].dropna()
        current = result_df.iloc[-1]

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_stddev": round(current['stddev'], 4),
            "stddev_percent": round(current['stddev_pct'], 2),
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
    """Register all volatility indicator tools."""

    @mcp_server.tool()
    async def av_bbands(
        symbol: str, period: int = 20, std_dev: float = 2.0, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Bollinger Bands.

        Bollinger Bands consist of a middle band (SMA) with upper and lower bands
        at standard deviation distances. They adapt to volatility.

        COMPONENTS:
        - Upper Band: SMA + (std_dev * StdDev)
        - Middle Band: SMA(period)
        - Lower Band: SMA - (std_dev * StdDev)
        - Bandwidth: (Upper - Lower) / Middle * 100
        - %B: (Price - Lower) / (Upper - Lower)

        SIGNALS:
        - Price at upper band: Overbought
        - Price at lower band: Oversold
        - Band squeeze: Low volatility, breakout imminent
        """
        return await av_bbands_impl(symbol, period, std_dev, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_atr(
        symbol: str, period: int = 14, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Average True Range (ATR).

        ATR measures market volatility by analyzing the complete range of
        price movement. It's commonly used for position sizing and stop-loss placement.

        USES:
        - Stop-loss placement (typically 2x ATR)
        - Position sizing based on volatility
        - Identifying volatility regime changes
        """
        return await av_atr_impl(symbol, period, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_keltner(
        symbol: str, period: int = 20, multiplier: float = 2.0, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Keltner Channels.

        Keltner Channels are similar to Bollinger Bands but use ATR instead
        of standard deviation for the channel width.

        COMPONENTS:
        - Upper: EMA + (multiplier * ATR)
        - Middle: EMA(period)
        - Lower: EMA - (multiplier * ATR)

        Often used with Bollinger Bands for the "squeeze" setup.
        """
        return await av_keltner_impl(symbol, period, multiplier, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_stddev(
        symbol: str, period: int = 20, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Standard Deviation of prices.

        Standard deviation measures the dispersion of prices around the mean.
        Higher values indicate higher volatility.
        """
        return await av_stddev_impl(symbol, period, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    return ['av_bbands', 'av_atr', 'av_keltner', 'av_stddev']
