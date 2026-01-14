"""
Trend Indicator Tools for Algorithmic Trading.

This module provides trend-following indicators:
- ADX (Average Directional Index)
- AROON (Aroon Indicator and Oscillator)
- SAR (Parabolic Stop and Reverse)
- SuperTrend
"""

import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

from .data_provider import get_ohlcv, get_fx_ohlcv

logger = logging.getLogger(__name__)


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    """
    Calculate Average Directional Index (ADX), +DI, and -DI.
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Smooth with EMA
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()

    return adx, plus_di, minus_di


def calculate_aroon(high: pd.Series, low: pd.Series, period: int = 25) -> tuple:
    """Calculate Aroon Up, Aroon Down, and Aroon Oscillator."""
    aroon_up = pd.Series(index=high.index, dtype=float)
    aroon_down = pd.Series(index=low.index, dtype=float)

    for i in range(period, len(high)):
        high_window = high.iloc[i-period:i+1]
        low_window = low.iloc[i-period:i+1]

        days_since_high = period - high_window.argmax()
        days_since_low = period - low_window.argmin()

        aroon_up.iloc[i] = ((period - days_since_high) / period) * 100
        aroon_down.iloc[i] = ((period - days_since_low) / period) * 100

    aroon_osc = aroon_up - aroon_down
    return aroon_up, aroon_down, aroon_osc


def calculate_parabolic_sar(high: pd.Series, low: pd.Series, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Calculate Parabolic SAR."""
    length = len(high)
    sar = pd.Series(index=high.index, dtype=float)
    trend = pd.Series(index=high.index, dtype=int)
    af = af_start
    ep = low.iloc[0]
    sar.iloc[0] = high.iloc[0]
    trend.iloc[0] = -1

    for i in range(1, length):
        if trend.iloc[i-1] == 1:  # Uptrend
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])

            if low.iloc[i] < sar.iloc[i]:
                trend.iloc[i] = -1
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                trend.iloc[i] = 1
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_start, af_max)
        else:  # Downtrend
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])

            if high.iloc[i] > sar.iloc[i]:
                trend.iloc[i] = 1
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                trend.iloc[i] = -1
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_start, af_max)

    return sar, trend


def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> tuple:
    """Calculate SuperTrend indicator."""
    hl2 = (high + low) / 2

    # Calculate ATR
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Calculate bands
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(close)):
        if close.iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] < supertrend.iloc[i-1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]

            if direction.iloc[i] == 1 and lower_band.iloc[i] > supertrend.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
            elif direction.iloc[i] == -1 and upper_band.iloc[i] < supertrend.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]

    return supertrend, direction


async def av_adx_impl(
    symbol: str, period: int = 14, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate Average Directional Index (ADX)."""
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < period * 2:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'], period)
        result_df = df[['timestamp', 'close', 'adx', 'plus_di', 'minus_di']].dropna()

        current = result_df.iloc[-1]

        # Interpret ADX
        if current['adx'] < 20:
            trend_strength = "weak/absent"
        elif current['adx'] < 40:
            trend_strength = "strong"
        elif current['adx'] < 60:
            trend_strength = "very_strong"
        else:
            trend_strength = "extremely_strong"

        trend_direction = "bullish" if current['plus_di'] > current['minus_di'] else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_adx": round(current['adx'], 2),
            "current_plus_di": round(current['plus_di'], 2),
            "current_minus_di": round(current['minus_di'], 2),
            "trend_strength": trend_strength,
            "trend_direction": trend_direction,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_aroon_impl(
    symbol: str, period: int = 25, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate Aroon Indicator."""
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

        df['aroon_up'], df['aroon_down'], df['aroon_osc'] = calculate_aroon(df['high'], df['low'], period)
        result_df = df[['timestamp', 'close', 'aroon_up', 'aroon_down', 'aroon_osc']].dropna()

        current = result_df.iloc[-1]
        trend = "bullish" if current['aroon_osc'] > 0 else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "aroon_up": round(current['aroon_up'], 2),
            "aroon_down": round(current['aroon_down'], 2),
            "aroon_oscillator": round(current['aroon_osc'], 2),
            "trend": trend,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "timestamp"],
                "sql_template": "SELECT timestamp, high, low FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_sar_impl(
    symbol: str, acceleration: float = 0.02, maximum: float = 0.2, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate Parabolic SAR."""
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < 5:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['sar'], df['trend'] = calculate_parabolic_sar(df['high'], df['low'], acceleration, maximum)
        result_df = df[['timestamp', 'close', 'sar', 'trend']].dropna()

        current = result_df.iloc[-1]
        previous = result_df.iloc[-2] if len(result_df) > 1 else current

        trend = "bullish" if current['trend'] == 1 else "bearish"
        reversal = None
        if previous['trend'] != current['trend']:
            reversal = "bullish_reversal" if current['trend'] == 1 else "bearish_reversal"

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"acceleration": acceleration, "maximum": maximum},
            "interval": interval,
            "current_sar": round(current['sar'], 4),
            "current_price": round(current['close'], 4),
            "trend": trend,
            "reversal": reversal,
            "stop_loss": round(current['sar'], 4),
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "timestamp"],
                "sql_template": "SELECT timestamp, high, low FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_supertrend_impl(
    symbol: str, period: int = 10, multiplier: float = 3.0, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Calculate SuperTrend indicator."""
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

        df['supertrend'], df['direction'] = calculate_supertrend(df['high'], df['low'], df['close'], period, multiplier)
        result_df = df[['timestamp', 'close', 'supertrend', 'direction']].dropna()

        current = result_df.iloc[-1]
        trend = "bullish" if current['direction'] == 1 else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"period": period, "multiplier": multiplier},
            "interval": interval,
            "supertrend": round(current['supertrend'], 4),
            "current_price": round(current['close'], 4),
            "trend": trend,
            "stop_loss": round(current['supertrend'], 4),
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server: FastMCP) -> List[str]:
    """Register all trend indicator tools."""

    @mcp_server.tool()
    async def av_adx(
        symbol: str, period: int = 14, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Average Directional Index (ADX) with +DI and -DI.

        ADX measures trend strength (0-100) regardless of direction.
        +DI and -DI show directional movement.

        INTERPRETATION:
        - ADX < 20: Weak/absent trend (range-bound)
        - ADX 20-40: Strong trend
        - ADX > 40: Very strong trend
        - +DI > -DI: Bullish
        - -DI > +DI: Bearish
        """
        return await av_adx_impl(symbol, period, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_aroon(
        symbol: str, period: int = 25, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Aroon Indicator (Up, Down, and Oscillator).

        Aroon measures time since the highest high and lowest low.
        Oscillator = Aroon Up - Aroon Down.

        INTERPRETATION:
        - Aroon Up > 70, Aroon Down < 30: Strong uptrend
        - Aroon Down > 70, Aroon Up < 30: Strong downtrend
        - Both low: Consolidation
        """
        return await av_aroon_impl(symbol, period, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_sar(
        symbol: str, acceleration: float = 0.02, maximum: float = 0.2, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Parabolic SAR (Stop and Reverse).

        Parabolic SAR provides potential entry/exit points by trailing price.
        It flips from above to below price on trend changes.

        USE CASES:
        - Trailing stop-loss levels
        - Trend direction identification
        - Entry/exit signals on SAR flip
        """
        return await av_sar_impl(symbol, acceleration, maximum, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_supertrend(
        symbol: str, period: int = 10, multiplier: float = 3.0, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate SuperTrend indicator.

        SuperTrend is a trend-following indicator based on ATR.
        It flips between support and resistance based on price action.

        TRADING:
        - Price above SuperTrend: Bullish (buy zone)
        - Price below SuperTrend: Bearish (sell zone)
        - SuperTrend can be used as dynamic stop-loss
        """
        return await av_supertrend_impl(symbol, period, multiplier, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    return ['av_adx', 'av_aroon', 'av_sar', 'av_supertrend']
