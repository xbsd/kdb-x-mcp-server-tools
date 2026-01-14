"""
Signal Detection Tools for Algorithmic Trading.

This module provides signal generation and pattern detection:
- Golden/Death Cross (SMA crossovers)
- MACD Crossover Signals
- RSI Divergence Detection
- Bollinger Band Breakouts
- Multi-indicator Signal Generator
- Trend Strength Analysis
"""

import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

from .data_provider import get_ohlcv, get_fx_ohlcv
from .moving_averages import calculate_sma, calculate_ema
from .momentum import calculate_rsi, calculate_macd, calculate_stochastic
from .trend import calculate_adx
from .volatility import calculate_bollinger_bands

logger = logging.getLogger(__name__)


async def av_golden_cross_impl(
    symbol: str, fast_period: int = 50, slow_period: int = 200, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Detect Golden Cross and Death Cross patterns."""
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval, output_size="full")
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval, output_size="full")

        if df.empty or len(df) < slow_period:
            return {"status": "error", "message": f"Insufficient data for {symbol}. Need {slow_period} points."}

        df['sma_fast'] = calculate_sma(df['close'], fast_period)
        df['sma_slow'] = calculate_sma(df['close'], slow_period)
        df = df.dropna()

        if len(df) < 2:
            return {"status": "error", "message": "Not enough data after calculation"}

        current = df.iloc[-1]
        previous = df.iloc[-2]

        # Detect crossover
        crossover = None
        if previous['sma_fast'] <= previous['sma_slow'] and current['sma_fast'] > current['sma_slow']:
            crossover = "golden_cross"
        elif previous['sma_fast'] >= previous['sma_slow'] and current['sma_fast'] < current['sma_slow']:
            crossover = "death_cross"

        # Current position
        position = "bullish" if current['sma_fast'] > current['sma_slow'] else "bearish"

        # Distance to crossover
        spread = current['sma_fast'] - current['sma_slow']
        spread_pct = (spread / current['sma_slow']) * 100

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"fast_period": fast_period, "slow_period": slow_period},
            "interval": interval,
            "current_price": round(current['close'], 4),
            "sma_fast": round(current['sma_fast'], 4),
            "sma_slow": round(current['sma_slow'], 4),
            "crossover_detected": crossover,
            "current_position": position,
            "spread": round(spread, 4),
            "spread_percent": round(spread_pct, 2),
            "signal": crossover if crossover else position,
            "data_points": len(df),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_macd_crossover_impl(
    symbol: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Detect MACD crossover signals."""
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < slow_period + signal_period:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['macd'], df['signal'], df['histogram'] = calculate_macd(df['close'], fast_period, slow_period, signal_period)
        df = df.dropna()

        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current

        # Detect crossovers
        crossover = None
        if previous['macd'] <= previous['signal'] and current['macd'] > current['signal']:
            crossover = "bullish_crossover"
        elif previous['macd'] >= previous['signal'] and current['macd'] < current['signal']:
            crossover = "bearish_crossover"

        # Zero line crossover
        zero_crossover = None
        if previous['macd'] <= 0 and current['macd'] > 0:
            zero_crossover = "bullish_zero_cross"
        elif previous['macd'] >= 0 and current['macd'] < 0:
            zero_crossover = "bearish_zero_cross"

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "macd": round(current['macd'], 4),
            "signal_line": round(current['signal'], 4),
            "histogram": round(current['histogram'], 4),
            "crossover_detected": crossover,
            "zero_crossover": zero_crossover,
            "histogram_trend": "expanding" if abs(current['histogram']) > abs(previous['histogram']) else "contracting",
            "overall_signal": crossover if crossover else ("bullish" if current['macd'] > current['signal'] else "bearish"),
            "data_points": len(df),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_rsi_divergence_impl(
    symbol: str, period: int = 14, lookback: int = 20, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Detect RSI divergence patterns."""
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < period + lookback:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['rsi'] = calculate_rsi(df['close'], period)
        df = df.dropna()

        # Check for divergence in the lookback window
        recent = df.tail(lookback)
        price_trend = "up" if recent['close'].iloc[-1] > recent['close'].iloc[0] else "down"
        rsi_trend = "up" if recent['rsi'].iloc[-1] > recent['rsi'].iloc[0] else "down"

        divergence = None
        if price_trend == "up" and rsi_trend == "down":
            divergence = "bearish_divergence"
        elif price_trend == "down" and rsi_trend == "up":
            divergence = "bullish_divergence"

        current = df.iloc[-1]

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"period": period, "lookback": lookback},
            "interval": interval,
            "current_rsi": round(current['rsi'], 2),
            "current_price": round(current['close'], 4),
            "price_trend": price_trend,
            "rsi_trend": rsi_trend,
            "divergence_detected": divergence,
            "signal_strength": "strong" if divergence else "none",
            "data_points": len(df),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_bb_breakout_impl(
    symbol: str, period: int = 20, std_dev: float = 2.0, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Detect Bollinger Band breakouts."""
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
        df = df.dropna()

        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current

        # Detect breakouts
        breakout = None
        if previous['close'] < previous['bb_upper'] and current['close'] >= current['bb_upper']:
            breakout = "upper_breakout"
        elif previous['close'] > previous['bb_lower'] and current['close'] <= current['bb_lower']:
            breakout = "lower_breakout"

        # Squeeze detection
        avg_bandwidth = df['bandwidth'].mean()
        squeeze = current['bandwidth'] < avg_bandwidth * 0.75

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "current_price": round(current['close'], 4),
            "upper_band": round(current['bb_upper'], 4),
            "lower_band": round(current['bb_lower'], 4),
            "percent_b": round(current['percent_b'], 2),
            "breakout_detected": breakout,
            "squeeze_active": squeeze,
            "signal": breakout if breakout else ("squeeze_pending" if squeeze else "inside_bands"),
            "data_points": len(df),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_multi_signal_impl(
    symbol: str, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Generate multi-indicator signal analysis."""
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < 50:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        # Calculate all indicators
        df['sma_20'] = calculate_sma(df['close'], 20)
        df['sma_50'] = calculate_sma(df['close'], 50)
        df['ema_12'] = calculate_ema(df['close'], 12)
        df['ema_26'] = calculate_ema(df['close'], 26)
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
        df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])

        df = df.dropna()
        current = df.iloc[-1]

        # Calculate signals
        signals = {
            "sma_trend": "bullish" if current['sma_20'] > current['sma_50'] else "bearish",
            "ema_trend": "bullish" if current['ema_12'] > current['ema_26'] else "bearish",
            "price_vs_sma": "bullish" if current['close'] > current['sma_20'] else "bearish",
            "rsi_signal": "overbought" if current['rsi'] > 70 else ("oversold" if current['rsi'] < 30 else "neutral"),
            "macd_signal": "bullish" if current['macd'] > current['macd_signal'] else "bearish",
            "stoch_signal": "overbought" if current['stoch_k'] > 80 else ("oversold" if current['stoch_k'] < 20 else "neutral"),
        }

        # Count bullish/bearish signals
        bullish_count = sum(1 for v in signals.values() if v == "bullish")
        bearish_count = sum(1 for v in signals.values() if v == "bearish")

        if bullish_count >= 4:
            overall = "strong_bullish"
        elif bullish_count >= 3:
            overall = "bullish"
        elif bearish_count >= 4:
            overall = "strong_bearish"
        elif bearish_count >= 3:
            overall = "bearish"
        else:
            overall = "neutral"

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "current_price": round(current['close'], 4),
            "indicators": {
                "sma_20": round(current['sma_20'], 4),
                "sma_50": round(current['sma_50'], 4),
                "rsi": round(current['rsi'], 2),
                "macd": round(current['macd'], 4),
                "stoch_k": round(current['stoch_k'], 2)
            },
            "signals": signals,
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
            "overall_signal": overall,
            "confidence": round(max(bullish_count, bearish_count) / 6 * 100, 0),
            "data_points": len(df),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_trend_strength_impl(
    symbol: str, interval: str = "60min",
    is_fx: bool = False, from_currency: str = None, to_currency: str = None
) -> Dict[str, Any]:
    """Analyze overall trend strength using multiple indicators."""
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < 50:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'])
        df['sma_20'] = calculate_sma(df['close'], 20)
        df['sma_50'] = calculate_sma(df['close'], 50)

        df = df.dropna()
        current = df.iloc[-1]

        # Trend direction
        direction = "bullish" if current['plus_di'] > current['minus_di'] else "bearish"

        # ADX strength
        if current['adx'] < 20:
            strength = "weak"
            action = "avoid_trending_strategies"
        elif current['adx'] < 40:
            strength = "moderate"
            action = "consider_trend_following"
        elif current['adx'] < 60:
            strength = "strong"
            action = "follow_the_trend"
        else:
            strength = "very_strong"
            action = "strong_trend_in_place"

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "current_price": round(current['close'], 4),
            "adx": round(current['adx'], 2),
            "plus_di": round(current['plus_di'], 2),
            "minus_di": round(current['minus_di'], 2),
            "trend_direction": direction,
            "trend_strength": strength,
            "recommended_action": action,
            "ma_alignment": "aligned" if (current['close'] > current['sma_20'] > current['sma_50']) or
                                        (current['close'] < current['sma_20'] < current['sma_50']) else "misaligned",
            "data_points": len(df),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server: FastMCP) -> List[str]:
    """Register all signal detection tools."""

    @mcp_server.tool()
    async def av_golden_cross(
        symbol: str, fast_period: int = 50, slow_period: int = 200, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Detect Golden Cross and Death Cross patterns.

        Golden Cross: Fast SMA crosses above Slow SMA (bullish)
        Death Cross: Fast SMA crosses below Slow SMA (bearish)

        These are major trend reversal signals used for medium-term trading.
        """
        return await av_golden_cross_impl(symbol, fast_period, slow_period, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_macd_crossover(
        symbol: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Detect MACD crossover signals.

        Identifies both MACD/Signal line crossovers and zero line crossovers.
        Also tracks histogram momentum.
        """
        return await av_macd_crossover_impl(symbol, fast_period, slow_period, signal_period, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_rsi_divergence(
        symbol: str, period: int = 14, lookback: int = 20, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Detect RSI divergence patterns.

        Bullish Divergence: Price makes lower lows, RSI makes higher lows
        Bearish Divergence: Price makes higher highs, RSI makes lower highs
        """
        return await av_rsi_divergence_impl(symbol, period, lookback, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_bb_breakout(
        symbol: str, period: int = 20, std_dev: float = 2.0, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Detect Bollinger Band breakouts and squeezes.

        Upper Breakout: Price breaks above upper band
        Lower Breakout: Price breaks below lower band
        Squeeze: Low bandwidth indicating imminent breakout
        """
        return await av_bb_breakout_impl(symbol, period, std_dev, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_multi_signal(
        symbol: str, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Generate comprehensive multi-indicator signal analysis.

        Combines: SMA trend, EMA trend, RSI, MACD, and Stochastic
        Provides overall signal strength and confidence score.
        """
        return await av_multi_signal_impl(symbol, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    @mcp_server.tool()
    async def av_trend_strength(
        symbol: str, interval: str = "60min",
        is_fx: bool = False, from_currency: str = "", to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze overall trend strength using ADX and moving averages.

        Provides trend direction, strength classification, and
        recommended trading action based on current market conditions.
        """
        return await av_trend_strength_impl(symbol, interval, is_fx,
            from_currency if from_currency else None, to_currency if to_currency else None)

    return ['av_golden_cross', 'av_macd_crossover', 'av_rsi_divergence', 'av_bb_breakout', 'av_multi_signal', 'av_trend_strength']
