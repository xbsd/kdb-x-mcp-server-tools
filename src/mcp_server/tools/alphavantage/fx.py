"""
Forex-Specific Tools for Algorithmic Trading.

This module provides forex-specific indicators and tools:
- FX Quote (current exchange rate)
- FX Technical Analysis (comprehensive)
- FX Pivot Points
- FX Correlation Analysis
"""

import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

from .data_provider import get_fx_ohlcv, get_fx_rate
from .moving_averages import calculate_sma, calculate_ema
from .momentum import calculate_rsi, calculate_macd
from .volatility import calculate_atr

logger = logging.getLogger(__name__)


def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculate classic pivot points."""
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)

    return {
        "pivot": round(pivot, 5),
        "r1": round(r1, 5),
        "r2": round(r2, 5),
        "r3": round(r3, 5),
        "s1": round(s1, 5),
        "s2": round(s2, 5),
        "s3": round(s3, 5)
    }


async def av_fx_quote_impl(from_currency: str, to_currency: str) -> Dict[str, Any]:
    """Get current FX exchange rate."""
    try:
        rate_data = await get_fx_rate(from_currency, to_currency)

        return {
            "status": "success",
            "pair": f"{from_currency.upper()}/{to_currency.upper()}",
            "exchange_rate": rate_data['exchange_rate'],
            "bid": rate_data.get('bid_price'),
            "ask": rate_data.get('ask_price'),
            "spread": round(rate_data.get('ask_price', 0) - rate_data.get('bid_price', 0), 6) if rate_data.get('ask_price') else None,
            "last_refreshed": rate_data.get('last_refreshed'),
            "metadata": {
                "required_columns": ["from_currency", "to_currency", "exchange_rate", "timestamp"],
                "sql_template": "SELECT * FROM fx_rates WHERE from_currency = ? AND to_currency = ? ORDER BY timestamp DESC LIMIT 1"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_fx_technical_impl(
    from_currency: str, to_currency: str, interval: str = "60min"
) -> Dict[str, Any]:
    """Comprehensive FX technical analysis."""
    try:
        df = await get_fx_ohlcv(from_currency, to_currency, interval)

        if df.empty or len(df) < 50:
            return {"status": "error", "message": "Insufficient FX data"}

        symbol = f"{from_currency.upper()}/{to_currency.upper()}"

        # Calculate indicators
        df['sma_20'] = calculate_sma(df['close'], 20)
        df['sma_50'] = calculate_sma(df['close'], 50)
        df['ema_12'] = calculate_ema(df['close'], 12)
        df['ema_26'] = calculate_ema(df['close'], 26)
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'], _ = calculate_macd(df['close'])
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)

        df = df.dropna()
        current = df.iloc[-1]

        # Calculate pivot points from previous day high/low/close
        pivots = calculate_pivot_points(
            df['high'].tail(24).max(),
            df['low'].tail(24).min(),
            current['close']
        )

        # Generate signals
        signals = {
            "trend": "bullish" if current['sma_20'] > current['sma_50'] else "bearish",
            "momentum": "overbought" if current['rsi'] > 70 else ("oversold" if current['rsi'] < 30 else "neutral"),
            "macd": "bullish" if current['macd'] > current['macd_signal'] else "bearish"
        }

        bullish = sum(1 for v in signals.values() if v in ['bullish'])
        bearish = sum(1 for v in signals.values() if v in ['bearish'])

        overall = "bullish" if bullish > bearish else ("bearish" if bearish > bullish else "neutral")

        return {
            "status": "success",
            "pair": symbol,
            "interval": interval,
            "current_rate": round(current['close'], 5),
            "indicators": {
                "sma_20": round(current['sma_20'], 5),
                "sma_50": round(current['sma_50'], 5),
                "ema_12": round(current['ema_12'], 5),
                "ema_26": round(current['ema_26'], 5),
                "rsi": round(current['rsi'], 2),
                "macd": round(current['macd'], 6),
                "atr": round(current['atr'], 5),
                "atr_pips": round(current['atr'] * 10000, 1)
            },
            "pivot_points": pivots,
            "signals": signals,
            "overall_signal": overall,
            "data_points": len(df),
            "metadata": {
                "required_columns": ["timestamp", "open", "high", "low", "close"],
                "sql_template": "SELECT timestamp, open, high, low, close FROM fx_ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_fx_pivot_impl(
    from_currency: str, to_currency: str, interval: str = "60min"
) -> Dict[str, Any]:
    """Calculate FX pivot points and levels."""
    try:
        df = await get_fx_ohlcv(from_currency, to_currency, interval)

        if df.empty or len(df) < 24:
            return {"status": "error", "message": "Insufficient FX data"}

        symbol = f"{from_currency.upper()}/{to_currency.upper()}"

        # Use last 24 hours for daily pivot calculation
        recent = df.tail(24)
        high = recent['high'].max()
        low = recent['low'].min()
        close = df.iloc[-1]['close']

        pivots = calculate_pivot_points(high, low, close)

        current = df.iloc[-1]['close']

        # Determine position relative to pivots
        if current >= pivots['r2']:
            position = "above_r2"
        elif current >= pivots['r1']:
            position = "between_r1_r2"
        elif current >= pivots['pivot']:
            position = "between_pivot_r1"
        elif current >= pivots['s1']:
            position = "between_s1_pivot"
        elif current >= pivots['s2']:
            position = "between_s2_s1"
        else:
            position = "below_s2"

        # Calculate distances
        distances = {
            "to_r1": round((pivots['r1'] - current) * 10000, 1),
            "to_s1": round((current - pivots['s1']) * 10000, 1),
            "to_pivot": round((pivots['pivot'] - current) * 10000, 1)
        }

        return {
            "status": "success",
            "pair": symbol,
            "current_rate": round(current, 5),
            "pivot_points": pivots,
            "position": position,
            "distances_pips": distances,
            "daily_range": {
                "high": round(high, 5),
                "low": round(low, 5),
                "range_pips": round((high - low) * 10000, 1)
            },
            "metadata": {
                "required_columns": ["timestamp", "high", "low", "close"],
                "sql_template": "SELECT timestamp, high, low, close FROM fx_ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_fx_volatility_impl(
    from_currency: str, to_currency: str, interval: str = "60min"
) -> Dict[str, Any]:
    """Analyze FX pair volatility."""
    try:
        df = await get_fx_ohlcv(from_currency, to_currency, interval, output_size="full")

        if df.empty or len(df) < 50:
            return {"status": "error", "message": "Insufficient FX data"}

        symbol = f"{from_currency.upper()}/{to_currency.upper()}"

        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
        df['returns'] = df['close'].pct_change()
        df = df.dropna()

        current = df.iloc[-1]

        # Calculate volatility metrics
        hist_vol = df['returns'].std() * np.sqrt(252 * 24)  # Annualized hourly vol
        recent_vol = df['returns'].tail(24).std() * np.sqrt(252 * 24)

        # ATR in pips
        atr_pips = current['atr'] * 10000

        # Volatility percentile
        atr_percentile = (df['atr'] < current['atr']).mean() * 100

        return {
            "status": "success",
            "pair": symbol,
            "interval": interval,
            "current_rate": round(current['close'], 5),
            "atr": round(current['atr'], 5),
            "atr_pips": round(atr_pips, 1),
            "historical_volatility": round(hist_vol * 100, 2),
            "recent_volatility": round(recent_vol * 100, 2),
            "volatility_ratio": round(recent_vol / hist_vol, 2) if hist_vol > 0 else None,
            "atr_percentile": round(atr_percentile, 0),
            "volatility_regime": "high" if atr_percentile > 70 else ("low" if atr_percentile < 30 else "normal"),
            "suggested_stop_pips": round(atr_pips * 1.5, 1),
            "data_points": len(df),
            "metadata": {
                "required_columns": ["timestamp", "high", "low", "close"],
                "sql_template": "SELECT timestamp, high, low, close FROM fx_ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_fx_strength_impl(pairs: str = "EURUSD,GBPUSD,USDJPY,AUDUSD") -> Dict[str, Any]:
    """Analyze relative currency strength across pairs."""
    try:
        pair_list = [p.strip().upper() for p in pairs.split(',')]
        strength_data = {}

        for pair in pair_list:
            # Parse pair
            if len(pair) == 6:
                from_curr = pair[:3]
                to_curr = pair[3:]
            elif '/' in pair:
                from_curr, to_curr = pair.split('/')
            else:
                continue

            try:
                df = await get_fx_ohlcv(from_curr, to_curr, "60min")
                if not df.empty and len(df) >= 20:
                    current = df.iloc[-1]['close']
                    prev_20 = df.iloc[-20]['close']
                    change_pct = ((current / prev_20) - 1) * 100

                    df['rsi'] = calculate_rsi(df['close'], 14)
                    rsi = df['rsi'].iloc[-1] if not df['rsi'].isna().all() else 50

                    strength_data[pair] = {
                        "rate": round(current, 5),
                        "change_20_periods": round(change_pct, 2),
                        "rsi": round(rsi, 1),
                        "trend": "bullish" if change_pct > 0 else "bearish"
                    }
            except Exception:
                continue

        return {
            "status": "success",
            "pairs_analyzed": list(strength_data.keys()),
            "data": strength_data,
            "strongest": max(strength_data.items(), key=lambda x: x[1]['change_20_periods'])[0] if strength_data else None,
            "weakest": min(strength_data.items(), key=lambda x: x[1]['change_20_periods'])[0] if strength_data else None,
            "metadata": {
                "required_columns": ["timestamp", "close"],
                "sql_template": "SELECT timestamp, close FROM fx_ohlcv WHERE symbol IN (?) ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server: FastMCP) -> List[str]:
    """Register all FX-specific tools."""

    @mcp_server.tool()
    async def av_fx_quote(from_currency: str, to_currency: str) -> Dict[str, Any]:
        """
        Get current FX exchange rate.

        Returns real-time exchange rate, bid, ask, and spread.

        Examples:
            av_fx_quote(from_currency="EUR", to_currency="USD")
            av_fx_quote(from_currency="GBP", to_currency="JPY")
        """
        return await av_fx_quote_impl(from_currency, to_currency)

    @mcp_server.tool()
    async def av_fx_technical(
        from_currency: str, to_currency: str, interval: str = "60min"
    ) -> Dict[str, Any]:
        """
        Comprehensive FX technical analysis.

        Calculates multiple indicators (SMA, EMA, RSI, MACD, ATR)
        and pivot points for a currency pair.
        """
        return await av_fx_technical_impl(from_currency, to_currency, interval)

    @mcp_server.tool()
    async def av_fx_pivot(
        from_currency: str, to_currency: str, interval: str = "60min"
    ) -> Dict[str, Any]:
        """
        Calculate FX pivot points and support/resistance levels.

        Classic pivot point calculation with R1-R3 and S1-S3 levels.
        Shows current price position and distances in pips.
        """
        return await av_fx_pivot_impl(from_currency, to_currency, interval)

    @mcp_server.tool()
    async def av_fx_volatility(
        from_currency: str, to_currency: str, interval: str = "60min"
    ) -> Dict[str, Any]:
        """
        Analyze FX pair volatility.

        Calculates ATR in pips, historical and recent volatility,
        volatility percentile, and suggested stop-loss distance.
        """
        return await av_fx_volatility_impl(from_currency, to_currency, interval)

    @mcp_server.tool()
    async def av_fx_strength(pairs: str = "EURUSD,GBPUSD,USDJPY,AUDUSD") -> Dict[str, Any]:
        """
        Analyze relative currency strength across multiple pairs.

        Compares momentum and trend direction for multiple FX pairs
        to identify strongest and weakest currencies.

        Args:
            pairs: Comma-separated currency pairs (e.g., "EURUSD,GBPUSD")
        """
        return await av_fx_strength_impl(pairs)

    return ['av_fx_quote', 'av_fx_technical', 'av_fx_pivot', 'av_fx_volatility', 'av_fx_strength']
