"""
Momentum Indicator Tools for Algorithmic Trading.

This module provides momentum-based technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Stochastic RSI
- ROC (Rate of Change)
- MOM (Momentum)
- PPO (Percentage Price Oscillator)

These indicators help identify the speed and magnitude of price movements.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

from .data_provider import get_ohlcv, get_fx_ohlcv
from .moving_averages import calculate_ema

logger = logging.getLogger(__name__)


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple:
    """
    Calculate MACD, Signal Line, and Histogram.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line)
    Histogram = MACD Line - Signal Line
    """
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> tuple:
    """
    Calculate Stochastic Oscillator (%K and %D).

    %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def calculate_stochrsi(data: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
    """
    Calculate Stochastic RSI.

    StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)
    """
    rsi = calculate_rsi(data, rsi_period)
    lowest_rsi = rsi.rolling(window=stoch_period).min()
    highest_rsi = rsi.rolling(window=stoch_period).max()

    stoch_rsi = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi)
    return stoch_rsi * 100


def calculate_roc(data: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Rate of Change.

    ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
    """
    return ((data - data.shift(period)) / data.shift(period)) * 100


def calculate_momentum(data: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Momentum.

    Momentum = Current Price - Price n periods ago
    """
    return data - data.shift(period)


def calculate_ppo(data: pd.Series, fast_period: int = 12, slow_period: int = 26) -> pd.Series:
    """
    Calculate Percentage Price Oscillator.

    PPO = ((EMA(fast) - EMA(slow)) / EMA(slow)) * 100
    """
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    return ((ema_fast - ema_slow) / ema_slow) * 100


async def av_rsi_impl(
    symbol: str,
    period: int = 14,
    interval: str = "60min",
    overbought: float = 70,
    oversold: float = 30,
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions. Values above 70 typically indicate
    overbought, while values below 30 indicate oversold.

    Required Data Columns:
        - close: Closing price
        - timestamp: Time of the candle

    SQL Equivalent:
        SELECT timestamp, close,
               -- RSI calculation requires gains/losses tracking
               RSI_14 as rsi
        FROM technical_indicators
        WHERE symbol = '{symbol}'

    Args:
        symbol: Stock ticker or ignored if is_fx=True
        period: RSI period (default: 14)
        interval: Time interval
        overbought: Overbought threshold (default: 70)
        oversold: Oversold threshold (default: 30)
        is_fx: Whether this is a forex pair
        from_currency: Source currency for FX
        to_currency: Target currency for FX

    Returns:
        Dictionary with RSI values and trading signals
    """
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

        df['rsi'] = calculate_rsi(df['close'], period)
        result_df = df[['timestamp', 'close', 'rsi']].dropna()

        current_rsi = result_df.iloc[-1]['rsi']
        prev_rsi = result_df.iloc[-2]['rsi'] if len(result_df) > 1 else current_rsi

        # Determine signal
        if current_rsi >= overbought:
            signal = "overbought"
            action = "potential_sell"
        elif current_rsi <= oversold:
            signal = "oversold"
            action = "potential_buy"
        else:
            signal = "neutral"
            action = "hold"

        # Divergence detection (simplified)
        rsi_slope = current_rsi - prev_rsi
        price_change = result_df.iloc[-1]['close'] - result_df.iloc[-2]['close'] if len(result_df) > 1 else 0

        divergence = None
        if price_change > 0 and rsi_slope < 0:
            divergence = "bearish_divergence"
        elif price_change < 0 and rsi_slope > 0:
            divergence = "bullish_divergence"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_rsi": round(current_rsi, 2),
            "previous_rsi": round(prev_rsi, 2),
            "rsi_change": round(current_rsi - prev_rsi, 2),
            "signal": signal,
            "action": action,
            "divergence": divergence,
            "overbought_level": overbought,
            "oversold_level": oversold,
            "distance_to_overbought": round(overbought - current_rsi, 2),
            "distance_to_oversold": round(current_rsi - oversold, 2),
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_macd_impl(
    symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    interval: str = "60min",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Moving Average Convergence Divergence (MACD).

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.

    Components:
    - MACD Line: EMA(12) - EMA(26)
    - Signal Line: EMA(9) of MACD Line
    - Histogram: MACD Line - Signal Line

    Required Data Columns:
        - close: Closing price
        - timestamp: Time of the candle

    SQL Equivalent:
        SELECT timestamp, close, macd_line, signal_line, histogram
        FROM technical_indicators
        WHERE symbol = '{symbol}'

    Trading Signals:
    - MACD crosses above Signal: Bullish
    - MACD crosses below Signal: Bearish
    - Histogram expanding: Trend strengthening
    - Histogram contracting: Trend weakening
    """
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

        df['macd'], df['signal'], df['histogram'] = calculate_macd(
            df['close'], fast_period, slow_period, signal_period
        )
        result_df = df[['timestamp', 'close', 'macd', 'signal', 'histogram']].dropna()

        current = result_df.iloc[-1]
        previous = result_df.iloc[-2] if len(result_df) > 1 else current

        # Detect crossover
        crossover = None
        if previous['macd'] <= previous['signal'] and current['macd'] > current['signal']:
            crossover = "bullish_crossover"
        elif previous['macd'] >= previous['signal'] and current['macd'] < current['signal']:
            crossover = "bearish_crossover"

        # Determine trend
        trend = "bullish" if current['macd'] > current['signal'] else "bearish"
        histogram_trend = "expanding" if abs(current['histogram']) > abs(previous['histogram']) else "contracting"

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period
            },
            "interval": interval,
            "current_macd": round(current['macd'], 4),
            "current_signal": round(current['signal'], 4),
            "current_histogram": round(current['histogram'], 4),
            "previous_histogram": round(previous['histogram'], 4),
            "trend": trend,
            "crossover": crossover,
            "histogram_trend": histogram_trend,
            "macd_above_zero": current['macd'] > 0,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_stochastic_impl(
    symbol: str,
    k_period: int = 14,
    d_period: int = 3,
    interval: str = "60min",
    overbought: float = 80,
    oversold: float = 20,
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Stochastic Oscillator.

    The Stochastic Oscillator compares a closing price to its price range
    over a given time period. It's used to generate overbought and oversold
    trading signals.

    Components:
    - %K (Fast): Main stochastic line
    - %D (Slow): Signal line (SMA of %K)

    Required Data Columns:
        - high: High price
        - low: Low price
        - close: Closing price
        - timestamp: Time of the candle

    SQL Equivalent:
        SELECT timestamp, high, low, close, stoch_k, stoch_d
        FROM technical_indicators
        WHERE symbol = '{symbol}'
    """
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < k_period + d_period:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['stoch_k'], df['stoch_d'] = calculate_stochastic(
            df['high'], df['low'], df['close'], k_period, d_period
        )
        result_df = df[['timestamp', 'close', 'stoch_k', 'stoch_d']].dropna()

        current = result_df.iloc[-1]
        previous = result_df.iloc[-2] if len(result_df) > 1 else current

        # Determine signal
        if current['stoch_k'] >= overbought:
            signal = "overbought"
        elif current['stoch_k'] <= oversold:
            signal = "oversold"
        else:
            signal = "neutral"

        # Detect crossover
        crossover = None
        if previous['stoch_k'] <= previous['stoch_d'] and current['stoch_k'] > current['stoch_d']:
            crossover = "bullish_crossover"
        elif previous['stoch_k'] >= previous['stoch_d'] and current['stoch_k'] < current['stoch_d']:
            crossover = "bearish_crossover"

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"k_period": k_period, "d_period": d_period},
            "interval": interval,
            "current_k": round(current['stoch_k'], 2),
            "current_d": round(current['stoch_d'], 2),
            "signal": signal,
            "crossover": crossover,
            "overbought_level": overbought,
            "oversold_level": oversold,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"Stochastic calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_stochrsi_impl(
    symbol: str,
    rsi_period: int = 14,
    stoch_period: int = 14,
    interval: str = "60min",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Stochastic RSI.

    StochRSI applies the Stochastic oscillator formula to RSI values
    instead of price values. It's more sensitive than regular RSI.

    Formula: StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)

    Required Data Columns:
        - close: Closing price
        - timestamp: Time of the candle
    """
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < rsi_period + stoch_period:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['rsi'] = calculate_rsi(df['close'], rsi_period)
        df['stoch_rsi'] = calculate_stochrsi(df['close'], rsi_period, stoch_period)
        result_df = df[['timestamp', 'close', 'rsi', 'stoch_rsi']].dropna()

        current_stoch_rsi = result_df.iloc[-1]['stoch_rsi']

        signal = "neutral"
        if current_stoch_rsi >= 80:
            signal = "overbought"
        elif current_stoch_rsi <= 20:
            signal = "oversold"

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"rsi_period": rsi_period, "stoch_period": stoch_period},
            "interval": interval,
            "current_stoch_rsi": round(current_stoch_rsi, 2),
            "current_rsi": round(result_df.iloc[-1]['rsi'], 2),
            "signal": signal,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"StochRSI calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_roc_impl(
    symbol: str,
    period: int = 10,
    interval: str = "60min",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Rate of Change (ROC).

    ROC measures the percentage change in price between the current price
    and the price n periods ago. It's a momentum oscillator.

    Formula: ROC = ((Current - Previous) / Previous) * 100

    Required Data Columns:
        - close: Closing price
        - timestamp: Time of the candle
    """
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

        df['roc'] = calculate_roc(df['close'], period)
        result_df = df[['timestamp', 'close', 'roc']].dropna()

        current_roc = result_df.iloc[-1]['roc']
        momentum = "bullish" if current_roc > 0 else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_roc": round(current_roc, 2),
            "momentum": momentum,
            "interpretation": f"Price is {abs(current_roc):.2f}% {'higher' if current_roc > 0 else 'lower'} than {period} periods ago",
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"ROC calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_momentum_impl(
    symbol: str,
    period: int = 10,
    interval: str = "60min",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Momentum (MOM).

    Momentum measures the rate of change in price by comparing current
    price to price n periods ago (absolute difference, not percentage).

    Formula: Momentum = Current Price - Price n periods ago

    Required Data Columns:
        - close: Closing price
        - timestamp: Time of the candle
    """
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

        df['momentum'] = calculate_momentum(df['close'], period)
        result_df = df[['timestamp', 'close', 'momentum']].dropna()

        current_momentum = result_df.iloc[-1]['momentum']
        trend = "bullish" if current_momentum > 0 else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_momentum": round(current_momentum, 4),
            "trend": trend,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"Momentum calculation error: {e}")
        return {"status": "error", "message": str(e)}


async def av_ppo_impl(
    symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    interval: str = "60min",
    is_fx: bool = False,
    from_currency: str = None,
    to_currency: str = None
) -> Dict[str, Any]:
    """
    Calculate Percentage Price Oscillator (PPO).

    PPO is similar to MACD but expressed as a percentage, making it easier
    to compare securities with different price levels.

    Formula: PPO = ((EMA(fast) - EMA(slow)) / EMA(slow)) * 100

    Required Data Columns:
        - close: Closing price
        - timestamp: Time of the candle
    """
    try:
        if is_fx:
            if not from_currency or not to_currency:
                return {"status": "error", "message": "FX requires from_currency and to_currency"}
            df = await get_fx_ohlcv(from_currency, to_currency, interval)
            symbol = f"{from_currency}/{to_currency}"
        else:
            df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < slow_period:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['ppo'] = calculate_ppo(df['close'], fast_period, slow_period)
        df['ppo_signal'] = calculate_ema(df['ppo'], 9)
        result_df = df[['timestamp', 'close', 'ppo', 'ppo_signal']].dropna()

        current = result_df.iloc[-1]
        trend = "bullish" if current['ppo'] > 0 else "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "parameters": {"fast_period": fast_period, "slow_period": slow_period},
            "interval": interval,
            "current_ppo": round(current['ppo'], 2),
            "current_signal": round(current['ppo_signal'], 2),
            "trend": trend,
            "data_points": len(result_df),
            "data": result_df.tail(50).to_dict('records'),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }

    except Exception as e:
        logger.error(f"PPO calculation error: {e}")
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server: FastMCP) -> List[str]:
    """Register all momentum indicator tools with the MCP server."""

    @mcp_server.tool()
    async def av_rsi(
        symbol: str,
        period: int = 14,
        interval: str = "60min",
        overbought: float = 70,
        oversold: float = 30,
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Relative Strength Index (RSI).

        RSI is a momentum oscillator that measures the speed and magnitude
        of price movements. It ranges from 0 to 100.

        TRADING SIGNALS:
        - RSI > 70: Overbought (potential sell)
        - RSI < 30: Oversold (potential buy)
        - RSI divergence: Price/RSI disagreement signals reversal

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            period: RSI period (default: 14)
            interval: Time interval
            overbought: Overbought threshold (default: 70)
            oversold: Oversold threshold (default: 30)
            is_fx: Set to True for forex pairs
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with RSI values, signals, and divergence detection

        Examples:
            >>> av_rsi(symbol="AAPL", period=14)
            >>> av_rsi(symbol="TSLA", period=9, overbought=80, oversold=20)
        """
        return await av_rsi_impl(
            symbol, period, interval, overbought, oversold, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_macd(
        symbol: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        interval: str = "60min",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Moving Average Convergence Divergence (MACD).

        MACD is a trend-following momentum indicator showing the relationship
        between two EMAs. It's one of the most popular technical indicators.

        COMPONENTS:
        - MACD Line: EMA(12) - EMA(26)
        - Signal Line: EMA(9) of MACD
        - Histogram: MACD - Signal

        TRADING SIGNALS:
        - MACD crosses above Signal: Bullish crossover (buy)
        - MACD crosses below Signal: Bearish crossover (sell)
        - Histogram expanding: Trend strengthening
        - Zero line crossover: Trend change confirmation

        Args:
            symbol: Stock ticker
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            interval: Time interval
            is_fx: Set to True for forex
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with MACD line, signal, histogram, and crossover signals
        """
        return await av_macd_impl(
            symbol, fast_period, slow_period, signal_period, interval, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_stochastic(
        symbol: str,
        k_period: int = 14,
        d_period: int = 3,
        interval: str = "60min",
        overbought: float = 80,
        oversold: float = 20,
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Stochastic Oscillator.

        The Stochastic Oscillator compares closing price to its price range.
        It's useful for identifying overbought/oversold conditions and
        potential reversals.

        COMPONENTS:
        - %K (Fast line): Main stochastic
        - %D (Slow line): SMA of %K

        TRADING SIGNALS:
        - %K > 80: Overbought
        - %K < 20: Oversold
        - %K crosses above %D: Bullish
        - %K crosses below %D: Bearish

        Args:
            symbol: Stock ticker
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            interval: Time interval
            overbought: Overbought level (default: 80)
            oversold: Oversold level (default: 20)
            is_fx: Set to True for forex
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with %K, %D values and crossover signals
        """
        return await av_stochastic_impl(
            symbol, k_period, d_period, interval, overbought, oversold, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_stochrsi(
        symbol: str,
        rsi_period: int = 14,
        stoch_period: int = 14,
        interval: str = "60min",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Stochastic RSI.

        StochRSI applies Stochastic formula to RSI values instead of prices.
        It's more sensitive than regular RSI and oscillates between 0-100.

        This indicator is useful for identifying extreme conditions and
        potential reversals when regular RSI gives neutral readings.

        Args:
            symbol: Stock ticker
            rsi_period: RSI calculation period (default: 14)
            stoch_period: Stochastic calculation period (default: 14)
            interval: Time interval
            is_fx: Set to True for forex
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with StochRSI and underlying RSI values
        """
        return await av_stochrsi_impl(
            symbol, rsi_period, stoch_period, interval, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_roc(
        symbol: str,
        period: int = 10,
        interval: str = "60min",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Rate of Change (ROC).

        ROC measures percentage change between current price and price
        n periods ago. It's useful for identifying momentum and divergences.

        INTERPRETATION:
        - Positive ROC: Price increasing (bullish)
        - Negative ROC: Price decreasing (bearish)
        - ROC crossing zero: Trend change

        Args:
            symbol: Stock ticker
            period: Lookback period (default: 10)
            interval: Time interval
            is_fx: Set to True for forex
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with ROC values and momentum interpretation
        """
        return await av_roc_impl(
            symbol, period, interval, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_momentum(
        symbol: str,
        period: int = 10,
        interval: str = "60min",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Momentum indicator.

        Momentum is the absolute price difference between current price
        and price n periods ago. Unlike ROC, it's not a percentage.

        INTERPRETATION:
        - Positive momentum: Upward trend
        - Negative momentum: Downward trend
        - Momentum divergence: Potential reversal

        Args:
            symbol: Stock ticker
            period: Lookback period (default: 10)
            interval: Time interval
            is_fx: Set to True for forex
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with momentum values
        """
        return await av_momentum_impl(
            symbol, period, interval, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    @mcp_server.tool()
    async def av_ppo(
        symbol: str,
        fast_period: int = 12,
        slow_period: int = 26,
        interval: str = "60min",
        is_fx: bool = False,
        from_currency: str = "",
        to_currency: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate Percentage Price Oscillator (PPO).

        PPO is similar to MACD but expressed as a percentage.
        This makes it easier to compare across different securities.

        Formula: PPO = ((EMA_fast - EMA_slow) / EMA_slow) * 100

        TRADING SIGNALS:
        - PPO > 0: Bullish momentum
        - PPO < 0: Bearish momentum
        - PPO crossing signal line: Entry/exit signals

        Args:
            symbol: Stock ticker
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            interval: Time interval
            is_fx: Set to True for forex
            from_currency: Source currency for FX
            to_currency: Target currency for FX

        Returns:
            Dictionary with PPO values and trend analysis
        """
        return await av_ppo_impl(
            symbol, fast_period, slow_period, interval, is_fx,
            from_currency if from_currency else None,
            to_currency if to_currency else None
        )

    return [
        'av_rsi',
        'av_macd',
        'av_stochastic',
        'av_stochrsi',
        'av_roc',
        'av_momentum',
        'av_ppo'
    ]
