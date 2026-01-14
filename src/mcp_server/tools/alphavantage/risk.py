"""
Risk Management Tools for Algorithmic Trading.

This module provides risk analysis and position sizing tools:
- Volatility-based Position Sizing
- Maximum Drawdown Calculator
- Sharpe Ratio Calculator
- Sortino Ratio Calculator
- Risk/Reward Analyzer
"""

import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

from .data_provider import get_ohlcv, get_daily_ohlcv, get_fx_ohlcv
from .volatility import calculate_atr

logger = logging.getLogger(__name__)


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate percentage returns."""
    return prices.pct_change().dropna()


def calculate_max_drawdown(prices: pd.Series) -> tuple:
    """Calculate maximum drawdown and its duration."""
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    # Find peak before drawdown
    peak_idx = cumulative[:max_dd_idx].idxmax() if len(cumulative[:max_dd_idx]) > 0 else max_dd_idx

    return max_dd, peak_idx, max_dd_idx


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return 0
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
    return sharpe


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate Sortino ratio (uses downside deviation)."""
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0

    sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)
    return sortino


async def av_position_size_impl(
    symbol: str, account_size: float, risk_percent: float = 2.0,
    atr_multiplier: float = 2.0, interval: str = "60min"
) -> Dict[str, Any]:
    """Calculate position size based on ATR volatility."""
    try:
        df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < 15:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
        df = df.dropna()

        current = df.iloc[-1]
        current_price = current['close']
        current_atr = current['atr']

        # Calculate position size
        risk_amount = account_size * (risk_percent / 100)
        stop_distance = current_atr * atr_multiplier
        stop_price = current_price - stop_distance

        if stop_distance > 0:
            position_size = risk_amount / stop_distance
            shares = int(position_size)
            position_value = shares * current_price
        else:
            shares = 0
            position_value = 0

        return {
            "status": "success",
            "symbol": symbol,
            "account_size": account_size,
            "risk_percent": risk_percent,
            "risk_amount": round(risk_amount, 2),
            "current_price": round(current_price, 4),
            "atr": round(current_atr, 4),
            "stop_distance": round(stop_distance, 4),
            "stop_price": round(stop_price, 4),
            "recommended_shares": shares,
            "position_value": round(position_value, 2),
            "position_percent": round((position_value / account_size) * 100, 2),
            "risk_reward_ratio": round(atr_multiplier, 2),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_max_drawdown_impl(symbol: str, interval: str = "60min") -> Dict[str, Any]:
    """Calculate maximum drawdown."""
    try:
        df = await get_ohlcv(symbol, interval, output_size="full")

        if df.empty or len(df) < 10:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        max_dd, peak_date, trough_date = calculate_max_drawdown(df['close'])

        # Current drawdown
        cumulative = (1 + calculate_returns(df['close'])).cumprod()
        current_dd = (cumulative.iloc[-1] - cumulative.max()) / cumulative.max()

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "max_drawdown_percent": round(max_dd * 100, 2),
            "current_drawdown_percent": round(current_dd * 100, 2),
            "peak_date": str(peak_date),
            "trough_date": str(trough_date),
            "data_points": len(df),
            "status_assessment": "severe" if max_dd < -0.2 else ("moderate" if max_dd < -0.1 else "mild"),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_sharpe_impl(
    symbol: str, risk_free_rate: float = 0.02, interval: str = "60min"
) -> Dict[str, Any]:
    """Calculate Sharpe ratio."""
    try:
        df = await get_ohlcv(symbol, interval, output_size="full")

        if df.empty or len(df) < 30:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        returns = calculate_returns(df['close'])

        # Determine periods per year based on interval
        periods_map = {
            "1min": 252 * 390,
            "5min": 252 * 78,
            "15min": 252 * 26,
            "30min": 252 * 13,
            "60min": 252 * 6.5
        }
        periods = periods_map.get(interval, 252 * 6.5)

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate, int(periods))
        sortino = calculate_sortino_ratio(returns, risk_free_rate, int(periods))

        # Interpret Sharpe
        if sharpe > 2:
            interpretation = "excellent"
        elif sharpe > 1:
            interpretation = "good"
        elif sharpe > 0.5:
            interpretation = "average"
        elif sharpe > 0:
            interpretation = "below_average"
        else:
            interpretation = "poor"

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "risk_free_rate": risk_free_rate,
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "interpretation": interpretation,
            "avg_return_annualized": round(returns.mean() * periods * 100, 2),
            "volatility_annualized": round(returns.std() * np.sqrt(periods) * 100, 2),
            "data_points": len(returns),
            "metadata": {
                "required_columns": ["close", "timestamp"],
                "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_risk_reward_impl(
    symbol: str, entry_price: float = None, stop_loss: float = None,
    take_profit: float = None, interval: str = "60min"
) -> Dict[str, Any]:
    """Analyze risk/reward for a potential trade."""
    try:
        df = await get_ohlcv(symbol, interval)

        if df.empty or len(df) < 15:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
        df = df.dropna()

        current = df.iloc[-1]
        current_price = entry_price or current['close']
        current_atr = current['atr']

        # Calculate defaults if not provided
        if stop_loss is None:
            stop_loss = current_price - (2 * current_atr)
        if take_profit is None:
            take_profit = current_price + (3 * current_atr)

        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)

        if risk > 0:
            rr_ratio = reward / risk
        else:
            rr_ratio = 0

        # Win rate needed to be profitable
        breakeven_winrate = 1 / (1 + rr_ratio) if rr_ratio > 0 else 1

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "entry_price": round(current_price, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "risk_amount": round(risk, 4),
            "reward_amount": round(reward, 4),
            "risk_reward_ratio": round(rr_ratio, 2),
            "breakeven_winrate": round(breakeven_winrate * 100, 1),
            "assessment": "favorable" if rr_ratio >= 2 else ("acceptable" if rr_ratio >= 1 else "unfavorable"),
            "atr_reference": round(current_atr, 4),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_volatility_analysis_impl(symbol: str, interval: str = "60min") -> Dict[str, Any]:
    """Comprehensive volatility analysis."""
    try:
        df = await get_ohlcv(symbol, interval, output_size="full")

        if df.empty or len(df) < 30:
            return {"status": "error", "message": f"Insufficient data for {symbol}"}

        returns = calculate_returns(df['close'])
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)

        # Historical volatility (annualized)
        periods_per_year = 252 * 6.5  # For hourly
        hist_vol = returns.std() * np.sqrt(periods_per_year)

        # Recent vs historical volatility
        recent_vol = returns.tail(20).std() * np.sqrt(periods_per_year)
        vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1

        current = df.iloc[-1]

        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "current_price": round(current['close'], 4),
            "current_atr": round(current['atr'], 4),
            "atr_percent": round((current['atr'] / current['close']) * 100, 2),
            "historical_volatility": round(hist_vol * 100, 2),
            "recent_volatility": round(recent_vol * 100, 2),
            "volatility_ratio": round(vol_ratio, 2),
            "volatility_regime": "high" if vol_ratio > 1.2 else ("low" if vol_ratio < 0.8 else "normal"),
            "daily_range_estimate": round(current['atr'] * 1.5, 4),
            "data_points": len(df),
            "metadata": {
                "required_columns": ["high", "low", "close", "timestamp"],
                "sql_template": "SELECT timestamp, high, low, close FROM ohlcv WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server: FastMCP) -> List[str]:
    """Register all risk management tools."""

    @mcp_server.tool()
    async def av_position_size(
        symbol: str, account_size: float, risk_percent: float = 2.0,
        atr_multiplier: float = 2.0, interval: str = "60min"
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size based on ATR volatility.

        Uses the ATR method to determine position size that risks
        a fixed percentage of account on each trade.

        Args:
            symbol: Stock ticker
            account_size: Total account value
            risk_percent: Percentage of account to risk (default 2%)
            atr_multiplier: ATR multiple for stop distance (default 2x)
            interval: Time interval
        """
        return await av_position_size_impl(symbol, account_size, risk_percent, atr_multiplier, interval)

    @mcp_server.tool()
    async def av_max_drawdown(symbol: str, interval: str = "60min") -> Dict[str, Any]:
        """
        Calculate maximum drawdown from peak.

        Shows the largest peak-to-trough decline in the price history.
        Useful for understanding worst-case scenarios.
        """
        return await av_max_drawdown_impl(symbol, interval)

    @mcp_server.tool()
    async def av_sharpe(
        symbol: str, risk_free_rate: float = 0.02, interval: str = "60min"
    ) -> Dict[str, Any]:
        """
        Calculate Sharpe and Sortino ratios.

        Sharpe: Risk-adjusted return (excess return / volatility)
        Sortino: Uses only downside deviation (better for asymmetric returns)

        Interpretation:
        - Sharpe > 2: Excellent
        - Sharpe > 1: Good
        - Sharpe > 0.5: Average
        - Sharpe < 0: Poor
        """
        return await av_sharpe_impl(symbol, risk_free_rate, interval)

    @mcp_server.tool()
    async def av_risk_reward(
        symbol: str, entry_price: float = 0, stop_loss: float = 0,
        take_profit: float = 0, interval: str = "60min"
    ) -> Dict[str, Any]:
        """
        Analyze risk/reward for a potential trade.

        If prices aren't specified, uses current price and ATR-based defaults.
        Calculates R:R ratio and breakeven win rate needed.
        """
        return await av_risk_reward_impl(
            symbol,
            entry_price if entry_price > 0 else None,
            stop_loss if stop_loss > 0 else None,
            take_profit if take_profit > 0 else None,
            interval
        )

    @mcp_server.tool()
    async def av_volatility_analysis(symbol: str, interval: str = "60min") -> Dict[str, Any]:
        """
        Comprehensive volatility analysis.

        Compares current volatility to historical levels,
        identifies volatility regime, and provides range estimates.
        """
        return await av_volatility_analysis_impl(symbol, interval)

    return ['av_position_size', 'av_max_drawdown', 'av_sharpe', 'av_risk_reward', 'av_volatility_analysis']
