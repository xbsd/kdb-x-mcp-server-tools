"""
AlphaVantage Algorithmic Trading MCP Server Tools.

This module provides comprehensive algo trading tools powered by AlphaVantage data.
Tools are organized into categories:
- Moving Averages: SMA, EMA, WMA, DEMA, TEMA
- Momentum: RSI, MACD, Stochastic, ROC, MOM, PPO
- Trend: ADX, AROON, SAR, SuperTrend
- Volatility: Bollinger Bands, ATR, Keltner Channels
- Volume: OBV, VWAP, A/D Line, MFI
- Oscillators: CCI, Williams %R, Ultimate Oscillator
- Signals: Crossovers, Divergences, Multi-indicator
- Sentiment: News sentiment, correlation
- Risk: Position sizing, Drawdown, Sharpe ratio
"""

from typing import List
from mcp.server.fastmcp import FastMCP

# Import all tool modules
from . import moving_averages
from . import momentum
from . import trend
from . import volatility
from . import volume
from . import oscillators
from . import signals
from . import sentiment
from . import risk
from . import fx


def register_tools(mcp_server: FastMCP) -> List[str]:
    """
    Register all AlphaVantage algo trading tools with the MCP server.

    This function discovers and registers all tools from submodules.

    Args:
        mcp_server: The FastMCP server instance

    Returns:
        List of registered tool names
    """
    registered_tools = []

    # List of tool modules to register
    tool_modules = [
        moving_averages,
        momentum,
        trend,
        volatility,
        volume,
        oscillators,
        signals,
        sentiment,
        risk,
        fx,
    ]

    for module in tool_modules:
        try:
            if hasattr(module, 'register_tools'):
                tools = module.register_tools(mcp_server)
                if tools:
                    registered_tools.extend(tools if isinstance(tools, list) else [tools])
        except Exception as e:
            print(f"Failed to register tools from {module.__name__}: {e}")

    return registered_tools


__all__ = ['register_tools']
