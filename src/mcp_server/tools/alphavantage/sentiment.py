"""
Sentiment Analysis Tools for Algorithmic Trading.

This module provides sentiment-based indicators:
- News Sentiment Aggregator
- Sentiment Trend Analysis
- Sentiment-Price Correlation
- Sentiment Momentum
"""

import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

from .data_provider import get_news_sentiment, get_ohlcv

logger = logging.getLogger(__name__)


async def av_news_sentiment_impl(
    symbols: str, limit: int = 50, topics: str = ""
) -> Dict[str, Any]:
    """Get aggregated news sentiment for symbols."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        topic_list = [t.strip() for t in topics.split(',')] if topics else None

        df = await get_news_sentiment(symbols=symbol_list, topics=topic_list, limit=limit)

        if df.empty:
            return {"status": "success", "message": "No news articles found", "articles": 0}

        # Aggregate by symbol
        aggregated = {}
        for symbol in symbol_list:
            symbol_df = df[df['symbol'].str.upper() == symbol]
            if not symbol_df.empty:
                aggregated[symbol] = {
                    "articles": len(symbol_df),
                    "avg_sentiment": round(symbol_df['sentiment_score'].mean(), 3),
                    "avg_relevance": round(symbol_df['relevance_score'].mean(), 3),
                    "bullish_count": len(symbol_df[symbol_df['sentiment_score'] > 0.15]),
                    "bearish_count": len(symbol_df[symbol_df['sentiment_score'] < -0.15]),
                    "neutral_count": len(symbol_df[(symbol_df['sentiment_score'] >= -0.15) & (symbol_df['sentiment_score'] <= 0.15)]),
                    "latest_sentiment": symbol_df.iloc[0]['sentiment_label'] if len(symbol_df) > 0 else "unknown"
                }

        # Overall sentiment
        overall_sentiment = df['sentiment_score'].mean()
        if overall_sentiment > 0.15:
            sentiment_label = "bullish"
        elif overall_sentiment < -0.15:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"

        return {
            "status": "success",
            "symbols": symbol_list,
            "total_articles": len(df),
            "overall_sentiment_score": round(overall_sentiment, 3),
            "overall_sentiment": sentiment_label,
            "by_symbol": aggregated,
            "recent_headlines": df[['timestamp', 'symbol', 'title', 'sentiment_label', 'sentiment_score']].head(10).to_dict('records'),
            "metadata": {
                "required_columns": ["timestamp", "symbol", "title", "sentiment_score", "sentiment_label", "relevance_score"],
                "sql_template": "SELECT * FROM news_sentiment WHERE symbol IN (?) ORDER BY timestamp DESC LIMIT ?"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_sentiment_trend_impl(
    symbol: str, limit: int = 100
) -> Dict[str, Any]:
    """Analyze sentiment trend over time."""
    try:
        df = await get_news_sentiment(symbols=[symbol], limit=limit)

        if df.empty:
            return {"status": "success", "message": "No news articles found", "articles": 0}

        # Sort by time
        df = df.sort_values('timestamp')

        # Calculate rolling sentiment
        df['rolling_sentiment'] = df['sentiment_score'].rolling(window=5, min_periods=1).mean()

        # Trend analysis
        if len(df) >= 10:
            recent = df.tail(10)['sentiment_score'].mean()
            older = df.head(10)['sentiment_score'].mean()
            if recent > older + 0.1:
                trend = "improving"
            elif recent < older - 0.1:
                trend = "deteriorating"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        current_sentiment = df.iloc[-1]['sentiment_score'] if len(df) > 0 else 0

        return {
            "status": "success",
            "symbol": symbol,
            "total_articles": len(df),
            "current_sentiment": round(current_sentiment, 3),
            "avg_sentiment": round(df['sentiment_score'].mean(), 3),
            "sentiment_trend": trend,
            "sentiment_volatility": round(df['sentiment_score'].std(), 3),
            "time_range": {
                "from": str(df['timestamp'].min()) if len(df) > 0 else None,
                "to": str(df['timestamp'].max()) if len(df) > 0 else None
            },
            "trend_data": df[['timestamp', 'sentiment_score', 'rolling_sentiment']].tail(20).to_dict('records'),
            "metadata": {
                "required_columns": ["timestamp", "sentiment_score"],
                "sql_template": "SELECT timestamp, sentiment_score FROM news_sentiment WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_sentiment_price_corr_impl(
    symbol: str, limit: int = 50
) -> Dict[str, Any]:
    """Analyze correlation between sentiment and price movement."""
    try:
        # Get both sentiment and price data
        sentiment_df = await get_news_sentiment(symbols=[symbol], limit=limit)
        price_df = await get_ohlcv(symbol, interval="60min")

        if sentiment_df.empty:
            return {"status": "error", "message": "No sentiment data available"}

        if price_df.empty:
            return {"status": "error", "message": "No price data available"}

        # Aggregate daily sentiment
        sentiment_df['date'] = sentiment_df['timestamp'].dt.date
        daily_sentiment = sentiment_df.groupby('date').agg({
            'sentiment_score': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'article_count'})

        # Aggregate daily price
        price_df['date'] = price_df['timestamp'].dt.date
        daily_price = price_df.groupby('date').agg({
            'close': 'last',
            'open': 'first'
        })
        daily_price['return'] = (daily_price['close'] / daily_price['open'] - 1) * 100

        # Merge and calculate correlation
        merged = daily_sentiment.join(daily_price, how='inner')

        if len(merged) < 5:
            return {
                "status": "success",
                "symbol": symbol,
                "message": "Insufficient overlapping data for correlation",
                "data_points": len(merged)
            }

        correlation = merged['sentiment_score'].corr(merged['return'])

        # Leading indicator analysis
        merged['sentiment_lag1'] = merged['sentiment_score'].shift(1)
        lagged_corr = merged['sentiment_lag1'].corr(merged['return']) if len(merged) > 5 else None

        return {
            "status": "success",
            "symbol": symbol,
            "data_points": len(merged),
            "sentiment_price_correlation": round(correlation, 3) if not pd.isna(correlation) else None,
            "lagged_correlation": round(lagged_corr, 3) if lagged_corr and not pd.isna(lagged_corr) else None,
            "interpretation": "strong_positive" if correlation > 0.5 else
                            ("moderate_positive" if correlation > 0.2 else
                            ("weak" if abs(correlation) < 0.2 else
                            ("moderate_negative" if correlation > -0.5 else "strong_negative"))),
            "predictive_value": "high" if lagged_corr and abs(lagged_corr) > 0.3 else "low",
            "avg_sentiment": round(merged['sentiment_score'].mean(), 3),
            "avg_return": round(merged['return'].mean(), 2),
            "metadata": {
                "required_columns": ["timestamp", "close", "sentiment_score"],
                "sql_template": "SELECT a.date, a.sentiment_score, b.close FROM news_sentiment a JOIN ohlcv b ON a.symbol = b.symbol"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def av_sentiment_momentum_impl(symbol: str, limit: int = 100) -> Dict[str, Any]:
    """Calculate sentiment momentum indicator."""
    try:
        df = await get_news_sentiment(symbols=[symbol], limit=limit)

        if df.empty or len(df) < 10:
            return {"status": "error", "message": "Insufficient sentiment data"}

        df = df.sort_values('timestamp')

        # Calculate sentiment momentum (rate of change)
        df['rolling_5'] = df['sentiment_score'].rolling(window=5, min_periods=3).mean()
        df['rolling_10'] = df['sentiment_score'].rolling(window=10, min_periods=5).mean()
        df = df.dropna()

        if len(df) < 2:
            return {"status": "error", "message": "Insufficient data after calculation"}

        current = df.iloc[-1]
        momentum = current['rolling_5'] - current['rolling_10']

        # Classify momentum
        if momentum > 0.1:
            signal = "strong_bullish_momentum"
        elif momentum > 0:
            signal = "bullish_momentum"
        elif momentum > -0.1:
            signal = "bearish_momentum"
        else:
            signal = "strong_bearish_momentum"

        return {
            "status": "success",
            "symbol": symbol,
            "current_sentiment": round(current['sentiment_score'], 3),
            "fast_avg": round(current['rolling_5'], 3),
            "slow_avg": round(current['rolling_10'], 3),
            "momentum": round(momentum, 3),
            "signal": signal,
            "articles_analyzed": len(df),
            "metadata": {
                "required_columns": ["timestamp", "sentiment_score"],
                "sql_template": "SELECT timestamp, sentiment_score FROM news_sentiment WHERE symbol = ? ORDER BY timestamp"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server: FastMCP) -> List[str]:
    """Register all sentiment analysis tools."""

    @mcp_server.tool()
    async def av_news_sentiment(
        symbols: str, limit: int = 50, topics: str = ""
    ) -> Dict[str, Any]:
        """
        Get aggregated news sentiment for one or more symbols.

        Analyzes recent news articles and provides sentiment scores.

        Args:
            symbols: Comma-separated list of tickers (e.g., "AAPL,MSFT")
            limit: Number of articles to analyze (max 1000)
            topics: Optional comma-separated topics (e.g., "technology,earnings")

        Returns sentiment breakdown: bullish, bearish, neutral counts and overall score.
        """
        return await av_news_sentiment_impl(symbols, limit, topics)

    @mcp_server.tool()
    async def av_sentiment_trend(symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Analyze sentiment trend over time for a symbol.

        Tracks how sentiment is evolving (improving, deteriorating, stable).
        Includes rolling sentiment average and volatility.
        """
        return await av_sentiment_trend_impl(symbol, limit)

    @mcp_server.tool()
    async def av_sentiment_price_corr(symbol: str, limit: int = 50) -> Dict[str, Any]:
        """
        Analyze correlation between news sentiment and price movement.

        Calculates both concurrent and lagged correlations to determine
        if sentiment has predictive value for price movement.
        """
        return await av_sentiment_price_corr_impl(symbol, limit)

    @mcp_server.tool()
    async def av_sentiment_momentum(symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Calculate sentiment momentum indicator.

        Compares fast and slow sentiment averages to identify
        momentum shifts in market sentiment.
        """
        return await av_sentiment_momentum_impl(symbol, limit)

    return ['av_news_sentiment', 'av_sentiment_trend', 'av_sentiment_price_corr', 'av_sentiment_momentum']
