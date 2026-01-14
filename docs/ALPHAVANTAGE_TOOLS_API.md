# AlphaVantage Algo Trading MCP Tools - API Documentation

## Overview

This document provides comprehensive API documentation for the AlphaVantage algorithmic trading MCP server tools. The tools are designed for high-frequency traders and quantitative analysts working with equities and forex data.

**Total Tools: 47**

---

## Table of Contents

1. [Moving Average Tools](#1-moving-average-tools)
2. [Momentum Indicator Tools](#2-momentum-indicator-tools)
3. [Trend Indicator Tools](#3-trend-indicator-tools)
4. [Volatility Indicator Tools](#4-volatility-indicator-tools)
5. [Volume Indicator Tools](#5-volume-indicator-tools)
6. [Oscillator Tools](#6-oscillator-tools)
7. [Signal Detection Tools](#7-signal-detection-tools)
8. [Sentiment Analysis Tools](#8-sentiment-analysis-tools)
9. [Risk Management Tools](#9-risk-management-tools)
10. [FX-Specific Tools](#10-fx-specific-tools)
11. [Data Provider Integration](#11-data-provider-integration)

---

## 1. Moving Average Tools

### av_sma
**Simple Moving Average**

Calculate the unweighted mean of prices over a specified period.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| symbol | str | required | Stock ticker (e.g., 'AAPL') |
| period | int | 20 | Number of periods for averaging |
| interval | str | "60min" | Time interval ('1min', '5min', '15min', '30min', '60min') |
| price_type | str | "close" | Price type ('open', 'high', 'low', 'close') |
| is_fx | bool | False | Set True for forex pairs |
| from_currency | str | "" | Source currency for FX |
| to_currency | str | "" | Target currency for FX |

**Required Data Columns:** `close`, `timestamp`

**SQL Equivalent:**
```sql
SELECT timestamp, close,
       AVG(close) OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma_20
FROM ohlcv WHERE symbol = 'AAPL' ORDER BY timestamp
```

**Example Usage:**
```python
# Equity
av_sma(symbol="AAPL", period=20)
av_sma(symbol="MSFT", period=50, interval="30min")

# Forex
av_sma(is_fx=True, from_currency="EUR", to_currency="USD", period=20)
```

**Response:**
```json
{
  "status": "success",
  "symbol": "AAPL",
  "period": 20,
  "current_price": 261.05,
  "current_sma": 259.87,
  "trend": "bullish",
  "price_vs_sma_pct": 0.45
}
```

---

### av_ema
**Exponential Moving Average**

Calculate EMA with more weight on recent prices.

**Formula:** `EMA = Price * k + EMA(prev) * (1 - k)`, where `k = 2 / (period + 1)`

**Parameters:** Same as av_sma

**Example:**
```python
av_ema(symbol="TSLA", period=12)
av_ema(symbol="GOOGL", period=26, interval="15min")
```

---

### av_wma
**Weighted Moving Average**

Linear weighting with more recent prices receiving higher weights.

**Formula:** `WMA = (P1*1 + P2*2 + ... + Pn*n) / (1 + 2 + ... + n)`

---

### av_dema
**Double Exponential Moving Average**

Reduces lag by combining two EMAs.

**Formula:** `DEMA = 2 * EMA(n) - EMA(EMA(n))`

---

### av_tema
**Triple Exponential Moving Average**

Minimizes lag using three EMAs.

**Formula:** `TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))`

---

### av_ma_compare
**Compare Multiple Moving Averages**

Calculate and compare all MA types across multiple periods.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| symbol | str | required | Stock ticker |
| periods | str | "10,20,50" | Comma-separated periods |

**Example:**
```python
av_ma_compare(symbol="AAPL", periods="9,21,50,200")
```

---

## 2. Momentum Indicator Tools

### av_rsi
**Relative Strength Index**

Measures overbought/oversold conditions on a scale of 0-100.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| symbol | str | required | Stock ticker |
| period | int | 14 | RSI period |
| overbought | float | 70 | Overbought threshold |
| oversold | float | 30 | Oversold threshold |

**Required Data Columns:** `close`, `timestamp`

**SQL Template:**
```sql
SELECT timestamp, close FROM ohlcv
WHERE symbol = ? ORDER BY timestamp
```

**Trading Signals:**
- RSI > 70: Overbought (potential sell)
- RSI < 30: Oversold (potential buy)
- RSI divergence from price: Reversal signal

**Example:**
```python
av_rsi(symbol="AAPL", period=14)
av_rsi(symbol="TSLA", period=9, overbought=80, oversold=20)
```

---

### av_macd
**Moving Average Convergence Divergence**

Trend-following momentum indicator.

**Components:**
- MACD Line: EMA(12) - EMA(26)
- Signal Line: EMA(9) of MACD
- Histogram: MACD - Signal

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| fast_period | int | 12 | Fast EMA period |
| slow_period | int | 26 | Slow EMA period |
| signal_period | int | 9 | Signal line period |

**Trading Signals:**
- MACD crosses above Signal: Bullish
- MACD crosses below Signal: Bearish
- Histogram expanding: Trend strengthening

---

### av_stochastic
**Stochastic Oscillator**

Compares closing price to price range.

**Components:**
- %K: Main stochastic line
- %D: Signal line (SMA of %K)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| k_period | int | 14 | %K period |
| d_period | int | 3 | %D period |

**Required Data Columns:** `high`, `low`, `close`, `timestamp`

---

### av_stochrsi
**Stochastic RSI**

Applies Stochastic formula to RSI values.

More sensitive than regular RSI.

---

### av_roc
**Rate of Change**

Percentage change between current and n periods ago.

**Formula:** `ROC = ((Current - Previous) / Previous) * 100`

---

### av_momentum
**Momentum Indicator**

Absolute price difference from n periods ago.

**Formula:** `MOM = Current Price - Price n periods ago`

---

### av_ppo
**Percentage Price Oscillator**

MACD expressed as percentage for cross-security comparison.

**Formula:** `PPO = ((EMA_fast - EMA_slow) / EMA_slow) * 100`

---

## 3. Trend Indicator Tools

### av_adx
**Average Directional Index**

Measures trend strength (0-100).

**Components:**
- ADX: Trend strength
- +DI: Positive directional indicator
- -DI: Negative directional indicator

**Required Data Columns:** `high`, `low`, `close`, `timestamp`

**Interpretation:**
- ADX < 20: Weak/absent trend
- ADX 20-40: Strong trend
- ADX > 40: Very strong trend

---

### av_aroon
**Aroon Indicator**

Measures time since highest high and lowest low.

**Components:**
- Aroon Up
- Aroon Down
- Aroon Oscillator (Up - Down)

---

### av_sar
**Parabolic SAR**

Stop and Reverse indicator for trend following.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| acceleration | float | 0.02 | Acceleration factor |
| maximum | float | 0.2 | Maximum acceleration |

**Use Cases:**
- Trailing stop-loss levels
- Entry/exit signals on SAR flip

---

### av_supertrend
**SuperTrend**

Trend-following indicator based on ATR.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| period | int | 10 | ATR period |
| multiplier | float | 3.0 | ATR multiplier |

---

## 4. Volatility Indicator Tools

### av_bbands
**Bollinger Bands**

Volatility bands around SMA.

**Components:**
- Upper Band: SMA + (std_dev * StdDev)
- Middle Band: SMA
- Lower Band: SMA - (std_dev * StdDev)
- Bandwidth: (Upper - Lower) / Middle * 100
- %B: (Price - Lower) / (Upper - Lower)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| period | int | 20 | SMA period |
| std_dev | float | 2.0 | Standard deviation multiplier |

**Required Data Columns:** `close`, `timestamp`

---

### av_atr
**Average True Range**

Measures market volatility.

**Uses:**
- Stop-loss placement (typically 2x ATR)
- Position sizing based on volatility
- Volatility regime identification

**Required Data Columns:** `high`, `low`, `close`, `timestamp`

---

### av_keltner
**Keltner Channels**

Volatility channels using ATR instead of standard deviation.

**Components:**
- Upper: EMA + (multiplier * ATR)
- Middle: EMA
- Lower: EMA - (multiplier * ATR)

---

### av_stddev
**Standard Deviation**

Measures price dispersion.

---

## 5. Volume Indicator Tools

### av_obv
**On-Balance Volume**

Cumulative volume indicator.

**Formula:** Adds volume on up days, subtracts on down days.

**Required Data Columns:** `close`, `volume`, `timestamp`

**Signals:**
- Rising OBV: Buying pressure
- Falling OBV: Selling pressure
- OBV divergence: Potential reversal

---

### av_vwap
**Volume Weighted Average Price**

Institutional trading benchmark.

**Required Data Columns:** `high`, `low`, `close`, `volume`, `timestamp`

**Trading:**
- Price above VWAP: Bullish bias
- Price below VWAP: Bearish bias

---

### av_ad
**Accumulation/Distribution Line**

Measures cumulative money flow.

**Required Data Columns:** `high`, `low`, `close`, `volume`, `timestamp`

---

### av_mfi
**Money Flow Index**

Volume-weighted RSI.

**Signals:**
- MFI > 80: Overbought
- MFI < 20: Oversold

---

### av_adosc
**Chaikin A/D Oscillator**

Momentum of A/D Line.

---

## 6. Oscillator Tools

### av_cci
**Commodity Channel Index**

Measures price deviation from average.

**Signals:**
- CCI > 100: Overbought
- CCI < -100: Oversold

**Required Data Columns:** `high`, `low`, `close`, `timestamp`

---

### av_willr
**Williams %R**

Momentum oscillator (-100 to 0).

**Signals:**
- %R > -20: Overbought
- %R < -80: Oversold

---

### av_ultosc
**Ultimate Oscillator**

Uses three timeframes to reduce false signals.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| period1 | int | 7 | First period |
| period2 | int | 14 | Second period |
| period3 | int | 28 | Third period |

---

### av_trix
**TRIX**

Triple exponential average rate of change.

---

## 7. Signal Detection Tools

### av_golden_cross
**Golden/Death Cross Detection**

Detects SMA crossover patterns.

**Signals:**
- Golden Cross: Fast SMA crosses above Slow SMA (bullish)
- Death Cross: Fast SMA crosses below Slow SMA (bearish)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| fast_period | int | 50 | Fast SMA period |
| slow_period | int | 200 | Slow SMA period |

---

### av_macd_crossover
**MACD Crossover Signals**

Detects MACD/Signal and zero line crossovers.

---

### av_rsi_divergence
**RSI Divergence Detection**

Identifies bullish and bearish divergences.

**Divergence Types:**
- Bullish: Price lower lows, RSI higher lows
- Bearish: Price higher highs, RSI lower highs

---

### av_bb_breakout
**Bollinger Band Breakout**

Detects band breakouts and squeezes.

---

### av_multi_signal
**Multi-Indicator Signal Generator**

Combines 6 indicators for comprehensive analysis.

**Indicators Used:**
- SMA trend
- EMA trend
- Price vs SMA
- RSI signal
- MACD signal
- Stochastic signal

**Response includes confidence score (0-100%).**

---

### av_trend_strength
**Trend Strength Analysis**

Analyzes trend using ADX and moving averages.

---

## 8. Sentiment Analysis Tools

### av_news_sentiment
**News Sentiment Aggregator**

Aggregates news sentiment for symbols.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| symbols | str | required | Comma-separated tickers |
| limit | int | 50 | Number of articles |
| topics | str | "" | Optional topic filter |

**Required Data Columns:** `timestamp`, `symbol`, `title`, `sentiment_score`, `sentiment_label`, `relevance_score`

**Example:**
```python
av_news_sentiment(symbols="AAPL,MSFT", limit=100)
av_news_sentiment(symbols="TSLA", topics="technology,earnings")
```

---

### av_sentiment_trend
**Sentiment Trend Analysis**

Tracks sentiment evolution over time.

---

### av_sentiment_price_corr
**Sentiment-Price Correlation**

Analyzes correlation between sentiment and price.

---

### av_sentiment_momentum
**Sentiment Momentum**

Calculates sentiment momentum indicator.

---

## 9. Risk Management Tools

### av_position_size
**Volatility-Based Position Sizing**

Calculate optimal position size using ATR.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| symbol | str | required | Stock ticker |
| account_size | float | required | Total account value |
| risk_percent | float | 2.0 | % of account to risk |
| atr_multiplier | float | 2.0 | ATR multiple for stop |

**Example:**
```python
av_position_size(symbol="AAPL", account_size=100000, risk_percent=2.0)
```

---

### av_max_drawdown
**Maximum Drawdown Calculator**

Calculates largest peak-to-trough decline.

---

### av_sharpe
**Sharpe Ratio Calculator**

Calculates Sharpe and Sortino ratios.

**Interpretation:**
- Sharpe > 2: Excellent
- Sharpe > 1: Good
- Sharpe > 0.5: Average
- Sharpe < 0: Poor

---

### av_risk_reward
**Risk/Reward Analyzer**

Analyzes R:R ratio for potential trades.

---

### av_volatility_analysis
**Comprehensive Volatility Analysis**

Compares current vs historical volatility.

---

## 10. FX-Specific Tools

### av_fx_quote
**FX Exchange Rate**

Get current exchange rate.

**Example:**
```python
av_fx_quote(from_currency="EUR", to_currency="USD")
```

---

### av_fx_technical
**FX Technical Analysis**

Comprehensive technical analysis for currency pairs.

---

### av_fx_pivot
**FX Pivot Points**

Calculate pivot points with R1-R3 and S1-S3 levels.

---

### av_fx_volatility
**FX Volatility Analysis**

ATR in pips, volatility percentile.

---

### av_fx_strength
**FX Currency Strength**

Compare strength across multiple pairs.

---

## 11. Data Provider Integration

### Database Column Requirements

Each tool specifies required columns for database integration:

| Tool Category | Required Columns | Optional |
|---------------|-----------------|----------|
| Moving Averages | close, timestamp | symbol |
| Momentum | close, timestamp | high, low |
| Trend | high, low, close, timestamp | symbol |
| Volatility | high, low, close, timestamp | symbol |
| Volume | close, volume, timestamp | high, low |
| Sentiment | timestamp, sentiment_score | symbol, title |

### SQL Templates

Each tool provides an SQL template for database integration:

```sql
-- Moving Averages
SELECT timestamp, close FROM ohlcv
WHERE symbol = ? ORDER BY timestamp

-- Volume Indicators
SELECT timestamp, high, low, close, volume FROM ohlcv
WHERE symbol = ? ORDER BY timestamp

-- News Sentiment
SELECT timestamp, symbol, title, sentiment_score, sentiment_label, relevance_score
FROM news_sentiment WHERE symbol = ? ORDER BY timestamp DESC
```

### Switching Data Providers

To use a different data source (e.g., KDB+):

1. Implement the `DataProvider` interface in `providers/base.py`
2. Register the provider with `@register_provider("kdb")`
3. Update tool calls to use the new provider

---

## Rate Limits

- AlphaVantage API: 75 calls per minute
- Built-in rate limiter handles this automatically
- Response caching reduces API calls (5-minute TTL for real-time, 1-hour for historical)

---

## Error Handling

All tools return a consistent response structure:

**Success:**
```json
{
  "status": "success",
  "symbol": "AAPL",
  "data": [...],
  "metadata": {
    "required_columns": ["close", "timestamp"],
    "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ?"
  }
}
```

**Error:**
```json
{
  "status": "error",
  "message": "Insufficient data for AAPL. Need 50 points, got 20."
}
```

---

*Documentation Version: 1.0*
*Tools Count: 47*
*Last Updated: 2024*
