# AlphaVantage Algorithmic Trading MCP Server Tools

## Comprehensive Implementation Plan

---

## 1. Executive Summary

This document outlines the architecture and implementation plan for a comprehensive suite of algorithmic trading MCP server tools powered by AlphaVantage data. The system is designed with a modular data provider layer, enabling easy integration with KDB+, other databases, or direct API access.

### Key Design Principles
1. **Separation of Concerns**: Data fetching is abstracted from tool logic
2. **Database Agnostic**: Tools work with any data source implementing the provider interface
3. **Hourly Timeframe Focus**: Optimized for intraday trading strategies
4. **Comprehensive Testing**: Each tool validated with real market data
5. **Full Documentation**: API specs with SQL-equivalent column requirements

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MCP Server (FastMCP)                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Technical       │  │ Sentiment       │  │ Risk            │                  │
│  │ Indicator Tools │  │ Analysis Tools  │  │ Management Tools│                  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                    │                            │
│           └────────────────────┼────────────────────┘                            │
│                                │                                                 │
│                    ┌───────────▼───────────┐                                     │
│                    │   Data Provider       │                                     │
│                    │   Interface (ABC)     │                                     │
│                    └───────────┬───────────┘                                     │
│                                │                                                 │
│        ┌───────────────────────┼───────────────────────┐                         │
│        │                       │                       │                         │
│  ┌─────▼─────┐         ┌───────▼─────┐         ┌───────▼─────┐                   │
│  │AlphaVantage│         │ KDB+        │         │ Other DB    │                   │
│  │ Provider  │         │ Provider    │         │ Provider    │                   │
│  └───────────┘         └─────────────┘         └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Provider Layer

### 3.1 Abstract Interface

```python
class DataProvider(ABC):
    @abstractmethod
    async def get_ohlcv(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get OHLCV (Open, High, Low, Close, Volume) data"""
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get latest quote for symbol"""
        pass

    @abstractmethod
    async def get_news_sentiment(self, symbols: List[str], start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get news sentiment data"""
        pass
```

### 3.2 Standard Data Schema

All providers must return data conforming to these schemas:

#### OHLCV Schema
| Column     | Type      | Description                    | SQL Equivalent                     |
|------------|-----------|--------------------------------|-----------------------------------|
| timestamp  | datetime  | Candle timestamp               | WHERE timestamp BETWEEN ? AND ?    |
| symbol     | string    | Ticker symbol                  | WHERE symbol = ?                   |
| open       | float     | Opening price                  | SELECT open                        |
| high       | float     | High price                     | SELECT high                        |
| low        | float     | Low price                      | SELECT low                         |
| close      | float     | Closing price                  | SELECT close                       |
| volume     | int       | Trading volume                 | SELECT volume                      |

#### News Sentiment Schema
| Column            | Type      | Description                   | SQL Equivalent                     |
|-------------------|-----------|-------------------------------|-----------------------------------|
| timestamp         | datetime  | Article publication time       | WHERE timestamp BETWEEN ? AND ?    |
| symbol            | string    | Related ticker                 | WHERE symbol = ?                   |
| title             | string    | Article headline               | SELECT title                       |
| sentiment_score   | float     | Sentiment (-1 to 1)           | SELECT sentiment_score             |
| sentiment_label   | string    | Bearish/Neutral/Bullish       | SELECT sentiment_label             |
| relevance_score   | float     | Symbol relevance (0 to 1)     | SELECT relevance_score             |

---

## 4. Tool Categories and Specifications

### 4.1 Moving Average Tools (5 Tools)

| # | Tool Name | Description | Required Columns | Parameters |
|---|-----------|-------------|------------------|------------|
| 1 | `av_sma` | Simple Moving Average | close, timestamp | period (default: 20) |
| 2 | `av_ema` | Exponential Moving Average | close, timestamp | period (default: 20) |
| 3 | `av_wma` | Weighted Moving Average | close, timestamp | period (default: 20) |
| 4 | `av_dema` | Double Exponential Moving Average | close, timestamp | period (default: 20) |
| 5 | `av_tema` | Triple Exponential Moving Average | close, timestamp | period (default: 20) |

### 4.2 Momentum Indicators (7 Tools)

| # | Tool Name | Description | Required Columns | Parameters |
|---|-----------|-------------|------------------|------------|
| 6 | `av_rsi` | Relative Strength Index | close, timestamp | period (default: 14) |
| 7 | `av_macd` | MACD with Signal & Histogram | close, timestamp | fast (12), slow (26), signal (9) |
| 8 | `av_stoch` | Stochastic Oscillator | high, low, close, timestamp | k_period (14), d_period (3) |
| 9 | `av_stochrsi` | Stochastic RSI | close, timestamp | rsi_period (14), stoch_period (14) |
| 10 | `av_roc` | Rate of Change | close, timestamp | period (10) |
| 11 | `av_mom` | Momentum | close, timestamp | period (10) |
| 12 | `av_ppo` | Percentage Price Oscillator | close, timestamp | fast (12), slow (26) |

### 4.3 Trend Indicators (5 Tools)

| # | Tool Name | Description | Required Columns | Parameters |
|---|-----------|-------------|------------------|------------|
| 13 | `av_adx` | Average Directional Index | high, low, close, timestamp | period (14) |
| 14 | `av_aroon` | Aroon Indicator | high, low, timestamp | period (25) |
| 15 | `av_aroon_osc` | Aroon Oscillator | high, low, timestamp | period (25) |
| 16 | `av_sar` | Parabolic SAR | high, low, timestamp | acceleration (0.02), maximum (0.2) |
| 17 | `av_supertrend` | SuperTrend | high, low, close, timestamp | period (10), multiplier (3.0) |

### 4.4 Volatility Indicators (6 Tools)

| # | Tool Name | Description | Required Columns | Parameters |
|---|-----------|-------------|------------------|------------|
| 18 | `av_bbands` | Bollinger Bands | close, timestamp | period (20), std_dev (2.0) |
| 19 | `av_atr` | Average True Range | high, low, close, timestamp | period (14) |
| 20 | `av_natr` | Normalized ATR | high, low, close, timestamp | period (14) |
| 21 | `av_trange` | True Range | high, low, close, timestamp | - |
| 22 | `av_keltner` | Keltner Channels | high, low, close, timestamp | period (20), multiplier (2.0) |
| 23 | `av_stddev` | Standard Deviation | close, timestamp | period (20) |

### 4.5 Volume Indicators (5 Tools)

| # | Tool Name | Description | Required Columns | Parameters |
|---|-----------|-------------|------------------|------------|
| 24 | `av_obv` | On-Balance Volume | close, volume, timestamp | - |
| 25 | `av_vwap` | Volume Weighted Avg Price | high, low, close, volume, timestamp | - |
| 26 | `av_ad` | Accumulation/Distribution | high, low, close, volume, timestamp | - |
| 27 | `av_adosc` | A/D Oscillator (Chaikin) | high, low, close, volume, timestamp | fast (3), slow (10) |
| 28 | `av_mfi` | Money Flow Index | high, low, close, volume, timestamp | period (14) |

### 4.6 Oscillators (4 Tools)

| # | Tool Name | Description | Required Columns | Parameters |
|---|-----------|-------------|------------------|------------|
| 29 | `av_cci` | Commodity Channel Index | high, low, close, timestamp | period (20) |
| 30 | `av_willr` | Williams %R | high, low, close, timestamp | period (14) |
| 31 | `av_ultosc` | Ultimate Oscillator | high, low, close, timestamp | periods (7, 14, 28) |
| 32 | `av_trix` | Triple Exponential Average | close, timestamp | period (15) |

### 4.7 Signal Detection Tools (6 Tools)

| # | Tool Name | Description | Required Columns | Combines |
|---|-----------|-------------|------------------|----------|
| 33 | `av_golden_cross` | SMA Golden/Death Cross Detection | close, timestamp | SMA(50), SMA(200) |
| 34 | `av_macd_crossover` | MACD Line/Signal Crossover | close, timestamp | MACD |
| 35 | `av_rsi_divergence` | RSI Price Divergence | close, timestamp | RSI |
| 36 | `av_bb_breakout` | Bollinger Band Breakout | close, timestamp | Bollinger Bands |
| 37 | `av_multi_signal` | Multi-indicator Confluence | all OHLCV | RSI, MACD, Stoch, ADX |
| 38 | `av_trend_strength` | Trend Strength Score | high, low, close, timestamp | ADX, AROON, SAR |

### 4.8 Sentiment Analysis Tools (4 Tools)

| # | Tool Name | Description | Required Columns | Data Source |
|---|-----------|-------------|------------------|-------------|
| 39 | `av_news_sentiment` | Aggregate News Sentiment | - | NEWS_SENTIMENT API |
| 40 | `av_sentiment_trend` | Sentiment Trend Analysis | - | NEWS_SENTIMENT API |
| 41 | `av_sentiment_price_corr` | Sentiment-Price Correlation | close, timestamp + sentiment | Combined |
| 42 | `av_sentiment_momentum` | Sentiment Momentum Indicator | - | NEWS_SENTIMENT API |

### 4.9 Risk Management Tools (5 Tools)

| # | Tool Name | Description | Required Columns | Parameters |
|---|-----------|-------------|------------------|------------|
| 43 | `av_volatility_position` | Volatility-based Position Sizing | high, low, close, timestamp | risk_pct (2.0) |
| 44 | `av_max_drawdown` | Maximum Drawdown Calculator | close, timestamp | - |
| 45 | `av_sharpe_ratio` | Sharpe Ratio Calculator | close, timestamp | risk_free_rate (0.02) |
| 46 | `av_sortino_ratio` | Sortino Ratio Calculator | close, timestamp | risk_free_rate (0.02) |
| 47 | `av_risk_reward` | Risk/Reward Analyzer | high, low, close, timestamp | - |

### 4.10 FX-Specific Tools (5 Tools)

| # | Tool Name | Description | Required Columns | Parameters |
|---|-----------|-------------|------------------|------------|
| 48 | `av_fx_sma` | FX Simple Moving Average | close, timestamp | period (20) |
| 49 | `av_fx_rsi` | FX Relative Strength Index | close, timestamp | period (14) |
| 50 | `av_fx_macd` | FX MACD | close, timestamp | fast, slow, signal |
| 51 | `av_fx_bbands` | FX Bollinger Bands | close, timestamp | period, std_dev |
| 52 | `av_fx_pivot` | FX Pivot Points | high, low, close | - |

---

## 5. SQL Column Requirements Summary

For integration with KDB+ or other SQL-compatible databases, here are the required column specifications:

### Base OHLCV Query Template
```sql
SELECT timestamp, symbol, open, high, low, close, volume
FROM {table_name}
WHERE symbol = '{symbol}'
  AND timestamp >= '{start_time}'
  AND timestamp <= '{end_time}'
ORDER BY timestamp ASC
```

### Tool-Specific Requirements

| Tool Category | Minimum Columns | Optional | WHERE Clauses |
|---------------|-----------------|----------|---------------|
| Moving Averages | close, timestamp | symbol | timestamp range, symbol filter |
| Momentum | close, timestamp | high, low | timestamp range, symbol filter |
| Trend | high, low, close, timestamp | symbol | timestamp range, symbol filter |
| Volatility | high, low, close, timestamp | symbol | timestamp range, symbol filter |
| Volume | close, volume, timestamp | high, low | timestamp range, symbol filter |
| Sentiment | timestamp | symbol | timestamp range, symbol filter, sentiment_score filter |

---

## 6. Implementation Phases

### Phase 1: Foundation (Data Provider Layer)
- [ ] Abstract DataProvider interface
- [ ] AlphaVantage implementation
- [ ] Rate limiting (75 calls/minute)
- [ ] Caching layer for API responses
- [ ] Error handling and retry logic

### Phase 2: Core Technical Indicators
- [ ] Moving Averages (5 tools)
- [ ] Momentum Indicators (7 tools)
- [ ] Testing with real data

### Phase 3: Advanced Indicators
- [ ] Trend Indicators (5 tools)
- [ ] Volatility Indicators (6 tools)
- [ ] Volume Indicators (5 tools)
- [ ] Oscillators (4 tools)

### Phase 4: Signal Detection & Analysis
- [ ] Signal Detection tools (6 tools)
- [ ] Sentiment Analysis tools (4 tools)
- [ ] Risk Management tools (5 tools)

### Phase 5: FX Tools & Testing
- [ ] FX-Specific tools (5 tools)
- [ ] Comprehensive testing suite
- [ ] Performance optimization

### Phase 6: Documentation
- [ ] API documentation for all tools
- [ ] Usage examples
- [ ] Integration guides (KDB+, other DBs)

---

## 7. Testing Strategy

### Test Symbols
- **Equities**: AAPL, MSFT, GOOGL, TSLA, JPM
- **FX Pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD

### Test Scenarios
1. **Accuracy**: Compare tool outputs with known values
2. **Edge Cases**: Empty data, single data point, holidays
3. **Time Ranges**: 1 day, 1 week, 1 month of hourly data
4. **Cross-validation**: Compare multiple indicators

---

## 8. API Rate Limit Management

```python
class RateLimiter:
    """Rate limiter for 75 calls/minute"""
    def __init__(self, max_calls=75, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    async def acquire(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            await asyncio.sleep(sleep_time)
        self.calls.append(time.time())
```

---

## 9. File Structure

```
src/mcp_server/
├── providers/
│   ├── __init__.py
│   ├── base.py                    # Abstract DataProvider
│   ├── alphavantage.py            # AlphaVantage implementation
│   └── cache.py                   # Response caching
├── tools/
│   ├── alphavantage/
│   │   ├── __init__.py
│   │   ├── moving_averages.py     # SMA, EMA, WMA, DEMA, TEMA
│   │   ├── momentum.py            # RSI, MACD, Stochastic, etc.
│   │   ├── trend.py               # ADX, AROON, SAR, etc.
│   │   ├── volatility.py          # BB, ATR, Keltner, etc.
│   │   ├── volume.py              # OBV, VWAP, A/D, MFI
│   │   ├── oscillators.py         # CCI, Williams %R, etc.
│   │   ├── signals.py             # Crossovers, divergences
│   │   ├── sentiment.py           # News sentiment tools
│   │   ├── risk.py                # Risk management tools
│   │   └── fx.py                  # FX-specific tools
└── docs/
    └── api/
        ├── moving_averages.md
        ├── momentum.md
        └── ...
```

---

## 10. Sample Tool Implementation

```python
# Example: SMA Tool Implementation

async def av_sma_impl(
    symbol: str,
    interval: str = "60min",
    period: int = 20,
    start_time: str = None,
    end_time: str = None,
    data_provider: str = "alphavantage"
) -> Dict[str, Any]:
    """
    Calculate Simple Moving Average.

    Required Data Columns:
    - close: Closing price
    - timestamp: Time of the candle

    SQL Equivalent:
    SELECT timestamp, close,
           AVG(close) OVER (ORDER BY timestamp ROWS BETWEEN {period-1} PRECEDING AND CURRENT ROW) as sma
    FROM ohlcv
    WHERE symbol = '{symbol}' AND timestamp BETWEEN '{start_time}' AND '{end_time}'
    """
    provider = get_provider(data_provider)
    data = await provider.get_ohlcv(symbol, interval, start_time, end_time)

    # Calculate SMA
    data['sma'] = data['close'].rolling(window=period).mean()

    return {
        "status": "success",
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "data": data[['timestamp', 'close', 'sma']].dropna().to_dict('records'),
        "latest_sma": data['sma'].iloc[-1],
        "metadata": {
            "required_columns": ["close", "timestamp"],
            "sql_template": "SELECT timestamp, close FROM ohlcv WHERE symbol = ? AND timestamp BETWEEN ? AND ?"
        }
    }
```

---

## 11. Next Steps

1. **Immediate**: Implement data provider layer
2. **Day 1**: Core moving average and momentum tools
3. **Day 2**: Trend, volatility, and volume tools
4. **Day 3**: Signal detection and sentiment tools
5. **Day 4**: Risk management and FX tools
6. **Day 5**: Testing and documentation

---

*Document Version: 1.0*
*Created: 2024*
*AlphaVantage API Key: [Configured]*
