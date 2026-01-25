# 📊 期权蝴蝶策略量化模型完整指南

> **完整版本整合文档 - 包含理论、方法与实现**

---

## 目录

1. [核心理论基础](#一核心理论基础)
2. [数学模型详解](#二数学模型详解)
3. [关键技术问题](#三关键技术问题)
4. [数据获取方案](#四数据获取方案)
5. [完整实现框架](#五完整实现框架)
6. [评分与风险管理](#六评分与风险管理)
7. [回测框架设计](#七回测框架设计)
8. [实战指南](#八实战指南)

---

## 一、核心理论基础

### 1.1 期权蝴蝶策略原理

**Long Call Butterfly结构：**
```
买入 1份 较低行权价Call (K1)
卖出 2份 中间行权价Call (K2)
买入 1份 较高行权价Call (K3)

其中：K2 - K1 = K3 - K2 (等间距)

盈亏特征：
- 最大收益：到期时股价 = K2
- 最大损失：初始成本（净权利金）
- 盈亏平衡：K1 + 成本, K3 - 成本
```

**策略适用场景：**
- 预期价格小幅波动后盘整
- 波动率被高估（做空波动率）
- 时间价值衰减获利（正Theta）

---

### 1.2 决策流程图

```
输入股票代码
    ↓
【傅立叶变换分析】- 识别价格周期与趋势
    ├─ FFT分解价格序列（去趋势处理）
    ├─ 低频滤波 → 长期趋势 (>60天)
    ├─ 中频滤波 → 季节周期 (7-60天)
    └─ 功率谱分析 → 主导周期
    ↓
【ARIMA预测】- 价格方向预测
    ├─ 自动选择最优(p,d,q)参数
    ├─ 预测未来7-30天价格
    └─ 输出置信区间
    ↓
【GARCH波动率】- 波动率预测与IV分析
    ├─ 预测未来波动率
    ├─ 获取真实市场IV
    └─ 构建IV Skew曲线
    ↓
【Black-Scholes定价】- 精确期权定价
    ├─ 根据IV Skew调整每个行权价的σ
    ├─ 计算理论价格
    └─ 与市场价格比较
    ↓
【策略选择】
    ┌────────────────────────────────┐
    │ UP + TROUGH   → CALL Butterfly │
    │ 上涨+波谷 → 看涨后盘整           │
    ├────────────────────────────────┤
    │ DOWN + PEAK   → PUT Butterfly  │
    │ 下跌+波峰 → 看跌后盘整           │
    ├────────────────────────────────┤
    │ FLAT + ANY    → IRON Butterfly │
    │ 平稳+任意 → 双向中性盘整         │
    └────────────────────────────────┘
    ↓
【综合评分】- 多因子评分系统
    ├─ 价格匹配度 (35%)
    ├─ 波动率错配 (30%)
    ├─ 价格稳定性 (20%)
    ├─ 傅立叶对齐 (15%)
    └─ Greeks惩罚
    ↓
【风险检查】
    ├─ 流动性过滤
    ├─ Delta中性检验
    ├─ IV百分位检查
    └─ 盈亏比验证
    ↓
【输出推荐】
    STRONG_BUY / BUY / NEUTRAL / AVOID
```

---

## 二、数学模型详解

### 2.1 时间序列分析

#### **ARIMA模型**

**数学形式：**
```
ARIMA(p,d,q):
φ(L)(1-L)^d Y_t = θ(L)ε_t

其中：
- AR(p): φ(L) = 1 - φ₁L - φ₂L² - ... - φₚLᵖ
- I(d): 差分阶数
- MA(q): θ(L) = 1 + θ₁L + θ₂L² + ... + θ_qLᵍ
```

**实现要点：**
```python
# 自动选择最优参数
candidate_orders = [
    (1, 1, 1),  # 最简单
    (2, 1, 2),  # 标准配置
    (1, 1, 2),
    (2, 1, 1),
]

best_model = min(
    [ARIMA(data, order).fit() for order in candidates],
    key=lambda m: m.aic
)

# 预测含置信区间
forecast_result = best_model.get_forecast(steps=30)
forecast_df = forecast_result.summary_frame(alpha=0.05)
```

**关键指标：**
- 预测均值：作为中心行权价K2的参考
- 置信区间宽度：衡量价格稳定性
- AIC值：模型选择依据

---

#### **GARCH波动率模型**

**数学形式：**
```
GARCH(1,1):
r_t = μ + ε_t
ε_t = σ_t × z_t,  z_t ~ N(0,1)
σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²

条件：
- ω > 0
- α, β ≥ 0
- α + β < 1 (平稳性)
```

**实现要点：**
```python
returns = log(prices / prices.shift(1)) * 100
model = arch_model(returns, vol='Garch', p=1, q=1)
fitted = model.fit(disp='off')

# 预测未来波动率
forecast = fitted.forecast(horizon=30)
predicted_vol_annual = sqrt(forecast.variance) / 100 * sqrt(252)
```

**核心用途：**
1. 预测未来波动率 → 用于BS定价
2. 与市场IV比较 → 识别波动率错误定价
3. 波动率聚集检测 → 避开高波动期

---

### 2.2 傅立叶分析（关键改进）

#### **❌ 错误做法：直接对价格FFT**

```python
# 这是错误的！
prices = [100, 102, 105, 103, ...]
fft_result = np.fft.fft(prices)  # ❌
```

**问题：**
- 价格序列非平稳（有趋势）
- 随机游走产生虚假低频能量
- 无法区分真实周期 vs 噪声

---

#### **✅ 正确方法1：相对VWAP去趋势**

```python
def fourier_with_vwap_detrend(prices, volumes):
    """使用VWAP去趋势的傅立叶分析"""
    
    # 计算VWAP
    window = min(20, len(prices) // 3)
    pv = prices * volumes
    cumsum_pv = pd.Series(pv).rolling(window).sum()
    cumsum_v = pd.Series(volumes).rolling(window).sum()
    vwap = (cumsum_pv / cumsum_v).fillna(method='bfill').values
    
    # 去趋势：价格相对VWAP的偏移
    detrended = prices - vwap  # ✅ 真正的去趋势
    detrended = detrended[~np.isnan(detrended)]
    
    # 加窗函数（减少频谱泄漏）
    window_func = np.hanning(len(detrended))
    signal = detrended * window_func
    
    # FFT
    fft_result = np.fft.fft(signal)
    power = np.abs(fft_result) ** 2
    freqs = np.fft.fftfreq(len(signal), d=1)  # 采样间隔=1天
    
    # 只分析正频率
    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    power = power[positive_mask]
    
    # 转换为周期（天数）
    periods = 1 / freqs
    
    # 过滤有效范围（7-180天）
    valid_mask = (periods >= 7) & (periods <= 180)
    periods = periods[valid_mask]
    power = power[valid_mask]
    
    # 找主导周期
    dominant_idx = np.argmax(power)
    dominant_period = periods[dominant_idx]
    period_strength = power[dominant_idx] / power.sum()
    
    return {
        'dominant_period': float(dominant_period),
        'period_strength': float(period_strength),
        'has_strong_cycle': period_strength > 0.15,
        'all_periods': periods.tolist(),
        'all_power': power.tolist()
    }
```

**数学原理：**
- VWAP是成交量加权的移动平均
- 相当于低频滤波器，自动去除趋势
- 去趋势后的信号更接近平稳过程

---

#### **✅ 正确方法2：对数收益率**

```python
def fourier_with_returns(prices):
    """使用对数收益率的傅立叶分析"""
    
    # 对数收益率（天然平稳）
    returns = np.log(prices[1:] / prices[:-1])
    
    # 加窗
    window_func = np.hanning(len(returns))
    signal = returns * window_func
    
    # FFT（后续同上）
    fft_result = np.fft.fft(signal)
    # ...
```

**优点：**
- 对数收益率天然平稳
- 符合几何布朗运动假设
- 无量纲，可跨资产比较

---

#### **傅立叶在策略中的应用**

```python
# 周期 → DTE映射
if dominant_period < 14:
    # 高频波动
    preferred_dte = [7, 14, 21]
    strategy_hint = "短期震荡"
    
elif 14 <= dominant_period <= 45:
    # 标准周期
    preferred_dte = [
        dominant_period - 7,
        dominant_period,
        dominant_period + 7
    ]
    strategy_hint = "周期匹配"
    
else:
    # 长周期/趋势
    preferred_dte = [30, 45, 60]
    strategy_hint = "中期趋势"

# 趋势+周期 → 策略类型
if trend == 'UP' and cycle_position == 'TROUGH':
    butterfly_type = 'CALL'  # 上涨趋势，短期回调到位
elif trend == 'DOWN' and cycle_position == 'PEAK':
    butterfly_type = 'PUT'   # 下跌趋势，短期反弹到位
else:
    butterfly_type = 'IRON'  # 盘整
```

---

### 2.3 Black-Scholes定价与IV Skew

#### **标准BS公式**

```python
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes期权定价
    
    Args:
        S: 标的现价
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率（年化）
        option_type: 'call' 或 'put'
    """
    if T <= 0:
        # 到期时内在价值
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    if sigma <= 0:
        sigma = 0.01  # 避免除零
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return max(price, 0.01)
```

---

#### **IV Skew的关键性**

**❌ 错误：所有行权价用同一波动率**
```python
# 这会导致20%的定价误差！
sigma = 0.25  # GARCH预测
price_K1 = black_scholes(S, K1, T, r, sigma)
price_K2 = black_scholes(S, K2, T, r, sigma)
price_K3 = black_scholes(S, K3, T, r, sigma)
```

**✅ 正确：根据钱性调整波动率**

```python
def get_iv_skew(ticker, current_price):
    """从真实期权链获取IV Skew"""
    stock = yf.Ticker(ticker)
    expirations = stock.options
    
    if not expirations:
        return estimate_iv_skew()  # fallback
    
    chain = stock.option_chain(expirations[0])
    calls = chain.calls
    
    # ATM IV
    calls['moneyness'] = abs(calls['strike'] - current_price) / current_price
    atm_option = calls.loc[calls['moneyness'].idxmin()]
    iv_atm = float(atm_option['impliedVolatility'])
    
    # OTM Call (5% OTM)
    otm_calls = calls[calls['strike'] > current_price * 1.05]
    if not otm_calls.empty:
        iv_otm_call = float(otm_calls.iloc[0]['impliedVolatility'])
    else:
        iv_otm_call = iv_atm * 0.95  # 典型Call侧低5%
    
    # OTM Put (5% OTM)
    puts = chain.puts
    otm_puts = puts[puts['strike'] < current_price * 0.95]
    if not otm_puts.empty:
        iv_otm_put = float(otm_puts.iloc[-1]['impliedVolatility'])
    else:
        iv_otm_put = iv_atm * 1.10  # 典型Put侧高10%
    
    return {
        'atm': iv_atm,
        'otm_call': iv_otm_call,
        'otm_put': iv_otm_put,
        'skew_call': (iv_otm_call - iv_atm) / iv_atm * 100,
        'skew_put': (iv_otm_put - iv_atm) / iv_atm * 100
    }

def get_sigma_for_strike(strike, current_price, iv_skew):
    """根据行权价钱性返回对应的波动率"""
    moneyness = strike / current_price
    
    if moneyness < 0.95:  # OTM Put区域
        return iv_skew['otm_put']
    elif moneyness > 1.05:  # OTM Call区域
        return iv_skew['otm_call']
    else:  # ATM区域
        return iv_skew['atm']
```

**实际影响示例：**
```
Long Call Butterfly: K1=$470, K2=$480, K3=$490
当前价格 S=$480

不考虑Skew（错误）：
  σ = 25% (统一)
  BS(470) = $12.50
  BS(480) = $8.00
  BS(490) = $4.50
  净成本 = 12.50 - 16.00 + 4.50 = $1.00

考虑Skew（正确）：
  σ(470) = 26% (轻微ITM，IV略高)
  σ(480) = 25% (ATM)
  σ(490) = 24% (OTM Call，IV低)
  BS(470) = $13.00
  BS(480) = $8.00
  BS(490) = $4.20
  净成本 = 13.00 - 16.00 + 4.20 = $1.20

误差 = 20%！
```

---

### 2.4 Greeks计算

```python
def calculate_greeks(S, K, T, r, sigma):
    """计算单个期权的Greeks"""
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Delta
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    
    # Gamma（Call和Put相同）
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega（每1%波动率变化的价格变化）
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Theta（每日时间价值衰减）
    theta_call = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
        r * K * np.exp(-r*T) * norm.cdf(d2)
    ) / 365
    
    theta_put = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
        r * K * np.exp(-r*T) * norm.cdf(-d2)
    ) / 365
    
    return {
        'delta_call': delta_call,
        'delta_put': delta_put,
        'gamma': gamma,
        'vega': vega,
        'theta_call': theta_call,
        'theta_put': theta_put
    }

def butterfly_greeks(S, strikes, T, r, sigmas):
    """计算蝴蝶组合的Greeks
    
    组合：+1 K1, -2 K2, +1 K3
    """
    g1 = calculate_greeks(S, strikes[0], T, r, sigmas[0])
    g2 = calculate_greeks(S, strikes[1], T, r, sigmas[1])
    g3 = calculate_greeks(S, strikes[2], T, r, sigmas[2])
    
    return {
        'delta': g1['delta_call'] - 2*g2['delta_call'] + g3['delta_call'],
        'gamma': g1['gamma'] - 2*g2['gamma'] + g3['gamma'],
        'vega': g1['vega'] - 2*g2['vega'] + g3['vega'],
        'theta': g1['theta_call'] - 2*g2['theta_call'] + g3['theta_call']
    }
```

**理想蝴蝶的Greeks特征：**
```
Delta ≈ 0      # 方向中性
Gamma > 0      # 在K2附近Gamma为正
Vega < 0       # 做空波动率
Theta > 0      # 正时间价值衰减（每天+$0.05~$0.15）
```

---

## 三、关键技术问题

### 3.1 理论定价 vs 市场定价

**核心矛盾：**
- BS模型：理想假设（恒定σ、无成本、连续交易）
- 真实市场：流动性约束、价差、IV Skew

**解决方案：动态加权**

```python
def hybrid_pricing(bs_price, market_price, liquidity_score):
    """
    混合定价策略
    
    Args:
        bs_price: Black-Scholes理论价格
        market_price: 真实市场价格（mid price）
        liquidity_score: 流动性评分 [0, 1]
    """
    deviation_pct = abs(market_price - bs_price) / bs_price * 100
    
    if deviation_pct < 10:
        # 正常范围，完全相信市场
        return market_price
    
    elif deviation_pct < 20:
        # 警惕区域，加权平均
        w_market = liquidity_score
        w_bs = 1 - liquidity_score
        return w_market * market_price + w_bs * bs_price
    
    else:
        # 严重偏差，优先怀疑数据质量
        print(f"警告：定价偏差{deviation_pct:.1f}%，请检查数据")
        # 流动性好→相信市场，流动性差→相信模型
        return market_price if liquidity_score > 0.7 else bs_price
```

---

### 3.2 流动性评估

```python
def assess_liquidity(option_data):
    """
    流动性综合评分
    
    Returns:
        score: [0, 1]
        tier: 1-4级
        executable: bool
    """
    bid = option_data['bid']
    ask = option_data['ask']
    volume = option_data['volume']
    open_interest = option_data['openInterest']
    
    # 价差百分比
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid * 100 if mid > 0 else 100
    
    # 流动性分级
    if spread_pct < 5 and volume > 500:
        tier = 1  # 优秀
        score = 1.0
    elif spread_pct < 10 and volume > 200:
        tier = 2  # 良好
        score = 0.7
    elif spread_pct < 15 and volume > 100:
        tier = 3  # 可接受
        score = 0.4
    else:
        tier = 4  # 拒绝
        score = 0.0
    
    executable = tier <= 3
    
    return {
        'score': score,
        'tier': tier,
        'executable': executable,
        'spread_pct': spread_pct,
        'volume': volume,
        'open_interest': open_interest
    }
```

---

## 四、数据获取方案

### 4.1 yfinance能提供的数据

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")

# 1. 历史价格（日级）✅
price_data = ticker.history(period="1y", interval="1d")
# 包含：Open, High, Low, Close, Volume

# 2. 分钟级数据（最近7天）✅
intraday_data = ticker.history(period="7d", interval="1m")

# 3. 期权链（当前快照）✅
expirations = ticker.options
option_chain = ticker.option_chain(expirations[0])
# 包含：strike, bid, ask, lastPrice, volume, openInterest, impliedVolatility

# 4. 无风险利率（间接）✅
treasury = yf.Ticker("^IRX")  # 13周国债
rf_rate = treasury.history(period="1d")['Close'].iloc[-1] / 100
```

**❌ 无法获取：**
- 历史期权链数据
- 历史Bid-Ask Spread
- 逐笔Tick数据
- 历史IV曲面

---

### 4.2 VWAP计算

```python
def calculate_vwap(ticker, date, use_intraday=False):
    """
    计算VWAP
    
    Args:
        ticker: 股票代码
        date: 目标日期
        use_intraday: 是否使用分钟级数据（更精确但仅限7天内）
    """
    stock = yf.Ticker(ticker)
    
    if use_intraday and (pd.Timestamp.now() - date).days <= 7:
        # 分钟级VWAP（精确）
        df = stock.history(period='7d', interval='1m')
        df = df[df.index.date == date.date()]
        
        df['PV'] = df['Close'] * df['Volume']
        df['VWAP'] = df['PV'].cumsum() / df['Volume'].cumsum()
        
        return df['VWAP'].iloc[-1]
    
    else:
        # 日级VWAP（近似）
        df = stock.history(start=date - pd.Timedelta(days=30), end=date)
        
        # Typical Price = (High + Low + Close) / 3
        df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (
            (df['TypicalPrice'] * df['Volume']).cumsum() / 
            df['Volume'].cumsum()
        )
        
        return df['VWAP'].iloc[-1]
```

---

### 4.3 合成历史期权链

由于yfinance只提供当前期权链快照，回测需要合成历史数据：

```python
class HistoricalOptionChainSynthesizer:
    """历史期权链合成器"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        # 预先收集真实市场统计
        self.spread_distributions = self.collect_spread_stats()
        self.liquidity_stats = self.collect_liquidity_stats()
    
    def collect_spread_stats(self):
        """从当前期权链收集Bid-Ask Spread统计"""
        expirations = self.stock.options[:4]  # 前4个到期日
        spread_data = {'OTM': [], 'ATM': [], 'ITM': []}
        
        for exp in expirations:
            chain = self.stock.option_chain(exp)
            current_price = self.stock.history(period='1d')['Close'].iloc[-1]
            
            for opt_type in ['calls', 'puts']:
                df = getattr(chain, opt_type)
                df['spread_pct'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid'])/2) * 100
                df['moneyness'] = df['strike'] / current_price
                
                # 分类
                for _, row in df.iterrows():
                    m = row['moneyness']
                    if m < 0.95:
                        category = 'OTM'
                    elif m < 1.05:
                        category = 'ATM'
                    else:
                        category = 'ITM'
                    
                    if row['spread_pct'] > 0 and row['spread_pct'] < 50:
                        spread_data[category].append(row['spread_pct'])
        
        # 拟合分布
        distributions = {}
        for category, spreads in spread_data.items():
            if len(spreads) > 10:
                distributions[category] = {
                    'mean': np.mean(spreads),
                    'std': np.std(spreads)
                }
        
        return distributions
    
    def synthesize_chain(self, historical_date, underlying_price, dte):
        """
        为历史某天合成期权链
        
        Args:
            historical_date: 历史日期
            underlying_price: 当天股价
            dte: 到期天数
        """
        # 生成行权价
        strikes = self.generate_strikes(underlying_price)
        
        # 历史波动率
        historical_vol = self.get_historical_volatility(historical_date)
        
        # 无风险利率
        rf_rate = self.get_risk_free_rate(historical_date)
        
        # 合成IV Skew
        iv_skew = self.estimate_iv_skew(historical_vol)
        
        synthetic_chain = []
        
        for strike in strikes:
            moneyness = strike / underlying_price
            
            # 确定钱性类别
            if moneyness < 0.95:
                category = 'OTM'
            elif moneyness < 1.05:
                category = 'ATM'
            else:
                category = 'ITM'
            
            # 获取对应的波动率
            sigma = self.get_sigma_for_moneyness(moneyness, iv_skew)
            
            # BS定价
            call_price = black_scholes(
                S=underlying_price,
                K=strike,
                T=dte/365,
                r=rf_rate,
                sigma=sigma,
                option_type='call'
            )
            
            put_price = black_scholes(
                S=underlying_price,
                K=strike,
                T=dte/365,
                r=rf_rate,
                sigma=sigma,
                option_type='put'
            )
            
            # 合成Bid-Ask Spread
            spread_pct = self.sample_spread(category)
            
            call_bid = call_price * (1 - spread_pct/200)
            call_ask = call_price * (1 + spread_pct/200)
            put_bid = put_price * (1 - spread_pct/200)
            put_ask = put_price * (1 + spread_pct/200)
            
            # 合成流动性
            volume = max(int(np.random.lognormal(5, 1)), 0)
            oi = max(int(np.random.lognormal(6, 1)), 0)
            
            synthetic_chain.append({
                'strike': strike,
                'call_bid': call_bid,
                'call_ask': call_ask,
                'call_last': call_price,
                'call_volume': volume,
                'call_oi': oi,
                'put_bid': put_bid,
                'put_ask': put_ask,
                'put_last': put_price,
                'put_volume': volume,
                'put_oi': oi,
                'impliedVolatility': sigma
            })
        
        return pd.DataFrame(synthetic_chain)
    
    def generate_strikes(self, price):
        """生成行权价网格"""
        strikes = []
        for i in range(-10, 11):
            strike = price * (1 + i * 0.05)
            strikes.append(round(strike / 5) * 5)
        return sorted(set(strikes))
    
    def get_historical_volatility(self, date, window=30):
        """计算历史波动率"""
        end = date
        start = date - pd.Timedelta(days=window+10)
        df = self.stock.history(start=start, end=end)
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        return returns.std() * np.sqrt(252)
    
    def estimate_iv_skew(self, atm_vol):
        """估计IV Skew（如果无真实数据）"""
        return {
            'atm': atm_vol,
            'otm_call': atm_vol * 0.95,
            'otm_put': atm_vol * 1.10
        }
    
    def sample_spread(self, category):
        """从分布中采样Spread"""
        if category in self.spread_distributions:
            mean = self.spread_distributions[category]['mean']
            std = self.spread_distributions[category]['std']
            spread = np.random.normal(mean, std)
            return np.clip(spread, 2, 50)
        return {'OTM': 10, 'ATM': 5, 'ITM': 7}[category]
```

---

## 五、完整实现框架

### 5.1 核心分析类

```python
class ButterflyAnalyzer:
    """蝴蝶策略完整分析器"""
    
    def __init__(self, ticker, days=180):
        self.ticker = ticker
        self.days = days
        self.stock = yf.Ticker(ticker)
        self.data = None
        self.prices = None
        
    def fetch_data(self):
        """获取基础数据"""
        self.data = self.stock.history(period=f"{self.days}d")
        self.prices = self.data['Close'].values
        
    def full_analysis(self):
        """完整分析流程"""
        self.fetch_data()
        
        # 1. 傅立叶分析
        fourier_result = self.fourier_analysis()
        
        # 2. ARIMA预测
        arima_result = self.arima_forecast()
        
        # 3. GARCH波动率
        garch_result = self.garch_volatility()
        
        # 4. 设计蝴蝶策略
        butterfly = self.design_butterfly(
            forecast_price=arima_result['mean_forecast'],
            volatility=garch_result['predicted_vol'],
            iv_skew=garch_result['iv_skew']
        )
        
        # 5. 综合评分
        score = self.calculate_score(
            fourier_result,
            arima_result,
            garch_result,
            butterfly
        )
        
        # 6. 风险评估
        risk_assessment = self.assess_risk(
            arima_result,
            garch_result,
            butterfly
        )
        
        return {
            'ticker': self.ticker,
            'current_price': float(self.prices[-1]),
            'fourier': fourier_result,
            'arima': arima_result,
            'garch': garch_result,
            'butterfly': butterfly,
            'score': score,
            'risk': risk_assessment
        }
    
    def fourier_analysis(self):
        """傅立叶分析（使用VWAP去趋势）"""
        volumes = self.data['Volume'].values
        
        # VWAP去趋势
        window = min(20, len(self.prices) // 3)
        pv = self.prices * volumes
        cumsum_pv = pd.Series(pv).rolling(window).sum()
        cumsum_v = pd.Series(volumes).rolling(window).sum()
        vwap = (cumsum_pv / cumsum_v).fillna(method='bfill').values
        
        detrended = self.prices - vwap
        detrended = detrended[~np.isnan(detrended)]
        
        # FFT
        window_func = np.hanning(len(detrended))
        signal = detrended * window_func
        
        fft_result = np.fft.fft(signal)
        power = np.abs(fft_result) ** 2
        freqs = np.fft.fftfreq(len(signal), d=1)
        
        # 正频率
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power = power[pos_mask]
        periods = 1 / freqs
        
        # 有效范围
        valid_mask = (periods >= 7) & (periods <= 180)
        periods = periods[valid_mask]
        power = power[valid_mask]
        
        # 主导周期
        if len(power) > 0:
            dominant_idx = np.argmax(power)
            dominant_period = periods[dominant_idx]
            period_strength = power[dominant_idx] / power.sum()
        else:
            dominant_period = 30
            period_strength = 0
        
        # 趋势判断（低频分量）
        low_freq_component = self._extract_low_freq(detrended)
        trend_slope = np.polyfit(range(len(low_freq_component)), low_freq_component, 1)[0]
        
        if trend_slope > 0.1:
            trend_direction = 'UP'
        elif trend_slope < -0.1:
            trend_direction = 'DOWN'
        else:
            trend_direction = 'FLAT'
        
        # 周期位置
        mid_freq_component = self._extract_mid_freq(detrended)
        cycle_position = 'PEAK' if np.mean(mid_freq_component[-5:]) > 0 else 'TROUGH'
        
        # 策略类型
        if trend_direction == 'UP' and cycle_position == 'TROUGH':
            butterfly_type = 'CALL'
        elif trend_direction == 'DOWN' and cycle_position == 'PEAK':
            butterfly_type = 'PUT'
        else:
            butterfly_type = 'IRON'
        
        return {
            'dominant_period': float(dominant_period),
            'period_strength': float(period_strength),
            'trend_direction': trend_direction,
            'cycle_position': cycle_position,
            'butterfly_type': butterfly_type,
            'low_freq': low_freq_component.tolist(),
            'mid_freq': mid_freq_component.tolist()
        }
    
    def arima_forecast(self, steps=30):
        """ARIMA预测（自动选参）"""
        train_data = self.prices[-120:]  # 120天训练
        
        # 候选参数
        candidates = [(1,1,1), (2,1,2), (1,1,2), (2,1,1)]
        
        best_aic = np.inf
        best_model = None
        
        for order in candidates:
            try:
                model = ARIMA(train_data, order=order)
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
            except:
                continue
        
        if best_model is None:
            # Fallback
            return {
                'mean_forecast': float(self.prices[-1]),
                'upper_bound': [self.prices[-1] * 1.05] * steps,
                'lower_bound': [self.prices[-1] * 0.95] * steps
            }
        
        # 预测
        forecast_result = best_model.get_forecast(steps=steps)
        forecast_df = forecast_result.summary_frame(alpha=0.05)
        
        return {
            'forecast': forecast_df['mean'].values.tolist(),
            'upper_bound': forecast_df['mean_ci_upper'].values.tolist(),
            'lower_bound': forecast_df['mean_ci_lower'].values.tolist(),
            'mean_forecast': float(forecast_df['mean'].mean()),
            'model_order': best_model.model_order,
            'aic': float(best_aic)
        }
    
    def garch_volatility(self):
        """GARCH波动率预测"""
        returns = pd.Series(self.prices).pct_change().dropna() * 100
        
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1)
            fitted = model.fit(disp='off')
            
            forecast = fitted.forecast(horizon=30)
            predicted_vol = np.sqrt(forecast.variance.values[-1, :])
            predicted_vol_annual = predicted_vol / 100 * np.sqrt(252)
            
            # 获取真实IV
            iv_skew = get_iv_skew(self.ticker, self.prices[-1])
            
            # 波动率错误定价
            vol_mispricing = (
                (iv_skew['atm'] - np.mean(predicted_vol_annual)) / 
                iv_skew['atm'] * 100
            )
            
            return {
                'predicted_vol': float(np.mean(predicted_vol_annual)),
                'current_iv': iv_skew['atm'],
                'iv_skew': iv_skew,
                'vol_mispricing': float(vol_mispricing),
                'garch_params': {
                    'omega': float(fitted.params['omega']),
                    'alpha': float(fitted.params['alpha[1]']),
                    'beta': float(fitted.params['beta[1]'])
                }
            }
        except Exception as e:
            print(f"GARCH错误: {e}")
            return {
                'predicted_vol': 0.25,
                'current_iv': 0.25,
                'iv_skew': estimate_iv_skew(0.25),
                'vol_mispricing': 0
            }
    
    def design_butterfly(self, forecast_price, volatility, iv_skew):
        """设计蝴蝶策略"""
        current_price = self.prices[-1]
        
        # 行权价间隔
        if current_price < 50:
            strike_step = 2.5
        elif current_price < 200:
            strike_step = 5
        else:
            strike_step = 10
        
        # 中心行权价
        center_strike = round(forecast_price / strike_step) * strike_step
        
        # 翼宽
        wing_width = strike_step * 2  # 默认2个间隔
        
        lower_strike = center_strike - wing_width
        upper_strike = center_strike + wing_width
        
        # DTE
        T = 30 / 365
        r = get_risk_free_rate()
        
        # 根据IV Skew定价
        sigma_lower = get_sigma_for_strike(lower_strike, current_price, iv_skew)
        sigma_center = get_sigma_for_strike(center_strike, current_price, iv_skew)
        sigma_upper = get_sigma_for_strike(upper_strike, current_price, iv_skew)
        
        # BS定价
        lower_call = black_scholes(current_price, lower_strike, T, r, sigma_lower, 'call')
        center_call = black_scholes(current_price, center_strike, T, r, sigma_center, 'call')
        upper_call = black_scholes(current_price, upper_strike, T, r, sigma_upper, 'call')
        
        # 加入Bid-Ask Spread
        spread_pct = 0.06  # 假设6%
        lower_cost = lower_call * (1 + spread_pct/2)
        center_credit = center_call * (1 - spread_pct/2)
        upper_cost = upper_call * (1 + spread_pct/2)
        
        net_debit = lower_cost - 2*center_credit + upper_cost
        max_profit = wing_width - net_debit
        
        # Greeks
        greeks = butterfly_greeks(
            current_price,
            [lower_strike, center_strike, upper_strike],
            T, r,
            [sigma_lower, sigma_center, sigma_upper]
        )
        
        return {
            'center_strike': float(center_strike),
            'lower_strike': float(lower_strike),
            'upper_strike': float(upper_strike),
            'wing_width': float(wing_width),
            'net_debit': max(0.5, float(net_debit)),
            'max_profit': max(0.5, float(max_profit)),
            'max_loss': max(0.5, float(net_debit)),
            'profit_ratio': float(max_profit / max(0.5, net_debit)),
            'breakeven_lower': float(lower_strike + net_debit),
            'breakeven_upper': float(upper_strike - net_debit),
            'dte': 30,
            'greeks': greeks
        }
```

---

## 六、评分与风险管理

### 6.1 综合评分系统

```python
def calculate_score(self, fourier, arima, garch, butterfly):
    """
    多因子综合评分（0-100）
    
    Score = Σ(w_i × factor_i) - Penalties
    """
    
    # 因子1：价格预测匹配度（35%）
    forecast_center_diff = abs(
        arima['mean_forecast'] - butterfly['center_strike']
    )
    price_match_score = max(
        0, 
        100 - (forecast_center_diff / arima['mean_forecast'] * 500)
    )
    
    # 因子2：波动率错误定价（30%）
    vol_score = min(100, abs(garch['vol_mispricing']) * 5)
    
    # 因子3：价格稳定性（20%）
    price_range = (
        max(arima['upper_bound']) - min(arima['lower_bound'])
    )
    stability = price_range / arima['mean_forecast'] * 100
    stability_score = max(0, 100 - stability * 5)
    
    # 因子4：傅立叶周期对齐（15%）
    if (fourier['butterfly_type'] == 'CALL' and 
        fourier['trend_direction'] == 'UP'):
        fourier_score = 100
    elif (fourier['butterfly_type'] == 'PUT' and 
          fourier['trend_direction'] == 'DOWN'):
        fourier_score = 100
    elif (fourier['butterfly_type'] == 'IRON' and 
          fourier['trend_direction'] == 'FLAT'):
        fourier_score = 100
    else:
        fourier_score = 50
    
    # 综合评分
    total_score = (
        price_match_score * 0.35 +
        vol_score * 0.30 +
        stability_score * 0.20 +
        fourier_score * 0.15
    )
    
    # Greeks惩罚
    delta_penalty = min(10, abs(butterfly['greeks']['delta']) * 50)
    total_score -= delta_penalty
    
    # 推荐等级
    if total_score >= 75 and butterfly['profit_ratio'] > 2:
        recommendation = 'STRONG_BUY'
    elif total_score >= 60 and butterfly['profit_ratio'] > 1.5:
        recommendation = 'BUY'
    elif total_score >= 45:
        recommendation = 'NEUTRAL'
    else:
        recommendation = 'AVOID'
    
    return {
        'total': round(total_score, 1),
        'components': {
            'price_match': round(price_match_score, 1),
            'vol_mispricing': round(vol_score, 1),
            'stability': round(stability_score, 1),
            'fourier_align': round(fourier_score, 1)
        },
        'delta_penalty': round(delta_penalty, 1),
        'recommendation': recommendation
    }
```

---

### 6.2 风险管理框架

```python
def assess_risk(self, arima, garch, butterfly):
    """全面风险评估"""
    
    # 基础风险等级
    price_range = (
        max(arima['upper_bound']) - min(arima['lower_bound'])
    )
    stability = price_range / arima['mean_forecast'] * 100
    
    if stability < 8 and garch['vol_mispricing'] > 15:
        base_risk = 'LOW'
    elif stability < 15 and garch['vol_mispricing'] > 5:
        base_risk = 'MEDIUM'
    else:
        base_risk = 'HIGH'
    
    # Greeks调整
    greeks = butterfly['greeks']
    
    if abs(greeks['delta']) > 0.15:
        base_risk = upgrade_risk(base_risk)
    
    if greeks['vega'] > -0.5:
        base_risk = upgrade_risk(base_risk)
    
    # IV百分位检查
    iv_percentile = self.calculate_iv_percentile(garch['current_iv'])
    if iv_percentile < 50:
        base_risk = upgrade_risk(base_risk)
    
    # 仓位建议
    if base_risk == 'LOW' and butterfly['profit_ratio'] > 2:
        position_size = '3-5%'
        stop_loss = -0.5 * butterfly['net_debit']
        take_profit = 0.7 * butterfly['max_profit']
    elif base_risk == 'MEDIUM':
        position_size = '2-3%'
        stop_loss = -0.4 * butterfly['net_debit']
        take_profit = 0.6 * butterfly['max_profit']
    else:
        position_size = '1-2%'
        stop_loss = -0.3 * butterfly['net_debit']
        take_profit = 0.5 * butterfly['max_profit']
    
    return {
        'risk_level': base_risk,
        'position_size': position_size,
        'stop_loss': float(stop_loss),
        'take_profit': float(take_profit),
        'time_stop': 7,  # DTE < 7天强制平仓
        'vol_stop': 0.30,  # IV飙升30%退出
        'warnings': self.generate_warnings(
            greeks, iv_percentile, stability
        )
    }

def generate_warnings(self, greeks, iv_percentile, stability):
    """生成风险警告"""
    warnings = []
    
    if abs(greeks['delta']) > 0.10:
        warnings.append(
            f"Delta={greeks['delta']:.3f}，非完全中性，存在方向性风险"
        )
    
    if iv_percentile < 30:
        warnings.append(
            f"IV处于历史{iv_percentile:.0f}%分位，波动率可能上升"
        )
    
    if stability > 15:
        warnings.append(
            f"价格预测区间宽度{stability:.1f}%，不确定性较高"
        )
    
    if greeks['vega'] > -0.3:
        warnings.append(
            "Vega不够负，对波动率上升敏感度不足"
        )
    
    return warnings
```

---

## 七、回测框架设计

### 7.1 滑点建模

```python
class SlippageModel:
    """滑点综合模型"""
    
    @staticmethod
    def calculate_total_slippage(
        order_size,
        avg_volume,
        volatility,
        spread_pct,
        side  # 'buy' or 'sell'
    ):
        """
        Total_Slippage = Fixed_Spread + Market_Impact
        
        Args:
            order_size: 下单数量
            avg_volume: 日均成交量
            volatility: 当前波动率
            spread_pct: Bid-Ask价差百分比
            side: 'buy' 或 'sell'
        """
        
        # 1. Fixed Spread
        fixed_spread = spread_pct / 200  # 除以2取半个价差
        
        # 2. Market Impact（Kyle's Lambda模型）
        lambda_coef = 0.10  # 期权市场冲击系数
        vol_factor = volatility / 0.25  # 归一化
        size_ratio = order_size / max(avg_volume, 1)
        
        market_impact = (
            lambda_coef * 
            np.sqrt(size_ratio) * 
            vol_factor
        )
        market_impact = min(market_impact, 0.20)  # 上限20%
        
        # 总滑点
        total_slippage = fixed_spread + market_impact
        
        # 方向
        if side == 'buy':
            return total_slippage  # 买入付出更多
        else:
            return -total_slippage  # 卖出收到更少

class ButterflyBacktest:
    """蝴蝶策略回测框架"""
    
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.dates = pd.date_range(start_date, end_date, freq='D')
        self.analyzer = ButterflyAnalyzer(ticker)
        self.synthesizer = HistoricalOptionChainSynthesizer(ticker)
        
        # 预加载数据
        self.price_history = yf.download(ticker, start=start_date, end=end_date)
        
    def run(self):
        """运行回测"""
        portfolio = []
        equity_curve = [10000]  # 初始资金
        
        for date in self.dates:
            if date not in self.price_history.index:
                continue
            
            current_price = self.price_history.loc[date, 'Close']
            
            # 1. 运行分析
            self.analyzer.prices = self.price_history.loc[:date, 'Close'].values
            analysis = self.analyzer.full_analysis()
            
            # 2. 评分筛选
            if analysis['score']['total'] < 60:
                continue  # 评分不够，跳过
            
            # 3. 合成期权链
            option_chain = self.synthesizer.synthesize_chain(
                historical_date=date,
                underlying_price=current_price,
                dte=30
            )
            
            # 4. 流动性检查
            butterfly = analysis['butterfly']
            strikes = [
                butterfly['lower_strike'],
                butterfly['center_strike'],
                butterfly['upper_strike']
            ]
            
            if not self.check_liquidity(option_chain, strikes):
                continue
            
            # 5. 模拟执行（含滑点）
            execution_result = self.simulate_execution(
                butterfly,
                option_chain,
                date
            )
            
            # 6. 加入组合
            portfolio.append({
                'entry_date': date,
                'butterfly': butterfly,
                'entry_cost': execution_result['total_cost'],
                'dte': 30,
                'exit_date': None,
                'exit_value': None,
                'pnl': 0
            })
            
            # 7. 持仓管理
            portfolio = self.manage_positions(
                portfolio,
                date,
                current_price,
                option_chain
            )
            
            # 8. 计算权益
            total_pnl = sum([p['pnl'] for p in portfolio])
            equity_curve.append(equity_curve[0] + total_pnl)
        
        return self.calculate_metrics(equity_curve, portfolio)
    
    def simulate_execution(self, butterfly, option_chain, date):
        """模拟执行（含滑点）"""
        strikes = [
            butterfly['lower_strike'],
            butterfly['center_strike'],
            butterfly['upper_strike']
        ]
        
        total_cost = 0
        slippage_model = SlippageModel()
        
        for i, strike in enumerate(strikes):
            option = option_chain[option_chain['strike'] == strike].iloc[0]
            
            # 理论价格（mid）
            mid_price = (option['call_bid'] + option['call_ask']) / 2
            
            # 计算滑点
            spread_pct = (option['call_ask'] - option['call_bid']) / mid_price * 100
            
            slippage = slippage_model.calculate_total_slippage(
                order_size=2 if i == 1 else 1,  # 中间腿2份
                avg_volume=option['call_volume'],
                volatility=option['impliedVolatility'],
                spread_pct=spread_pct,
                side='buy' if i != 1 else 'sell'
            )
            
            # 实际成交价
            if i == 1:  # 卖出中间腿
                execution_price = mid_price * (1 - slippage)
                total_cost -= 2 * execution_price
            else:  # 买入两翼
                execution_price = mid_price * (1 + slippage)
                total_cost += execution_price
        
        return {
            'total_cost': total_cost,
            'slippage_impact': total_cost - butterfly['net_debit']
        }
    
    def check_liquidity(self, option_chain, strikes):
        """流动性检查"""
        for strike in strikes:
            option = option_chain[option_chain['strike'] == strike]
            if option.empty:
                return False
            
            option = option.iloc[0]
            
            # 流动性标准
            if option['call_volume'] < 100:
                return False
            
            spread_pct = (
                (option['call_ask'] - option['call_bid']) / 
                ((option['call_ask'] + option['call_bid']) / 2) * 100
            )
            
            if spread_pct > 15:
                return False
        
        return True
    
    def manage_positions(self, portfolio, current_date, current_price, option_chain):
        """持仓管理（止损/止盈/到期）"""
        for position in portfolio:
            if position['exit_date'] is not None:
                continue  # 已平仓
            
            # 计算持仓时间
            days_held = (current_date - position['entry_date']).days
            dte = position['dte'] - days_held
            
            # 到期平仓
            if dte <= 0:
                position['exit_date'] = current_date
                position['exit_value'] = self.calculate_expiry_value(
                    position['butterfly'],
                    current_price
                )
                position['pnl'] = position['exit_value'] - position['entry_cost']
                continue
            
            # 时间止损
            if dte < 7:
                position['exit_date'] = current_date
                position['exit_value'] = self.estimate_current_value(
                    position['butterfly'],
                    current_price,
                    dte,
                    option_chain
                )
                position['pnl'] = position['exit_value'] - position['entry_cost']
                continue
            
            # 价格止损/止盈
            current_value = self.estimate_current_value(
                position['butterfly'],
                current_price,
                dte,
                option_chain
            )
            unrealized_pnl = current_value - position['entry_cost']
            
            # 止损：亏损50%
            if unrealized_pnl < -0.5 * position['entry_cost']:
                position['exit_date'] = current_date
                position['exit_value'] = current_value
                position['pnl'] = unrealized_pnl
            
            # 止盈：达到最大收益的70%
            elif unrealized_pnl > 0.7 * position['butterfly']['max_profit']:
                position['exit_date'] = current_date
                position['exit_value'] = current_value
                position['pnl'] = unrealized_pnl
        
        return portfolio
    
    def calculate_expiry_value(self, butterfly, final_price):
        """到期时的内在价值"""
        K1 = butterfly['lower_strike']
        K2 = butterfly['center_strike']
        K3 = butterfly['upper_strike']
        
        if final_price <= K1:
            return 0
        elif final_price <= K2:
            return final_price - K1
        elif final_price <= K3:
            return K3 - final_price
        else:
            return 0
    
    def calculate_metrics(self, equity_curve, portfolio):
        """计算回测指标"""
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # 总收益
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # 夏普比率
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0 else 0
        )
        
        # 最大回撤
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 胜率
        closed_positions = [p for p in portfolio if p['exit_date'] is not None]
        wins = [p for p in closed_positions if p['pnl'] > 0]
        win_rate = len(wins) / len(closed_positions) if closed_positions else 0
        
        # 盈利因子
        gross_profit = sum([p['pnl'] for p in wins])
        gross_loss = abs(sum([p['pnl'] for p in closed_positions if p['pnl'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(closed_positions),
            'equity_curve': equity_curve
        }
```

---

## 八、实战指南

### 8.1 实施优先级

**P0 - 立即实现（核心功能）：**

1. ✅ 傅立叶去趋势（VWAP方法）
2. ✅ Black-Scholes精确定价
3. ✅ IV Skew调整
4. ✅ Greeks计算
5. ✅ 综合评分系统

**P1 - 重要改进（1-2周）：**

1. ⏳ 回测框架（合成数据）
2. ⏳ 滑点建模（三因子）
3. ⏳ 流动性过滤
4. ⏳ ARIMA自动选参

**P2 - 锦上添花（1-2月）：**

1. 🔮 ML增强（XGBoost）
2. 🔮 多策略组合
3. 🔮 实时监控
4. 🔮 个性化推荐

---

### 8.2 关键Insights总结

**定价原则：**
- 市场价格 > 理论价格（市场反映真实供需）
- IV Skew不可忽视（不同行权价用不同σ）
- 流动性是硬约束（价格再好也要能交易）

**傅立叶分析：**
- 必须去趋势（VWAP或收益率）
- 用于检测而非预测（识别周期和机构行为）
- 权重适中（15%，辅助决策）

**回测设计：**
- 现实主义（接受数据限制，用合成补充）
- 保守估计（高估滑点好于过拟合）
- 流动性优先（评分再高也要能执行）

**风险管理：**
- 仓位控制（评分>75才3-5%）
- 多重止损（价格+时间+波动率）
- Greeks监控（Delta偏离立即调整）

---

### 8.3 核心决策公式

```python
最优蝴蝶策略 = argmax {
    Score(K1, K2, K3) = 
        0.35 × [100 - |ARIMA预测 - K2| / K2 × 500] +
        0.30 × [min(100, (IV - σ_GARCH) / IV × 500)] +
        0.20 × [100 - (CI宽度 / 预测值) × 500] +
        0.15 × Fourier_Alignment
        - Delta_Penalty
}

约束条件：
1. K2 ∈ [ARIMA预测 ± 1.5σ]
2. NetDebit > 0
3. Bid-Ask Spread < 10%
4. Volume > 100
5. DTE ∈ [21, 45]
6. |Delta| < 0.10
7. Vega < 0
8. IV_percentile > 50%

风险管理：
- 止损：-50%成本
- 止盈：+70%最大收益
- 时间止损：DTE < 7天
- 波动率止损：IV飙升>30%
```

---

### 8.4 模型局限性与改进方向

**当前局限：**

1. **数据限制**：yfinance只有当前期权链快照
2. **单一标的**：未考虑组合对冲
3. **静态策略**：未实现动态调整
4. **简化IV**：Skew模型可以更精细

**未来改进：**

1. **引入ML**：XGBoost预测策略成功率
2. **多策略**：同时运行不同DTE的蝴蝶
3. **实时监控**：Greeks实时跟踪与预警
4. **个性化**：根据用户风险偏好调整

---

## 结语

这个完整的蝴蝶策略量化模型整合了：

- **时间序列分析**（ARIMA/GARCH）
- **频域分析**（傅立叶变换）
- **期权定价理论**（Black-Scholes + IV Skew）
- **风险管理**（Greeks + 多因子评分）
- **回测验证**（滑点建模 + 现实约束）

核心哲学是：

> **在有限数据和现实约束下，构建一个"足够好"的量化决策系统**

不追求完美预测，而是通过多因子综合评估，识别"高概率"机会，结合严格的风险管理，实现长期稳定收益。

**关键是：可解释、可验证、可优化。**