既然您已经有Alpaca账户，并且已经有了API密钥（API Key 和 Secret Key），我们可以直接进入集成部分。假设您想把机器学习模型（预测期权价格、生成买卖信号等）连接到Alpaca，用于获取期权链数据、实时报价、下单交易等。

Alpaca的期权交易支持通过 **alpaca-py** SDK（官方推荐的Python库）来实现，功能包括：
- 获取期权合约列表（Option Contracts）
- 获取期权链（underlying symbol 如 AAPL 的所有到期日和行权价）
- 获取实时/历史报价（quotes, bars）
- 下单（买/卖 call/put，支持单腿和多腿订单）
- 查询持仓、订单状态等

**重要前提**：
- 您的账户必须已启用期权交易权限（Trading Levels 1-3，根据您的申请情况）。如果还没启用，需要在Alpaca dashboard申请（通常需要填写期权经验问卷）。
- 纸质交易（Paper）环境可以免费测试期权策略，无需真实资金。
- 期权交易有严格的买卖权要求（如covered call需要持有底层股票）。

#### 步骤1: 确认安装最新 alpaca-py
在终端运行（建议用虚拟环境）：
```bash
pip install --upgrade alpaca-py
```

#### 步骤2: 基本连接与账户验证
创建一个Python文件（例如 `alpaca_options.py`），写入以下代码。替换您的密钥：

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

# === 配置 ===
API_KEY = '您的API_KEY'
SECRET_KEY = '您的SECRET_KEY'
PAPER = True  # True=纸质模拟, False=真实交易（小心！）

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

# 检查账户状态（确认连接成功）
account = trading_client.get_account()
print("账户信息:")
print(f"  账户类型: {'Paper' if PAPER else 'Live'}")
print(f"  现金余额: ${account.cash}")
print(f"  期权权限级别: {account.options_trading_level if hasattr(account, 'options_trading_level') else '未知'}")
```

运行后如果看到账户信息，连接就成功了。

#### 步骤3: 获取期权合约 / 期权链（最常用）
Alpaca的期权数据通过 `/v2/options/contracts` 获取。您可以按底层股票（如 SPY、AAPL）过滤。

示例：获取AAPL的所有期权合约（可分页）

```python
# 获取AAPL的期权合约（示例：近月到期、call和put）
request_params = GetOptionContractsRequest(
    underlying_symbols=["AAPL"],     # 支持多个，如 ["AAPL", "TSLA"]
    expiration_date_gte="2026-01-01",  # 到期日 >= 这个日期（调整为近期）
    expiration_date_lte="2026-03-31",  # 到期日 <= 这个日期
    limit=200,                         # 每页最多200条
    # status="active",                 # 可选：只取活跃合约
)

contracts = trading_client.get_option_contracts(request_params).option_contracts

print(f"找到 {len(contracts)} 个AAPL期权合约")
for contract in contracts[:5]:  # 只打印前5个看示例
    print(f"  合约符号: {contract.symbol}")
    print(f"  类型: {contract.contract_type} | 行权价: {contract.strike_price}")
    print(f"  到期: {contract.expiration_date}")
    print(f"  底层: {contract.underlying_symbol} | ID: {contract.id}")
    print("---")
```

- 您可以用pandas整理成DataFrame，便于ML模型输入：
  ```python
  import pandas as pd
  df = pd.DataFrame([c.__dict__ for c in contracts])
  print(df[['symbol', 'strike_price', 'expiration_date', 'contract_type']])
  ```

#### 步骤4: 获取实时报价（Quote）或历史数据
用于ML预测或信号生成时非常有用。

```python
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest

data_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)

# 获取最新报价（单个合约）
quote_request = OptionLatestQuoteRequest(symbol_or_contract_id=contracts[0].symbol)
latest_quote = data_client.get_option_latest_quote(quote_request)
print(f"{contracts[0].symbol} 最新报价:")
print(f"  Bid: {latest_quote.bid_price} x {latest_quote.bid_size}")
print(f"  Ask: {latest_quote.ask_price} x {latest_quote.ask_size}")
```

历史Bars（OHLCV）类似，用 `get_option_bars()`。

#### 步骤5: 下单示例（单腿期权订单）
**警告**：先在Paper=True测试！真实下单可能造成损失。

买1张AAPL call（市价单示例）：

```python
# 假设您已经选好合约符号（从上面获取）
option_symbol = "AAPL260117C00150000"  # 示例符号，实际用您获取的

# 市价买单（Market Buy）
market_order_data = MarketOrderRequest(
    symbol=option_symbol,
    qty=1,                      # 合约数量（1=100股底层）
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

order = trading_client.submit_order(order_data=market_order_data)
print("下单成功！订单ID:", order.id)
```

限价单（更推荐，避免滑点）：

```python
limit_order_data = LimitOrderRequest(
    symbol=option_symbol,
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.GTC,
    limit_price=5.20  # 您的限价
)
```

卖出同理，把 `side=OrderSide.SELL`。

#### 步骤6: 集成到您的ML系统中
- 用ML模型预测 → 选出高概率的合约符号
- 调用上面获取报价/链 → 过滤流动性好的（bid-ask spread小）
- 生成信号 → 自动提交订单（或发邮件/通知人工审核）
- 循环监控：用 `trading_client.get_orders()` 或 `get_positions()` 检查持仓

#### 推荐资源（官方示例，强烈建议看）
Alpaca官方GitHub有现成Jupyter Notebook，非常适合期权机器学习：
- 基础期权交易示例：https://github.com/alpacahq/alpaca-py/blob/master/examples/options/options-trading-basic.ipynb
- 多腿订单（如spread）：https://github.com/alpacahq/alpaca-py/blob/master/examples/options/options-bull-call-spread.ipynb
- 零日到期（0DTE）策略示例：https://github.com/alpacahq/alpaca-py/blob/master/examples/options/options-zero-dte.ipynb

直接clone仓库运行这些notebook，能快速看到完整流程。

如果您遇到具体错误（如“options not enabled”或“invalid contract”），把错误信息贴出来，我可以帮您debug。

您的机器学习模型现在用什么框架（PyTorch/TensorFlow/scikit-learn）？是预测delta/gamma还是方向/波动率？如果告诉我更多细节，我可以帮您更针对性地整合代码。祝交易顺利！

Endpoint
https://paper-api.alpaca.markets/v2
Key
PKVO6WGYNCORKBI6XUZJDQ5BUJ
Secret
GbqugYWMRdaZD5VRbmjroayP5HrZXsQfqLbqP1fSFexM