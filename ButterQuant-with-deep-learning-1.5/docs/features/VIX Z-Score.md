python library https://pypi.org/project/qserver-connect/
Home page https://www.quantconnect.com/research/

利用期权交易波动率

介绍
市场波动周期分为高波动期和低波动期。为了把握当前市场所处的波动周期，您可以利用标普500指数、芝加哥期权交易所波动率指数（VIX）和期权跨式期权策略。本文将详细介绍这些指标及其应用，并指导您如何运用它们制定波动率交易策略。

背景
该策略基于芝加哥期权交易所波动率指数（VIX）交易标普500指数（SPX）期权，以从预期未来波动率与实际未来波动率之间的差异中获利。SPX是美国市值最大的500家上市公司的市值加权指数。它并非美国市值排名前500家公司的精确列表，因为该指数还包含其他指标。该指数被广泛认为是衡量美国大盘股的最佳指标。

VIX指数是一个实时指数，代表市场对标普500指数（SPX）近期价格变动相对强度的预期。由于它源自近期到期的SPX指数期权价格，因此可以预测未来30天的波动率。波动率，即价格变动的速度，通常被视为衡量市场情绪，尤其是市场参与者恐慌程度的一种指标。

长跨式期权策略是一种期权交易策略，它由买入一份平值看涨期权和一份平值看跌期权组成，这两份合约的标的资产、行权价格和到期日均相同。该策略旨在从标的股票价格的波动中获利，无论价格上涨还是下跌。

多头跨式期权策略收益分解与分析
卖出跨式期权是指同时卖出一份看涨期权和一份看跌期权，这两份合约的标的资产、行权价格（通常为平值期权）和到期日均相同。如果您进行卖出跨式期权交易，您押注标的资产在期权到期前将保持相对稳定，不会出现大幅价格波动。

短跨式期权策略收益分解与分析
这种交易策略背后的理念是逆市场预期进行交易。也就是说，如果VIX指数相对较高（市场预期波动性较高），我们就开立空头跨式期权组合，押注即将到来的波动性较低。相反，如果VIX指数相对较低（市场预期波动性较低），我们就开立多头跨式期权组合，押注即将到来的波动性较高。

执行
为了实施这一策略，我们首先订阅 VIX 指数，并创建一些指标，以便我们可以计算 VIX z 分数。

self._vix = self.add_index('VIX')
self._vix.std = self.std(self._vix.symbol, 24*21, resolution=Resolution.DAILY)
self._vix.sma = self.sma(self._vix.symbol, 24*21, resolution=Resolution.DAILY)
然后我们订阅 SPX 指数期权，并将筛选条件设置为选择可以形成跨式期权的合约。

self._spx = self.add_index_option('SPX')
self._spx.set_filter(lambda universe: universe.straddle(30))
为了确保算法能够对最新的市场活动做出反应，我们会在每个市场开盘后 30 分钟重新平衡投资组合。

self.schedule.on(self.date_rules.every_day(self._spx.symbol), self.time_rules.after_market_open(self._spx.symbol, 30), self._trade)
每次再平衡时，我们都会计算VIX的z分数。我们希望仓位规模与z分数呈负相关，因此我们将订单数量设置为z分数的整数倍乘以-1。如果计算结果为零，则平仓。如果结果不为零且我们没有未平仓头寸，则对标普500指数（SPX）建立跨式期权组合。

def _trade(self):        
    z_score = (self._vix.price - self._vix.sma.current.value) / self._vix.std.current.value
    quantity = -int(z_score)
    if quantity == 0:
        self.liquidate()
    elif not self.portfolio.invested:
        chain = self.current_slice.option_chains.get(self._spx.symbol, None)
        if not chain: return
        chain = [c for c in chain if c.expiry > self.time]
        expiry = min([x.expiry for x in chain])
        strike = sorted([c for c in chain if c.expiry == expiry], key=lambda x: abs(x.strike - self._spx.price))[0].strike
        strategy = OptionStrategies.straddle(self._spx.symbol, strike, expiry)
        self.order(strategy, quantity)
展开
由于该数量是z分数的负整数值，因此当VIX指数相对较低时，算法会建立多头跨式期权组合；当VIX指数相对较高时，算法会建立空头跨式期权组合。多头跨式期权组合在建仓后标的资产波动较大时获利，而空头跨式期权组合在建仓后标的资产波动较小时获利。 

短跨式期权策略是通过卖出期权合约来实现的，因此买方有可能行使期权。为了确保我们始终在市场上持有跨式期权的两部分，我们会在被指派交易时平仓所有头寸。

def on_order_event(self, order_event):
    if order_event.status == OrderStatus.FILLED and order_event.is_assignment:
        self.liquidate()
结论
结合标普500指数（SPX）、VIX指数和期权跨式期权策略，可以交易市场波动率。我们实施的策略实际上是与市场预期相反的方向进行交易，即在市场预期波动率较低时押注未来波动率较高，反之亦然。该策略有利可图，但当无风险利率为联邦公开市场委员会（FOMC）公布的主要信用利率时，其夏普比率为负值。
