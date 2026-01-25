# -*- coding: utf-8 -*-
"""
AlpacaTrader - Alpaca 交易执行模块 / Alpaca Trading Execution Module
负责与 Alpaca Markets API 连接，执行交易指令 / Responsible for connecting with Alpaca Markets API and executing trading orders
"""

import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest, 
    MarketOrderRequest, 
    LimitOrderRequest, 
    TakeProfitRequest, 
    StopLossRequest,
    OrderRequest
)
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce, OrderType, OrderClass
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest

# 加载配置 / Load environment variables
load_dotenv()

# 配置日志 / Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AlpacaTrader')

class AlpacaTrader:
    def __init__(self, api_key=None, secret_key=None, paper=True):
        """
        初始化 Alpaca 交易模块 / Initialize Alpaca trading module
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper = str(os.getenv('ALPACA_PAPER', paper)).lower() == 'true'
        
        self.trading_client = None
        self.data_client = None
        self.account_summary = {}
        
        # 风险控制参数 (与 ButterTrader 保持一致)
        self.MAX_POSITIONS = 100
        self.ALLOCATION_PER_STRATEGY = 1000

    def connect(self):
        """连接到 Alpaca / Connect to Alpaca"""
        try:
            if not self.api_key or not self.secret_key:
                logger.error("缺少 Alpaca API Key 或 Secret Key / Missing Alpaca API keys")
                return False
            
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            self.data_client = OptionHistoricalDataClient(self.api_key, self.secret_key)
            
            # 测试连接 / Test connection
            account = self.trading_client.get_account()
            logger.info(f"成功连接到 Alpaca {'Paper' if self.paper else 'Live'} 账户!")
            logger.info(f"账户 ID: {account.id} | 状态: {account.status}")
            return True
        except Exception as e:
            logger.error(f"Alpaca 连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接 (Alpaca SDK 无需显式断开) / Disconnect"""
        logger.info("Alpaca 客户端已释放 / Alpaca client released")

    def get_account_summary(self):
        """获取账户资金摘要 / Get account fund summary"""
        if not self.trading_client:
            return None
        
        try:
            account = self.trading_client.get_account()
            summary = {
                'TotalCashValue': float(account.cash),
                'NetLiquidation': float(account.portfolio_value),
                'BuyingPower': float(account.buying_power),
                'AvailableFunds': float(account.non_marginable_buying_power)
            }
            self.account_summary = summary
            logger.info(f"账户摘要: {summary}")
            return summary
        except Exception as e:
            logger.error(f"获取账户摘要失败: {e}")
            return None

    def get_positions(self):
        """获取当前所有持仓 / Get all current positions"""
        if not self.trading_client:
            return []
        
        try:
            positions = self.trading_client.get_all_positions()
            logger.info(f"当前持仓数量: {len(positions)}")
            return positions
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            return []

    def get_active_symbols(self):
        """
        获取当前所有活跃股票代码 (返回底层的 ticker，以便与 execution_engine 匹配)
        """
        if not self.trading_client:
            return set()
            
        try:
            active_underlying = set()
            
            # 1. 获取持仓 / Get positions
            positions = self.trading_client.get_all_positions()
            for p in positions:
                if p.asset_class == AssetClass.US_OPTION:
                    # 对于期权，Alpaca 的 position 对象通常包含 underlying_symbol
                    # 如果没有，我们需要从 symbol (OCC) 中提取 (通常是前几个字母)
                    if hasattr(p, 'underlying_symbol') and p.underlying_symbol:
                        active_underlying.add(p.underlying_symbol)
                    else:
                        # 简单的 OCC 提取: AAPL230616C00150000 -> AAPL
                        # 寻找第一个数字
                        import re
                        match = re.match(r'^([A-Z]+)\d', p.symbol)
                        if match:
                            active_underlying.add(match.group(1))
                else:
                    active_underlying.add(p.symbol)
            
            # 2. 获取挂单 / Get pending orders
            orders = self.trading_client.get_orders(status='open')
            for o in orders:
                if o.asset_class == AssetClass.US_OPTION:
                    # 同样提取底层的 ticker
                    import re
                    match = re.match(r'^([A-Z]+)\d', o.symbol)
                    if match:
                        active_underlying.add(match.group(1))
                else:
                    active_underlying.add(o.symbol)
            
            logger.info(f"活跃/挂单股票汇总 (Underlying): {active_underlying}")
            return active_underlying
        except Exception as e:
            logger.error(f"获取活跃代码失败: {e}")
            return set()

    def check_risk_limits(self, current_positions_count):
        """风险控制检查"""
        if current_positions_count >= self.MAX_POSITIONS:
            logger.warning(f"🚫 风险控制触发: 达到最大持仓限制 ({self.MAX_POSITIONS})")
            return False
        return True

    def find_closest_contract(self, symbol, target_date_str, target_strike, right):
        """
        寻找最接近目标日期和行权价的有效合约
        :param right: 'C' or 'P'
        """
        try:
            # 转换日期格式 / Convert date format
            target_date = datetime.strptime(target_date_str.replace('-', ''), '%Y%m%d').date()
            
            # 请求期权合约 / Request option contracts
            request_params = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status='active',
                expiration_date_gte=target_date_str,
                limit=1000  # 增加限制以获取更多到期日 / Increase limit to get more expiries
            )
            
            result = self.trading_client.get_option_contracts(request_params)
            contracts = result.option_contracts
            
            if not contracts:
                logger.warning(f"无法找到 {symbol} 从 {target_date_str} 开始的期权合约 / No contracts found")
                return None
            
            logger.info(f"[{symbol}] 找到 {len(contracts)} 个候选合约")
            
            # 1. 寻找最近的到期日 / Find closest expiry
            unique_expiries = sorted(list(set(c.expiration_date for c in contracts)))
            closest_expiry = min(unique_expiries, key=lambda x: abs((x - target_date).days))
            
            logger.info(f"[{symbol}] 目标日期: {target_date_str} -> 匹配到期日: {closest_expiry}")

            # 2. 在该到期日下寻找最接近的行权价 / Find closest strike on that expiry
            filtered_contracts = [
                c for c in contracts 
                if c.expiration_date == closest_expiry 
                and c.contract_type.lower() == ('call' if right == 'C' else 'put')
            ]
            
            if not filtered_contracts:
                logger.warning(f"[{symbol}] 在 {closest_expiry} 未找到 {right} 类型的合约")
                return None
            
            closest_contract = min(filtered_contracts, key=lambda x: abs(float(x.strike_price) - target_strike))
            
            logger.info(f"[{symbol}] 目标行权价: {target_strike} -> 匹配: {closest_contract.strike_price} ({closest_contract.symbol})")
            return closest_contract

        except Exception as e:
            logger.error(f"搜索 Alpaca 合约失败: {e}")
            return None

    def get_option_contract(self, symbol, expiry, strike, right='C'):
        """兼容接口"""
        return self.find_closest_contract(symbol, expiry, strike, right)

    def place_butterfly_order(self, ticker, butterfly_details, strategy_type='AI', target_allocation=None, price_offset=0.0, use_market_order=False):
        """
        下单蝴蝶策略
        Alpaca 目前对多腿订单的支持主要是通过单个 Leg 提交或者使用特定的 OrderClass (如果 API 支持)。
        为了稳定起见，我们目前采用分腿下单或同步提交。
        注意: Alpaca API 正在快速更新对组合单的支持。
        """
        if target_allocation is None:
            target_allocation = self.ALLOCATION_PER_STRATEGY

        current_positions = len(self.get_positions())
        if not self.check_risk_limits(current_positions):
            return {'status': 'rejected', 'reason': 'Risk limit reached'}

        order_ref = f"Butter{strategy_type}"
        bf_type = butterfly_details.get('type', 'CALL')
        expiry = butterfly_details.get('expiry').replace('-', '')

        logger.info(f"[{order_ref}] 正在为 {ticker} 构建 {bf_type} 蝴蝶策略...")

        try:
            # 1. 获取合约 / Get contracts
            strikes = [butterfly_details['lower'], butterfly_details['center'], butterfly_details['upper']]
            legs_cfg = []
            
            if bf_type == 'CALL':
                for i, strike in enumerate(strikes):
                    c = self.get_option_contract(ticker, expiry, strike, 'C')
                    if not c: return {'status': 'failed', 'reason': f'Contract {strike}C not found'}
                    legs_cfg.append({'contract': c, 'qty_mult': (2 if i==1 else 1), 'side': (OrderSide.SELL if i==1 else OrderSide.BUY)})
            
            elif bf_type == 'PUT':
                for i, strike in enumerate(strikes):
                    p = self.get_option_contract(ticker, expiry, strike, 'P')
                    if not p: return {'status': 'failed', 'reason': f'Contract {strike}P not found'}
                    legs_cfg.append({'contract': p, 'qty_mult': (2 if i==1 else 1), 'side': (OrderSide.SELL if i==1 else OrderSide.BUY)})
            
            elif bf_type == 'IRON':
                # BUY lower Put, SELL center Put, SELL center Call, BUY upper Call
                p1 = self.get_option_contract(ticker, expiry, strikes[0], 'P')
                p2 = self.get_option_contract(ticker, expiry, strikes[1], 'P')
                c2 = self.get_option_contract(ticker, expiry, strikes[1], 'C')
                c3 = self.get_option_contract(ticker, expiry, strikes[2], 'C')
                if not (p1 and p2 and c2 and c3):
                    return {'status': 'failed', 'reason': 'Iron Butterfly contracts not found'}
                
                legs_cfg = [
                    {'contract': p1, 'qty_mult': 1, 'side': OrderSide.BUY},
                    {'contract': p2, 'qty_mult': 1, 'side': OrderSide.SELL},
                    {'contract': c2, 'qty_mult': 1, 'side': OrderSide.SELL},
                    {'contract': c3, 'qty_mult': 1, 'side': OrderSide.BUY}
                ]

            # 2. 计算价格及头寸 (由于分腿下单可能存在风险，我们在这里尝试获取市场中值)
            # 理想情况下应该使用组合订单。这里简化处理：计算每条腿的预期成本之和。
            total_net_debit = 0
            for leg in legs_cfg:
                quote_req = OptionLatestQuoteRequest(symbol_or_contract_id=leg['contract'].symbol)
                quote = self.data_client.get_option_latest_quote(quote_req)
                # Alpaca 结果是一个字典，key 是 symbol
                q = quote[leg['contract'].symbol]
                mid = (q.bid_price + q.ask_price) / 2
                if leg['side'] == OrderSide.BUY:
                    total_net_debit += mid * leg['qty_mult']
                else:
                    total_net_debit -= mid * leg['qty_mult']

            # 3. 计算数量 / Calculate Quantity
            unit_cost = abs(total_net_debit) * 100
            if unit_cost <= 0: unit_cost = 1.0
            quantity = int(target_allocation // unit_cost)
            if quantity < 1: quantity = 1

            # 4. 提交订单 / Submit Orders
            # 这里我们循环提交每一腿订单。在实际生产中，建议使用支持 Multi-leg 的 API 接口以避免腿风险。
            # Alpaca API v2 已初步支持组合单，但 SDK 文档可能滞后。
            # 暂时使用分腿提交以保证兼容性，并标记为同一 batch。
            
            results = []
            for leg in legs_cfg:
                qty = quantity * leg['qty_mult']
                if use_market_order:
                    req = MarketOrderRequest(
                        symbol=leg['contract'].symbol,
                        qty=qty,
                        side=leg['side'],
                        time_in_force=TimeInForce.DAY,
                        client_order_id=f"{order_ref}_{ticker}_{int(time.time())}_{leg['contract'].symbol[:5]}"
                    )
                else:
                    # 获取该腿的中值
                    quote_req = OptionLatestQuoteRequest(symbol_or_contract_id=leg['contract'].symbol)
                    q = self.data_client.get_option_latest_quote(quote_req)[leg['contract'].symbol]
                    leg_mid = (q.bid_price + q.ask_price) / 2
                    # 简单偏移逻辑
                    l_price = leg_mid + (price_offset if leg['side'] == OrderSide.BUY else -price_offset)
                    
                    req = LimitOrderRequest(
                        symbol=leg['contract'].symbol,
                        qty=qty,
                        side=leg['side'],
                        limit_price=round(l_price, 2),
                        time_in_force=TimeInForce.DAY,
                        client_order_id=f"{order_ref}_{ticker}_{int(time.time())}_{leg['contract'].symbol[:5]}"
                    )
                
                order = self.trading_client.submit_order(req)
                results.append(order.id)
                logger.info(f"✅ Leg {leg['contract'].symbol} 提交成功: {order.id}")

            return {'status': 'submitted', 'orders': results, 'ref': order_ref}

        except Exception as e:
            logger.error(f"Alpaca 下单失败: {e}")
            return {'status': 'error', 'reason': str(e)}

if __name__ == "__main__":
    trader = AlpacaTrader()
    if trader.connect():
        trader.get_account_summary()
        trader.get_positions()
        logger.info("AlpacaTrader 测试连接成功")
