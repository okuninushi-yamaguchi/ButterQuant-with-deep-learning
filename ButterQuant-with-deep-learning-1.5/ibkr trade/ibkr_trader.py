# -*- coding: utf-8 -*-
"""
IBKRTrader - IBKR 交易执行模块 / IBKR Trading Execution Module
(Restored from original trader.py)
"""

import logging
import time
import random
import os
import sys
from datetime import datetime
from ib_insync import *
import nest_asyncio

# 确保可以导入 backend
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.database import DatabaseManager

# 允许在 Jupyter 或已有事件循环的环境中运行
nest_asyncio.apply()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('IBKRTrader')

class IBKRTrader:
    def __init__(self, host='127.0.0.1', port=7497, client_id=None):
        self.host = host
        self.port = port
        self.client_id = client_id if client_id is not None else random.randint(100, 999)
        self.ib = IB()
        self.account_summary = {}
        self.db = DatabaseManager()
        self.MAX_POSITIONS = 100
        self.ALLOCATION_PER_STRATEGY = 1000

    def connect(self):
        try:
            if not self.ib.isConnected():
                logger.info(f"正在连接到 IBKR TWS ({self.host}:{self.port}, ID:{self.client_id})...")
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                logger.info("连接成功!")
                self.ib.reqMarketDataType(3)
            else:
                logger.info("已连接到 IBKR TWS")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("已断开连接")

    def get_account_summary(self):
        if not self.ib.isConnected(): return None
        tags = ['TotalCashValue', 'NetLiquidation', 'BuyingPower', 'AvailableFunds']
        summary = {}
        account_values = self.ib.accountSummary()
        for item in account_values:
            if item.tag in tags:
                if item.currency == 'USD' or item.currency == '': 
                    summary[item.tag] = float(item.value)
        self.account_summary = summary
        logger.info(f"账户摘要: {summary}")
        return summary

    def get_positions(self):
        if not self.ib.isConnected(): return []
        positions = self.ib.positions()
        logger.info(f"当前持仓数量: {len(positions)}")
        return positions

    def get_active_symbols(self):
        if not self.ib.isConnected(): return set()
        positions = self.ib.positions()
        pos_symbols = {p.contract.symbol for p in positions if p.position != 0}
        open_trades = self.ib.openTrades()
        order_symbols = {t.contract.symbol for t in open_trades if not t.isDone()}
        all_active = pos_symbols.union(order_symbols)
        logger.info(f"活跃/挂单股票汇总: {all_active}")
        return all_active

    def check_risk_limits(self, current_positions_count):
        if current_positions_count >= self.MAX_POSITIONS:
            logger.warning(f"🚫 风险控制触发: 达到最大持仓限制 ({self.MAX_POSITIONS})")
            return False
        return True

    def find_closest_contract(self, symbol, target_date_str, target_strike, right):
        try:
            stock = Stock(symbol, 'SMART', 'USD')
            details = self.ib.reqContractDetails(stock)
            if not details:
                logger.error(f"无法找到标的合约: {symbol}")
                return None
            underlying_conId = details[0].contract.conId
            chains = self.ib.reqSecDefOptParams(symbol, '', 'STK', underlying_conId)
            if not chains:
                logger.error(f"无法获取期权链: {symbol}")
                return None
            smart_chain = next((c for c in chains if c.exchange == 'SMART'), chains[0])
            all_expiries = sorted(list(smart_chain.expirations))
            target_date = datetime.strptime(target_date_str.replace('-', ''), '%Y%m%d')
            closest_expiry = None
            min_diff = float('inf')
            for exp in all_expiries:
                try:
                    exp_date = datetime.strptime(exp, '%Y%m%d')
                    diff = abs((exp_date - target_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        closest_expiry = exp
                except: continue
            if not closest_expiry: return None
            all_strikes = sorted(list(smart_chain.strikes))
            closest_strike = min(all_strikes, key=lambda x: abs(x - target_strike))
            contract = Option(symbol, closest_expiry, closest_strike, right, 'SMART', '100', 'USD')
            return contract
        except Exception as e:
            logger.error(f"智能搜寻合约失败: {e}")
            return None

    def get_option_contract(self, symbol, expiry, strike, right='C'):
        expiry = expiry.replace('-', '')
        contract = self.find_closest_contract(symbol, expiry, strike, right)
        if contract:
            try:
                self.ib.qualifyContracts(contract)
                return contract
            except Exception as e:
                logger.error(f"合约定全失败: {e}")
                return None
        return None

    def place_butterfly_order(self, ticker, butterfly_details, strategy_type='AI', target_allocation=None, price_offset=0.0, use_market_order=False):
        if target_allocation is None:
            target_allocation = self.ALLOCATION_PER_STRATEGY
        current_positions = len(self.ib.positions())
        if not self.check_risk_limits(current_positions):
            return {'status': 'rejected', 'reason': 'Risk limit reached'}
            
        order_ref = 'ButterAI' if strategy_type == 'AI' else 'ButterBaseline'
        bf_type = butterfly_details.get('type', 'CALL')
        expiry = butterfly_details.get('expiry').replace('-', '')
        
        logger.info(f"[{order_ref}] 正在构建 {bf_type} 蝴蝶策略订单: {ticker}")

        try:
            contract = Contract(symbol=ticker, secType='BAG', currency='USD', exchange='SMART')
            legs = []
            strikes = [butterfly_details['lower'], butterfly_details['center'], butterfly_details['upper']]
            
            if bf_type == 'CALL':
                for i, strike in enumerate(strikes):
                    c = self.get_option_contract(ticker, expiry, strike, 'C')
                    if not c: return {'status': 'failed', 'reason': f'Contract {strike}C not found'}
                    legs.append(ComboLeg(conId=c.conId, ratio=(2 if i==1 else 1), action=('SELL' if i==1 else 'BUY'), exchange='SMART'))
            elif bf_type == 'PUT':
                for i, strike in enumerate(strikes):
                    p = self.get_option_contract(ticker, expiry, strike, 'P')
                    if not p: return {'status': 'failed', 'reason': f'Contract {strike}P not found'}
                    legs.append(ComboLeg(conId=p.conId, ratio=(2 if i==1 else 1), action=('SELL' if i==1 else 'BUY'), exchange='SMART'))
            
            contract.comboLegs = legs
            theoretical_price = butterfly_details.get('price', 0.50)
            limit_price = theoretical_price + price_offset
            
            if use_market_order:
                order = MarketOrder('BUY', 1)
            else:
                order = LimitOrder('BUY', 1, round(limit_price, 2))
            
            order.orderRef = order_ref
            trade = self.ib.placeOrder(contract, order)
            return {'status': 'submitted', 'orderId': trade.order.orderId}
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {'status': 'error', 'reason': str(e)}

if __name__ == "__main__":
    trader = IBKRTrader()
    if trader.connect():
        trader.get_account_summary()
        trader.disconnect()
