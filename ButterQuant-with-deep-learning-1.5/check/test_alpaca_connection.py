# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from alpaca_trader import AlpacaTrader

def test_connection():
    trader = AlpacaTrader()
    print("正在尝试连接 Alpaca...")
    if trader.connect():
        print("✅ 连接成功!")
        summary = trader.get_account_summary()
        if summary:
            print(f"💰 账户资金总览: {summary}")
        
        positions = trader.get_positions()
        print(f"📦 当前持仓数量: {len(positions)}")
        
        # 测试合约搜索
        print("🔍 正在测试合约搜索 (AAPL)...")
        from datetime import datetime, timedelta
        target_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        contract = trader.find_closest_contract("AAPL", target_date, 200, "C")
        if contract:
            print(f"✅ 找到合约: {contract.symbol}")
        else:
            print("❌ 未找到合约")
            
        trader.disconnect()
    else:
        print("❌ 连接失败，请检查 .env 文件中的 API 密钥。")

if __name__ == "__main__":
    test_connection()
