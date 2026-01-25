# -*- coding: utf-8 -*-
"""
Connection Test Tool - IBKR 连接测试工具
验证与 TWS/Gateway 的连接以及行情权限 / Verifies connection with TWS/Gateway and market data permissions
"""

from ib_insync import *
import nest_asyncio

# 允许在 Jupyter 或现有循环中运行异步代码 / Allow running async code in existing loops
nest_asyncio.apply()

def check_connection():
    """验证 IBKR TWS 连接和行情数据状态 / Verify IBKR TWS connection and market data status"""
    ib = IB()
    print("Connecting to IBKR TWS on 127.0.0.1:7497...")
    try:
        # 连接到 TWS (默认 Paper Trading 端口为 7497) / Connect to TWS (Default Paper Port: 7497)
        ib.connect('127.0.0.1', 7497, clientId=1)
        print("✅ Connection Successful! / 连接成功!")
        print(f"Connected to Account: {ib.managedAccounts()}")
        
        # 切换到延迟行情模式 (类型 3) / Switch to Delayed market data (Type 3)
        # 1: 实时(Live), 2: 冻结(Frozen), 3: 延迟(Delayed), 4: 延迟冻结(Delayed Frozen)
        ib.reqMarketDataType(3)
        print("💡 Switched to Delayed Market Data mode / 已切换到延迟行情模式.")
        
        # 检查账户概览 / Check Account Summary
        summary = ib.accountSummary()
        cash = [s.value for s in summary if s.tag == 'NetLiquidation' and s.currency == 'USD']
        print(f"💰 Net Liquidation (USD): {cash[0] if cash else 'Not found'}")
        
        # 验证行情数据权限 (以 AAPL 为例) / Verify Market Data Permissions (e.g., AAPL)
        print("\nChecking Market Data Permissions for AAPL...")
        aapl = Stock('AAPL', 'SMART', 'USD')
        ib.qualifyContracts(aapl)
        
        # 请求行情快照 / Request market data snapshot
        ticker = ib.reqMktData(aapl, "", False, False)
        ib.sleep(2)
        
        if ticker.last != ticker.last: # 检查是否为 NaN / Check for NaN
            print("⚠️ Warning: Price not detected (Normal if market is closed or permissions missing)")
            print("💡 Tip: Ensure 'Send status updates for delayed market data' is enabled in TWS.")
        else:
            print(f"✅ Real-time/Delayed AAPL Price: {ticker.last}")
            
        ib.disconnect()
        print("\nVerification Complete / 验证完成.")
    except Exception as e:
        print(f"❌ Connection Failed: {e} / 连接失败")
        print("\nTroubleshooting Tips / 故障排除方案:")
        print("1. Ensure IBKR TWS or Gateway is OPEN and logged into PAPER account.")
        print("2. TWS -> Global Configuration -> API -> Settings -> 'Enable ActiveX and Socket Clients' must be checked.")
        print("3. Verify the socket port is 7497.")

if __name__ == "__main__":
    check_connection()
