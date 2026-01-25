# test_ibkr.py - 盈透 API 连接测试

from ib_insync import IB, Stock, util
from datetime import datetime

def test_connection():
    """测试盈透 API 连接"""
    
    print("=" * 60)
    print("🚀 盈透证券 API 连接测试")
    print("=" * 60)
    
    # 创建 IB 实例
    ib = IB()
    
    # 连接到 TWS（Paper Trading）
    try:
        print("\n📡 正在连接 TWS...")
        ib.connect(
            host='127.0.0.1',  # 本地连接
            port=7497,         # Paper Trading 端口
            clientId=1         # 客户端 ID（随意，1-32 之间）
        )
        print("✅ 连接成功！")
    except Exception as e:
        print(f"❌ 连接失败：{e}")
        print("\n🔍 排查步骤：")
        print("  1. 确认 TWS 已启动并登录")
        print("  2. 确认选择了 'Paper Trading' 模式")
        print("  3. 确认 API 已启用（Edit → Global Configuration → API → Settings）")
        print("  4. 确认端口号是 7497（不是 7496）")
        return
    
    # 测试 1：获取账户信息
    print("\n" + "=" * 60)
    print("📊 测试 1：获取账户信息")
    print("=" * 60)
    
    try:
        # 获取账户摘要
        account_summary = ib.accountSummary()
        
        # 提取关键信息
        account_info = {}
        for item in account_summary:
            if item.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower']:
                account_info[item.tag] = float(item.value)
        
        print(f"✅ 账户号：{ib.managedAccounts()[0]}")
        print(f"   净资产：${account_info.get('NetLiquidation', 0):,.2f}")
        print(f"   现金：  ${account_info.get('TotalCashValue', 0):,.2f}")
        print(f"   购买力：${account_info.get('BuyingPower', 0):,.2f}")
        
    except Exception as e:
        print(f"❌ 获取账户信息失败：{e}")
    
    # 测试 2：获取股票实时报价
    print("\n" + "=" * 60)
    print("📈 测试 2：获取 AAPL 股票实时报价")
    print("=" * 60)
    
    try:
        # 创建股票合约
        aapl = Stock('AAPL', 'SMART', 'USD')
        
        # 验证合约
        ib.qualifyContracts(aapl)
        print(f"✅ 合约验证成功：{aapl}")
        
        # 请求市场数据
        ticker = ib.reqMktData(aapl, '', False, False)
        
        # 等待数据更新
        ib.sleep(2)
        
        # 打印报价
        print(f"\n📊 AAPL 实时报价：")
        print(f"   买价：${ticker.bid:.2f}")
        print(f"   卖价：${ticker.ask:.2f}")
        print(f"   最新价：${ticker.last:.2f}")
        print(f"   成交量：{ticker.volume:,}")
        
        # 取消订阅
        ib.cancelMktData(aapl)
        
    except Exception as e:
        print(f"❌ 获取报价失败：{e}")
    
    # 测试 3：获取历史数据
    print("\n" + "=" * 60)
    print("📉 测试 3：获取 AAPL 历史数据（最近5天）")
    print("=" * 60)
    
    try:
        bars = ib.reqHistoricalData(
            aapl,
            endDateTime='',
            durationStr='5 D',  # 最近5天
            barSizeSetting='1 day',  # 日线
            whatToShow='TRADES',
            useRTH=True  # 仅常规交易时段
        )
        
        print(f"✅ 获取了 {len(bars)} 条历史数据：")
        for bar in bars[-5:]:  # 打印最近5条
            print(f"   {bar.date.date()}  开盘：${bar.open:.2f}  "
                  f"收盘：${bar.close:.2f}  成交量：{bar.volume:,}")
    
    except Exception as e:
        print(f"❌ 获取历史数据失败：{e}")
    
    # 测试 4：获取期权链（核心功能！）
    print("\n" + "=" * 60)
    print("📋 测试 4：获取 AAPL 期权链")
    print("=" * 60)
    
    try:
        # 获取期权链参数
        chains = ib.reqSecDefOptParams(aapl.symbol, '', aapl.secType, aapl.conId)
        
        if chains:
            chain = chains[0]
            print(f"✅ 期权链获取成功：")
            print(f"   交易所：{chain.exchange}")
            print(f"   到期日数量：{len(chain.expirations)}")
            print(f"   最近到期日：{sorted(chain.expirations)[:3]}")
            print(f"   行权价数量：{len(chain.strikes)}")
            print(f"   行权价范围：${min(chain.strikes):.2f} - ${max(chain.strikes):.2f}")
        else:
            print("⚠️ 未找到期权链（可能市场未开盘）")
    
    except Exception as e:
        print(f"❌ 获取期权链失败：{e}")
    
    # 测试 5：模拟下单（不会真正执行）
    print("\n" + "=" * 60)
    print("🧪 测试 5：模拟下单（预览模式）")
    print("=" * 60)
    
    try:
        from ib_insync import MarketOrder
        
        # 创建订单（1股 AAPL）
        order = MarketOrder('BUY', 1)
        
        # 预览订单（不会实际执行）
        print(f"✅ 订单创建成功：")
        print(f"   动作：{order.action}")
        print(f"   数量：{order.totalQuantity}")
        print(f"   类型：{order.orderType}")
        print("\n⚠️ 这只是预览，未实际下单（需要调用 ib.placeOrder() 才会执行）")
    
    except Exception as e:
        print(f"❌ 订单创建失败：{e}")
    
    # 断开连接
    print("\n" + "=" * 60)
    print("🔌 断开连接...")
    ib.disconnect()
    print("✅ 测试完成！")
    print("=" * 60)
    
    # 总结
    print("\n📝 总结：")
    print("   如果所有测试都通过，说明 API 配置正确！")
    print("   下一步：开始编写自动交易策略")
    print("\n⚠️ 重要提示：")
    print("   - 当前是 Paper Trading（模拟账户）")
    print("   - 端口 7497 = 模拟账户")
    print("   - 端口 7496 = 真实账户（谨慎使用！）")


if __name__ == '__main__':
    test_connection()