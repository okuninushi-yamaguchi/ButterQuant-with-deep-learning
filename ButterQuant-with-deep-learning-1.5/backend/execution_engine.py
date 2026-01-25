# -*- coding: utf-8 -*-
"""
Execution Engine Redirect - 执行引擎重定向
交易程序已移至独立文件夹以支持多账户隔离。
The trading programs have been moved to separate folders for multi-account isolation.
"""

import sys
import os

def main():
    print("="*60)
    print("⚠️  注意: 交易引擎位置已更改 / Trading engine location has changed")
    print("="*60)
    print("\n请根据您要使用的账户运行对应的程序:")
    print("Please run the corresponding program for the account you want to use:\n")
    print("1. Alpaca Trading (当前活跃):")
    print("   python 'alpaca trade/execution_engine_alpaca.py'")
    print("\n2. IBKR Trading (待平仓或资金恢复后):")
    print("   python 'ibkr trade/execution_engine_ibkr.py'")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
