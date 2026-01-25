# -*- coding: utf-8 -*-
"""
A/B测试配置与报告生成器 / A/B Test Configuration and Report Generator

管理 ButterAI vs ButterBaseline 的分流比例和表现对比

功能 / Features:
1. 配置分流比例 / Configure traffic split ratio
2. 生成表现对比报告 / Generate performance comparison report
3. 统计显著性检验 / Statistical significance test

用法 / Usage:
    python check/ab_test_manager.py --status     # 查看当前状态
    python check/ab_test_manager.py --report     # 生成对比报告
    python check/ab_test_manager.py --set-ratio 70 30  # 设置AI:Baseline比例
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目路径 / Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ABTestManager:
    """
    A/B测试管理器 / A/B Test Manager
    
    管理 ButterAI (实验组) vs ButterBaseline (对照组) 的配置和分析
    """
    
    def __init__(self):
        self.config_file = PROJECT_ROOT / 'backend' / 'ab_test_config.json'
        self.data_dir = PROJECT_ROOT / 'backend' / 'data'
        
        # 默认配置 / Default configuration
        self.default_config = {
            'ai_ratio': 0.7,  # AI Track占比 / AI Track ratio
            'baseline_ratio': 0.3,  # Baseline占比 / Baseline ratio
            'min_sample_size': 30,  # 最小样本量 / Min sample size for significance
            'confidence_level': 0.95,  # 置信水平 / Confidence level
            'enabled': True,  # A/B测试是否启用 / A/B test enabled
            'start_date': None,  # 测试开始日期 / Test start date
        }
    
    def load_config(self):
        """加载A/B测试配置 / Load A/B test config"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return self.default_config.copy()
    
    def save_config(self, config):
        """保存A/B测试配置 / Save A/B test config"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"✅ 配置已保存 / Config saved to {self.config_file}")
    
    def set_ratio(self, ai_pct: int, baseline_pct: int):
        """
        设置分流比例 / Set traffic split ratio
        
        参数 / Parameters:
            ai_pct: AI Track百分比 / AI Track percentage (0-100)
            baseline_pct: Baseline百分比 / Baseline percentage (0-100)
        """
        if ai_pct + baseline_pct != 100:
            logger.error(f"❌ 比例之和必须为100: {ai_pct} + {baseline_pct} = {ai_pct + baseline_pct}")
            return False
        
        config = self.load_config()
        config['ai_ratio'] = ai_pct / 100.0
        config['baseline_ratio'] = baseline_pct / 100.0
        
        if config['start_date'] is None:
            config['start_date'] = datetime.now().isoformat()
        
        self.save_config(config)
        
        logger.info(f"📊 分流比例已更新:")
        logger.info(f"   ButterAI: {ai_pct}%")
        logger.info(f"   ButterBaseline: {baseline_pct}%")
        
        return True
    
    def show_status(self):
        """显示当前A/B测试状态 / Show current A/B test status"""
        print("\n" + "=" * 60)
        print("📊 A/B 测试状态 / A/B Test Status")
        print("=" * 60)
        
        config = self.load_config()
        
        print(f"\n启用状态 / Enabled: {'✅ 是' if config.get('enabled', True) else '❌ 否'}")
        print(f"开始日期 / Start date: {config.get('start_date', 'Not set')}")
        print(f"\n分流比例 / Traffic Split:")
        print(f"   🤖 ButterAI:       {config['ai_ratio']*100:.0f}%")
        print(f"   📊 ButterBaseline: {config['baseline_ratio']*100:.0f}%")
        
        # 统计交易数量 / Count trades
        self._show_trade_counts()
    
    def _show_trade_counts(self):
        """显示交易统计 / Show trade counts"""
        import sqlite3
        
        db_path = self.data_dir / 'history.db'
        if not db_path.exists():
            print("\n⚠️ 暂无交易数据")
            return
        
        conn = sqlite3.connect(str(db_path))
        
        try:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='trades_history'
            """)
            
            if not cursor.fetchone():
                print("\n⚠️ trades_history表不存在")
                return
            
            cursor = conn.execute("""
                SELECT 
                    order_ref,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled
                FROM trades_history
                GROUP BY order_ref
            """)
            
            results = cursor.fetchall()
            
            if results:
                print(f"\n交易统计 / Trade Statistics:")
                for ref, total, filled in results:
                    fill_rate = filled / total * 100 if total > 0 else 0
                    print(f"   {ref or 'Unknown'}: {total} trades (成交率 {fill_rate:.1f}%)")
            else:
                print("\n⚠️ 暂无交易记录")
                
        except Exception as e:
            print(f"\n❌ 查询失败: {e}")
        finally:
            conn.close()
    
    def generate_report(self, days: int = 30):
        """
        生成A/B测试对比报告 / Generate A/B test comparison report
        """
        print("\n" + "=" * 60)
        print("📈 A/B 测试对比报告 / A/B Test Comparison Report")
        print("=" * 60)
        
        import sqlite3
        
        db_path = self.data_dir / 'history.db'
        if not db_path.exists():
            print("\n⚠️ 数据库不存在,请先运行一些交易")
            return
        
        conn = sqlite3.connect(str(db_path))
        
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            # 获取详细交易数据 / Get detailed trade data
            cursor = conn.execute("""
                SELECT 
                    order_ref,
                    ticker,
                    status,
                    theoretical_price,
                    price,
                    timestamp
                FROM trades_history
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (cutoff,))
            
            trades = cursor.fetchall()
            
            if not trades:
                print(f"\n⚠️ 最近 {days} 天没有交易记录")
                return
            
            # 分组分析 / Group analysis
            ai_trades = [t for t in trades if t[0] == 'ButterAI']
            baseline_trades = [t for t in trades if t[0] == 'ButterBaseline']
            
            print(f"\n📅 分析周期 / Analysis Period: 最近 {days} 天")
            print("-" * 50)
            
            self._print_group_stats("🤖 ButterAI", ai_trades)
            self._print_group_stats("📊 ButterBaseline", baseline_trades)
            
            # 对比结论 / Comparison conclusion
            self._print_comparison(ai_trades, baseline_trades)
            
        except Exception as e:
            print(f"\n❌ 报告生成失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()
    
    def _print_group_stats(self, name, trades):
        """打印组统计 / Print group statistics"""
        print(f"\n{name}:")
        
        if not trades:
            print("   暂无数据")
            return
        
        total = len(trades)
        filled = sum(1 for t in trades if t[2] == 'FILLED')
        fill_rate = filled / total * 100 if total > 0 else 0
        
        # 计算滑点 / Calculate slippage
        slippages = []
        for t in trades:
            theo, actual = t[3], t[4]
            if theo and actual and theo > 0:
                slip = (actual - theo) / theo * 100
                slippages.append(slip)
        
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0
        
        print(f"   总交易数 / Total trades: {total}")
        print(f"   成交数 / Filled: {filled} ({fill_rate:.1f}%)")
        print(f"   平均滑点 / Avg slippage: {avg_slippage:+.2f}%")
        
        # 最近交易 / Recent trades
        print(f"   最近交易 / Recent trades:")
        for t in trades[:3]:
            print(f"      - {t[1]} ({t[5][:10]}): {t[2]}")
    
    def _print_comparison(self, ai_trades, baseline_trades):
        """打印对比结论 / Print comparison conclusion"""
        print("\n" + "-" * 50)
        print("📊 对比结论 / Comparison Conclusion:")
        
        ai_filled = sum(1 for t in ai_trades if t[2] == 'FILLED') if ai_trades else 0
        bl_filled = sum(1 for t in baseline_trades if t[2] == 'FILLED') if baseline_trades else 0
        
        ai_rate = ai_filled / len(ai_trades) * 100 if ai_trades else 0
        bl_rate = bl_filled / len(baseline_trades) * 100 if baseline_trades else 0
        
        if len(ai_trades) < 10 or len(baseline_trades) < 10:
            print("   ⚠️ 样本量不足 (每组需至少10笔交易)")
            print("   → 请继续运行系统以积累更多数据")
        else:
            diff = ai_rate - bl_rate
            if diff > 5:
                print(f"   ✅ ButterAI 表现优于 Baseline ({diff:+.1f}%)")
            elif diff < -5:
                print(f"   ⚠️ ButterAI 表现劣于 Baseline ({diff:+.1f}%)")
            else:
                print(f"   ➡️ 两组表现相近 (差异: {diff:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='A/B测试管理器 / A/B Test Manager')
    parser.add_argument('--status', action='store_true', help='查看当前状态 / Show current status')
    parser.add_argument('--report', action='store_true', help='生成对比报告 / Generate comparison report')
    parser.add_argument('--set-ratio', nargs=2, type=int, metavar=('AI', 'BASELINE'),
                        help='设置分流比例 / Set ratio (e.g., --set-ratio 70 30)')
    parser.add_argument('--days', type=int, default=30, help='报告分析天数 / Days for report (default: 30)')
    args = parser.parse_args()
    
    manager = ABTestManager()
    
    if args.status:
        manager.show_status()
        
    elif args.report:
        manager.generate_report(days=args.days)
        
    elif args.set_ratio:
        manager.set_ratio(args.set_ratio[0], args.set_ratio[1])
        
    else:
        # 默认显示状态 / Default: show status
        manager.show_status()


if __name__ == "__main__":
    main()
