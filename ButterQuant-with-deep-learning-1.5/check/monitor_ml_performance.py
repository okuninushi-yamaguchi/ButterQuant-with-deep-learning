# -*- coding: utf-8 -*-
"""
ML性能监控脚本 / ML Performance Monitoring Script

追踪ML预测 vs 实际交易表现 / Track ML predictions vs actual trade performance

功能 / Features:
1. 统计 ButterAI vs ButterBaseline 交易成功率
2. 分析预期ROI vs 实际盈亏
3. 检测模型漂移 (Concept Drift)

用法 / Usage:
    python check/monitor_ml_performance.py
    python check/monitor_ml_performance.py --days 30  # 指定时间范围
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目路径 / Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))


def get_db_connection():
    """获取数据库连接 / Get database connection"""
    import sqlite3
    db_path = PROJECT_ROOT / 'backend' / 'data' / 'market_research.db'
    
    if not db_path.exists():
        print(f"❌ 数据库不存在: {db_path}")
        return None
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_history_db_connection():
    """获取历史数据库连接 / Get history database connection"""
    import sqlite3
    db_path = PROJECT_ROOT / 'backend' / 'data' / 'history.db'
    
    if not db_path.exists():
        print(f"❌ 历史数据库不存在: {db_path}")
        return None
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def analyze_trade_performance(days: int = 30):
    """
    分析交易表现 / Analyze trade performance
    
    比较 ButterAI vs ButterBaseline 的交易成功率
    """
    print("\n" + "=" * 60)
    print("🤖 ButterAI vs 📊 ButterBaseline 交易表现分析")
    print("=" * 60)
    
    conn = get_history_db_connection()
    if conn is None:
        return
    
    try:
        # 检查 trades_history 表是否存在 / Check if trades_history table exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='trades_history'
        """)
        
        if not cursor.fetchone():
            print("⚠️ trades_history 表不存在,请先执行交易")
            print("  → 运行 python backend/execution_engine.py 生成交易记录")
            return
        
        # 获取最近N天的交易记录 / Get trades from last N days
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor = conn.execute("""
            SELECT 
                order_ref,
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled_trades,
                AVG(price) as avg_price,
                AVG(theoretical_price) as avg_theoretical
            FROM trades_history
            WHERE timestamp >= ?
            GROUP BY order_ref
        """, (cutoff_date,))
        
        results = cursor.fetchall()
        
        if not results:
            print(f"⚠️ 最近 {days} 天没有交易记录")
            print("  → 系统需要运行一段时间才能积累数据")
            return
        
        print(f"\n📅 时间范围: 最近 {days} 天")
        print("-" * 50)
        
        for row in results:
            order_ref = row['order_ref'] or 'Unknown'
            total = row['total_trades'] or 0
            filled = row['filled_trades'] or 0
            fill_rate = (filled / total * 100) if total > 0 else 0
            
            icon = "🤖" if order_ref == "ButterAI" else "📊"
            print(f"\n{icon} {order_ref}:")
            print(f"   总交易数 / Total trades: {total}")
            print(f"   成交数 / Filled: {filled} ({fill_rate:.1f}%)")
            
            if row['avg_price'] and row['avg_theoretical']:
                slippage = ((row['avg_price'] - row['avg_theoretical']) / row['avg_theoretical'] * 100)
                print(f"   平均滑点 / Avg slippage: {slippage:+.2f}%")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
    finally:
        conn.close()


def analyze_rankings_predictions():
    """
    分析排名中的ML预测分布 / Analyze ML prediction distribution in rankings
    """
    print("\n" + "=" * 60)
    print("📊 排名数据中的ML预测分布")
    print("=" * 60)
    
    rankings_path = PROJECT_ROOT / 'backend' / 'data' / 'rankings_combined.json'
    
    if not rankings_path.exists():
        print(f"⚠️ 排名文件不存在: {rankings_path}")
        print("  → 请先运行 python backend/daily_scanner.py")
        return
    
    try:
        with open(rankings_path, 'r', encoding='utf-8') as f:
            rankings = json.load(f)
        
        print(f"\n总候选数 / Total candidates: {len(rankings)}")
        
        # 统计 ML 预测分布 / Analyze ML prediction distribution
        with_ml_roi = [r for r in rankings if r.get('ml_expected_roi')]
        with_ml_prob = [r for r in rankings if r.get('ml_success_prob')]
        with_ml_dist = [r for r in rankings if r.get('ml_roi_distribution')]
        
        print(f"\n有 ml_expected_roi: {len(with_ml_roi)}")
        print(f"有 ml_success_prob: {len(with_ml_prob)}")
        print(f"有 ml_roi_distribution: {len(with_ml_dist)}")
        
        if with_ml_roi:
            rois = [r['ml_expected_roi'] for r in with_ml_roi]
            import statistics
            avg_roi = statistics.mean(rois)
            max_roi = max(rois)
            min_roi = min(rois)
            above_threshold = len([r for r in rois if r >= 0.15])
            
            print(f"\n期望ROI统计 / Expected ROI Stats:")
            print(f"   平均 / Mean:   {avg_roi:.2%}")
            print(f"   最大 / Max:    {max_roi:.2%}")
            print(f"   最小 / Min:    {min_roi:.2%}")
            print(f"   ≥15% 阈值:   {above_threshold} 个 ({above_threshold/len(rois)*100:.1f}%)")
        
        # 分类分布 / Class distribution
        if with_ml_dist:
            print(f"\n类别概率分布 (前5个) / Class probability (top 5):")
            for i, r in enumerate(with_ml_dist[:5], 1):
                dist = r['ml_roi_distribution']
                ticker = r.get('ticker', 'N/A')
                print(f"   {i}. {ticker}: "
                      f"Loss={dist.get('prob_loss', 0):.1%} | "
                      f"Minor={dist.get('prob_minor', 0):.1%} | "
                      f"Good={dist.get('prob_good', 0):.1%} | "
                      f"Excel={dist.get('prob_excellent', 0):.1%}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")


def check_model_drift():
    """
    检测模型漂移 / Detect concept drift
    
    比较最近预测与历史预测的分布差异
    """
    print("\n" + "=" * 60)
    print("🔍 模型漂移检测 / Concept Drift Detection")
    print("=" * 60)
    
    print("\n⚠️ 此功能需要更多历史数据才能启用")
    print("   → 系统需要运行至少30天以积累足够样本")
    print("   → 未来将自动计算预测分布漂移")


def generate_summary():
    """生成汇总报告 / Generate summary report"""
    print("\n" + "=" * 60)
    print("📋 ML监控汇总 / ML Monitoring Summary")
    print("=" * 60)
    
    print("""
下一步建议 / Next Steps:
1. 让系统运行更长时间以积累交易数据
2. 定期运行此脚本监控ML表现
3. 如果ButterAI胜率明显低于ButterBaseline,考虑重新训练模型

自动化建议 / Automation:
- 可将此脚本加入每周定期任务
- 设置告警阈值: 如果准确率下降10%以上则告警
""")


def main():
    parser = argparse.ArgumentParser(description='ML性能监控 / ML Performance Monitor')
    parser.add_argument('--days', type=int, default=30, 
                        help='分析的天数范围 / Days to analyze (default: 30)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 ButterQuant ML 性能监控 / ML Performance Monitor")
    print("=" * 60)
    print(f"运行时间 / Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 执行各项分析 / Run analyses
    analyze_trade_performance(days=args.days)
    analyze_rankings_predictions()
    check_model_drift()
    generate_summary()
    
    print("\n✅ 监控完成 / Monitoring complete")


if __name__ == "__main__":
    main()
