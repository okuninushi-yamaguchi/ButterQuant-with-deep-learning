# -*- coding: utf-8 -*-
"""
ButterQuant 数据库导出脚本 / Database Export Script
从 market_research.db 导出分析数据用于ML训练 / Export analysis data from DB for ML training

用法 / Usage:
    python ml/export_db_to_parquet.py
    python ml/export_db_to_parquet.py --output new_training_data.parquet
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import argparse
import json

# 添加项目路径 / Add project path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))

from ml.features import FeatureExtractor, extract_features_v2, calculate_dynamic_evaluation_date

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseExporter:
    """数据库导出器 / Database Exporter"""
    
    def __init__(self, db_path: str = None, output_path: str = None):
        if db_path is None:
            db_path = PROJECT_ROOT / 'backend' / 'data' / 'market_research.db'
        self.db_path = Path(db_path)
        
        if output_path is None:
            output_path = PROJECT_ROOT / 'ml' / 'training_data_from_db.parquet'
        self.output_path = Path(output_path)
    
    def export_with_labels(self, lookback_days: int = 14) -> pd.DataFrame:
        """
        导出带标签的训练数据 / Export labeled training data
        
        标签计算: 使用14天后的实际价格计算ROI
        Label calculation: Use actual price after 14 days to calculate ROI
        """
        import sqlite3
        import yfinance as yf
        
        logger.info(f"📥 从数据库导出数据: {self.db_path}")
        
        if not self.db_path.exists():
            logger.error(f"❌ 数据库不存在: {self.db_path}")
            return pd.DataFrame()
        
        # 连接数据库 / Connect to database
        conn = sqlite3.connect(str(self.db_path))
        
        # 查询分析历史 / Query analysis history
        query = """
        SELECT ticker, analysis_date, full_result 
        FROM analysis_history 
        WHERE analysis_date < date('now', '-{} days')
        ORDER BY analysis_date DESC
        """.format(lookback_days)
        
        try:
            df_raw = pd.read_sql(query, conn)
        except Exception as e:
            logger.error(f"❌ 查询失败: {e}")
            conn.close()
            return pd.DataFrame()
        
        conn.close()
        
        logger.info(f"  获取 {len(df_raw)} 条记录")
        
        if len(df_raw) == 0:
            logger.warning("⚠️ 没有足够的历史数据 (需要至少14天前的数据)")
            return pd.DataFrame()
        
        # 处理每条记录 / Process each record
        samples = []
        processed = 0
        
        for idx, row in df_raw.iterrows():
            try:
                ticker = row['ticker']
                analysis_date = pd.to_datetime(row['analysis_date'])
                
                # 解析分析结果 / Parse analysis result
                if isinstance(row['full_result'], str):
                    analysis = json.loads(row['full_result'])
                else:
                    analysis = row['full_result']
                
                if not analysis:
                    continue
                
                # 提取特征 / Extract features
                features = extract_features_v2(analysis)
                
                # 获取蝴蝶策略参数 / Get butterfly parameters
                butterfly = analysis.get('butterfly', {})
                dte = butterfly.get('dte', 30)
                
                # 动态计算标签 / Calculate dynamic label
                # 1. 计算评估日期 / Calculate evaluation date
                eval_date, _ = calculate_dynamic_evaluation_date(analysis_date, dte)
                
                # 2. 获取未来价格 / Get future price
                future_data = yf.download(
                    ticker,
                    start=eval_date.strftime('%Y-%m-%d'),
                    end=(eval_date + timedelta(days=5)).strftime('%Y-%m-%d'),
                    progress=False
                )
                
                if len(future_data) == 0:
                    continue
                
                # Fix: Handle multi-level columns if present
                close_data = future_data['Close']
                if isinstance(close_data, pd.DataFrame):
                    future_price = float(close_data.iloc[0, 0])
                else:
                    future_price = float(close_data.iloc[0])
                
                # 3. 计算ROI 和 标签 / Calculate ROI and Label
                label, roi = self._calculate_label_and_roi(butterfly, future_price)
                
                # 合并样本 / Merge sample
                sample = {
                    **features,
                    'label': label,
                    '_ticker': ticker,
                    '_date': str(analysis_date.date()),
                    '_source': 'database',
                    '_debug_roi': roi,
                    # Save strikes for future debugging/relabeling
                    'lower_strike': butterfly.get('lower_strike'),
                    'center_strike': butterfly.get('center_strike'),
                    'upper_strike': butterfly.get('upper_strike'),
                    'net_debit': butterfly.get('net_debit'),
                    'dte': dte
                }
                samples.append(sample)
                
                processed += 1
                if processed % 50 == 0:
                    logger.info(f"  已处理 {processed} 条...")
                
            except Exception as e:
                logger.debug(f"  跳过记录: {e}")
                continue
        
        df = pd.DataFrame(samples)
        logger.info(f"✅ 成功导出 {len(df)} 条带标签数据")
        
        # 保存 / Save
        if len(df) > 0:
            df.to_parquet(self.output_path, index=False)
            logger.info(f"💾 已保存: {self.output_path}")
        
        return df
    
    def _calculate_label_and_roi(self, butterfly: dict, future_price: float):
        """计算4分类标签和ROI / Calculate 4-class label and ROI"""
        lower = butterfly.get('lower_strike', 0)
        center = butterfly.get('center_strike', 0)
        upper = butterfly.get('upper_strike', 0)
        cost = butterfly.get('net_debit', butterfly.get('max_loss', 1.0))
        max_profit = butterfly.get('max_profit', 1.0)
        
        if not all([lower, center, upper]):
            return 0, -1.0
        
        # 计算payoff / Calculate payoff
        if lower <= future_price <= upper:
            if future_price <= center:
                payoff = max_profit * (future_price - lower) / (center - lower + 1e-6)
            else:
                payoff = max_profit * (upper - future_price) / (upper - center + 1e-6)
        else:
            payoff = -cost
        
        roi = (payoff - cost) / (cost + 1e-6)
        
        # New thresholds from ml/features.py
        if roi < -0.10:
            label = 0  # 亏损 / Loss
        elif roi < 0.05:
            label = 1  # 微利 / Minor
        elif roi < 0.15:
            label = 2  # 良好 / Good
        else:
            label = 3  # 优秀 / Excellent
            
        return label, roi


def main():
    parser = argparse.ArgumentParser(description='从数据库导出ML训练数据')
    parser.add_argument('--db', type=str, help='数据库路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--lookback', type=int, default=14, help='标签计算回溯天数')
    args = parser.parse_args()
    
    exporter = DatabaseExporter(args.db, args.output)
    df = exporter.export_with_labels(lookback_days=args.lookback)
    
    if len(df) > 0:
        logger.info(f"\n📊 导出统计:")
        logger.info(f"  样本数: {len(df)}")
        logger.info(f"  标签分布: {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
