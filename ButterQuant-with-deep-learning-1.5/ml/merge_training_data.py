# -*- coding: utf-8 -*-
"""
ButterQuant 训练数据合并脚本 / Training Data Merge Script
合并历史模拟数据和数据库导出数据 / Merge historical simulation data and database exports

用法 / Usage:
    python ml/merge_training_data.py
    python ml/merge_training_data.py --output merged_data.parquet
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime

# 添加项目路径 / Add project path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataMerger:
    """训练数据合并器 / Training Data Merger"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = PROJECT_ROOT / 'ml'
        self.output_dir = Path(output_dir)
    
    def find_data_files(self) -> list:
        """查找所有训练数据文件 / Find all training data files"""
        data_files = []
        
        patterns = [
            'training_data_deep.parquet',       # 历史模拟 / Historical simulation
            'training_data_from_db.parquet',    # 数据库导出 / DB export
            'training_data_*.parquet'           # 其他匹配 / Other matches
        ]
        
        for pattern in patterns:
            for f in self.output_dir.glob(pattern):
                if f not in data_files and 'merged' not in f.name:
                    data_files.append(f)
        
        return data_files
    
    def merge_all(self, output_name: str = None) -> pd.DataFrame:
        """合并所有数据文件 / Merge all data files"""
        
        data_files = self.find_data_files()
        
        if not data_files:
            logger.error("❌ 未找到任何训练数据文件")
            return pd.DataFrame()
        
        logger.info("📁 找到以下数据文件:")
        for f in data_files:
            logger.info(f"  - {f.name}")
        
        # 读取并合并 / Read and merge
        dfs = []
        for f in data_files:
            try:
                df = pd.read_parquet(f)
                df['_source_file'] = f.name
                dfs.append(df)
                logger.info(f"  ✅ {f.name}: {len(df)} 行")
            except Exception as e:
                logger.warning(f"  ⚠️ {f.name}: 读取失败 ({e})")
        
        if not dfs:
            logger.error("❌ 没有可合并的数据")
            return pd.DataFrame()
        
        # 合并 / Merge
        merged = pd.concat(dfs, ignore_index=True)
        logger.info(f"\n📊 合并后总行数: {len(merged)}")
        
        # 去重 / Deduplicate
        feature_cols = FeatureExtractor.FEATURE_NAMES
        before_dedup = len(merged)
        
        # 基于特征和标签去重 / Deduplicate based on features and label
        dedup_cols = feature_cols + ['label']
        existing_cols = [c for c in dedup_cols if c in merged.columns]
        merged = merged.drop_duplicates(subset=existing_cols, keep='last')
        
        after_dedup = len(merged)
        if before_dedup != after_dedup:
            logger.info(f"  去重: {before_dedup} → {after_dedup} (移除 {before_dedup - after_dedup} 重复行)")
        
        # 验证特征完整性 / Validate feature completeness
        missing_features = [f for f in FeatureExtractor.FEATURE_NAMES if f not in merged.columns]
        if missing_features:
            logger.warning(f"⚠️ 缺少特征: {missing_features}")
            for f in missing_features:
                merged[f] = 0.0
        
        # 统计 / Statistics
        logger.info(f"\n📊 合并数据统计:")
        logger.info(f"  总样本数: {len(merged)}")
        
        if '_source_file' in merged.columns:
            source_dist = merged['_source_file'].value_counts()
            logger.info(f"  来源分布:")
            for src, count in source_dist.items():
                logger.info(f"    - {src}: {count}")
        
        if 'label' in merged.columns:
            label_dist = merged['label'].value_counts().sort_index()
            logger.info(f"  标签分布:")
            label_names = ['亏损', '微利', '良好', '优秀']
            for label, count in label_dist.items():
                pct = count / len(merged) * 100
                name = label_names[int(label)] if 0 <= label < 4 else f'Class {label}'
                logger.info(f"    {int(label)} ({name}): {count} ({pct:.1f}%)")
        
        # 保存 / Save
        if output_name is None:
            timestamp = datetime.now().strftime('%Y%m%d')
            output_name = f'training_data_merged_{timestamp}.parquet'
        
        output_path = self.output_dir / output_name
        merged.to_parquet(output_path, index=False)
        logger.info(f"\n💾 已保存: {output_path}")
        
        # 同时更新默认训练数据 / Also update default training data
        default_path = self.output_dir / 'training_data_deep.parquet'
        merged.to_parquet(default_path, index=False)
        logger.info(f"💾 已更新默认数据: {default_path}")
        
        return merged


def main():
    parser = argparse.ArgumentParser(description='合并ButterQuant训练数据')
    parser.add_argument('--output', type=str, help='输出文件名')
    args = parser.parse_args()
    
    merger = DataMerger()
    df = merger.merge_all(output_name=args.output)
    
    if len(df) > 0:
        logger.info("\n✅ 数据合并完成!")
        logger.info("下一步: python ml/train_model.py")


if __name__ == "__main__":
    main()
