# -*- coding: utf-8 -*-
"""
ButterQuant 数据质量验证脚本 / Data Quality Validation Script
验证训练数据的完整性和质量 / Validate training data integrity and quality

用法 / Usage:
    python ml/validate_data.py
    python ml/validate_data.py --file training_data_deep.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import sys

# 添加项目路径 / Add project path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import FeatureExtractor, validate_feature_quality

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """训练数据验证器 / Training Data Validator"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.issues = []
        self.warnings = []
        
    def load_data(self) -> bool:
        """加载数据 / Load data"""
        logger.info(f"📥 加载数据: {self.data_path}")
        
        if not self.data_path.exists():
            logger.error(f"❌ 文件不存在: {self.data_path}")
            return False
        
        try:
            if self.data_path.suffix == '.parquet':
                self.df = pd.read_parquet(self.data_path)
            elif self.data_path.suffix == '.csv':
                self.df = pd.read_csv(self.data_path)
            else:
                logger.error(f"❌ 不支持的文件格式: {self.data_path.suffix}")
                return False
            
            logger.info(f"✅ 成功加载 {len(self.df)} 行数据")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载失败: {e}")
            return False
    
    def validate_shape(self):
        """验证数据形状 / Validate data shape"""
        logger.info("\n📐 检查数据形状...")
        
        n_rows, n_cols = self.df.shape
        logger.info(f"  行数: {n_rows}")
        logger.info(f"  列数: {n_cols}")
        
        # 最小样本数检查 / Minimum sample check
        if n_rows < 1000:
            self.warnings.append(f"样本数较少: {n_rows} (建议 > 5000)")
        elif n_rows < 5000:
            self.warnings.append(f"样本数偏少: {n_rows} (建议 > 5000)")
        else:
            logger.info(f"  ✅ 样本数充足")
    
    def validate_features(self):
        """验证特征完整性 / Validate feature completeness"""
        logger.info("\n🔍 检查特征完整性...")
        
        expected_features = FeatureExtractor.FEATURE_NAMES
        
        # 检查缺失特征 / Check missing features
        missing = []
        for feat in expected_features:
            if feat not in self.df.columns:
                missing.append(feat)
        
        if missing:
            self.issues.append(f"缺失特征: {missing}")
            logger.error(f"  ❌ 缺失 {len(missing)} 个特征: {missing}")
        else:
            logger.info(f"  ✅ 所有 {len(expected_features)} 个特征都存在")
        
        # 检查标签列 / Check label column
        if 'label' not in self.df.columns:
            self.issues.append("缺失标签列 'label'")
            logger.error(f"  ❌ 缺失标签列 'label'")
        else:
            logger.info(f"  ✅ 标签列存在")
    
    def validate_values(self):
        """验证数据值 / Validate data values"""
        logger.info("\n📊 检查数据值...")
        
        feature_cols = [c for c in FeatureExtractor.FEATURE_NAMES if c in self.df.columns]
        
        # NaN检查 / NaN check
        nan_counts = self.df[feature_cols].isna().sum()
        total_nan = nan_counts.sum()
        
        if total_nan > 0:
            nan_features = nan_counts[nan_counts > 0]
            self.warnings.append(f"发现 {total_nan} 个NaN值")
            logger.warning(f"  ⚠️ 发现 {total_nan} 个NaN值:")
            for feat, count in nan_features.items():
                logger.warning(f"    - {feat}: {count}")
        else:
            logger.info(f"  ✅ 无NaN值")
        
        # Inf检查 / Inf check
        inf_count = 0
        for col in feature_cols:
            inf_count += np.isinf(self.df[col]).sum()
        
        if inf_count > 0:
            self.issues.append(f"发现 {inf_count} 个Inf值")
            logger.error(f"  ❌ 发现 {inf_count} 个Inf值")
        else:
            logger.info(f"  ✅ 无Inf值")
        
        # 零方差检查 / Zero variance check
        zero_var_cols = []
        for col in feature_cols:
            if self.df[col].std() == 0:
                zero_var_cols.append(col)
        
        if zero_var_cols:
            self.warnings.append(f"零方差列: {zero_var_cols}")
            logger.warning(f"  ⚠️ 零方差列 (常数): {zero_var_cols}")
        else:
            logger.info(f"  ✅ 无零方差列")
    
    def validate_labels(self):
        """验证标签分布 / Validate label distribution"""
        logger.info("\n🏷️ 检查标签分布...")
        
        if 'label' not in self.df.columns:
            return
        
        label_counts = self.df['label'].value_counts().sort_index()
        total = len(self.df)
        
        logger.info("  标签分布:")
        label_names = ['亏损/Loss', '微利/Minor', '良好/Good', '优秀/Excellent']
        
        for label, count in label_counts.items():
            pct = count / total * 100
            name = label_names[label] if 0 <= label < 4 else f'Class {label}'
            logger.info(f"    {label} ({name}): {count:5d} ({pct:5.1f}%)")
        
        # 检查类别是否完整 / Check class completeness
        expected_labels = {0, 1, 2, 3}
        actual_labels = set(label_counts.index)
        missing_labels = expected_labels - actual_labels
        
        if missing_labels:
            self.warnings.append(f"缺少类别: {missing_labels}")
            logger.warning(f"  ⚠️ 缺少类别: {missing_labels}")
        else:
            logger.info(f"  ✅ 所有4个类别都存在")
        
        # 检查严重不平衡 / Check severe imbalance
        min_pct = label_counts.min() / total * 100
        if min_pct < 5:
            self.warnings.append(f"严重类别不平衡: 最小类别仅占 {min_pct:.1f}%")
            logger.warning(f"  ⚠️ 严重类别不平衡: 最小类别仅占 {min_pct:.1f}%")
    
    def validate_statistics(self):
        """统计信息 / Statistics"""
        logger.info("\n📈 特征统计信息...")
        
        feature_cols = [c for c in FeatureExtractor.FEATURE_NAMES if c in self.df.columns]
        stats = self.df[feature_cols].describe().T
        
        # 只显示关键统计 / Show key stats only
        logger.info(f"  特征范围预览 (前5个):")
        for col in feature_cols[:5]:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            mean_val = self.df[col].mean()
            logger.info(f"    {col}: [{min_val:.4f}, {max_val:.4f}], mean={mean_val:.4f}")
    
    def run_full_validation(self) -> dict:
        """运行完整验证 / Run full validation"""
        logger.info("=" * 60)
        logger.info("🔬 ButterQuant 训练数据验证 / Training Data Validation")
        logger.info("=" * 60)
        
        if not self.load_data():
            return {'status': 'error', 'message': '数据加载失败'}
        
        self.validate_shape()
        self.validate_features()
        self.validate_values()
        self.validate_labels()
        self.validate_statistics()
        
        # 汇总报告 / Summary report
        logger.info("\n" + "=" * 60)
        logger.info("📋 验证报告汇总 / Validation Summary")
        logger.info("=" * 60)
        
        if self.issues:
            logger.error(f"\n❌ 发现 {len(self.issues)} 个错误:")
            for i, issue in enumerate(self.issues, 1):
                logger.error(f"  {i}. {issue}")
        
        if self.warnings:
            logger.warning(f"\n⚠️ 发现 {len(self.warnings)} 个警告:")
            for i, warning in enumerate(self.warnings, 1):
                logger.warning(f"  {i}. {warning}")
        
        if not self.issues and not self.warnings:
            logger.info("\n✅ 数据验证通过! 未发现任何问题。")
            status = 'pass'
        elif self.issues:
            logger.error("\n❌ 数据验证失败! 请修复上述错误。")
            status = 'fail'
        else:
            logger.warning("\n⚠️ 数据验证通过, 但有警告需要注意。")
            status = 'pass_with_warnings'
        
        return {
            'status': status,
            'n_samples': len(self.df),
            'n_features': len([c for c in FeatureExtractor.FEATURE_NAMES if c in self.df.columns]),
            'issues': self.issues,
            'warnings': self.warnings
        }


def main():
    parser = argparse.ArgumentParser(description='验证ButterQuant训练数据质量')
    parser.add_argument('--file', type=str, default='ml/training_data_deep.parquet',
                        help='数据文件路径 (默认: ml/training_data_deep.parquet)')
    args = parser.parse_args()
    
    validator = DataValidator(args.file)
    result = validator.run_full_validation()
    
    # 返回状态码 / Return status code
    if result['status'] == 'fail':
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
