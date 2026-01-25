# -*- coding: utf-8 -*-
"""
ButterQuant 端到端测试脚本 / End-to-End Test Script
测试完整ML流程: 数据生成 → 特征提取 → 模型训练 → 推理 / Test complete ML flow

用法 / Usage:
    python ml/end_to_end_test.py
    python ml/end_to_end_test.py --quick  # 快速测试模式
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse

# 添加项目路径 / Add project path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EndToEndTest:
    """端到端测试器 / End-to-End Tester"""
    
    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results = {}
        self.start_time = time.time()
        
    def test_feature_extractor(self) -> bool:
        """测试特征提取器 / Test Feature Extractor"""
        logger.info("\n" + "=" * 50)
        logger.info("📊 测试1: 特征提取器 / Test 1: Feature Extractor")
        logger.info("=" * 50)
        
        try:
            from ml.features import FeatureExtractor, extract_features_v2
            
            # 创建模拟分析结果 / Create mock analysis result
            mock_analysis = {
                'fourier': {
                    'trend_slope': 0.05,
                    'dominant_period_days': 21,
                    'period_strength': 0.3
                },
                'arima': {
                    'mean_forecast': 150.0,
                    'confidence_interval_width': 10.0
                },
                'garch': {
                    'predicted_vol': 0.25,
                    'current_iv': 0.30,
                    'vol_mispricing': 0.20,
                    'iv_percentile': 0.65
                },
                'greeks': {
                    'delta': 0.01,
                    'gamma': 0.05,
                    'vega': 20.0,
                    'theta': -5.0
                },
                'butterfly': {
                    'max_profit': 100,
                    'max_loss': 50,
                    'profit_ratio': 2.0,
                    'prob_profit': 0.6,
                    'dte': 30
                }
            }
            
            # 提取特征 / Extract features
            features = extract_features_v2(mock_analysis)
            
            # 验证 / Validate
            assert len(features) == 23, f"特征数应为23, 实际: {len(features)}"
            assert all(k in features for k in FeatureExtractor.FEATURE_NAMES), "缺少必要特征"
            
            # 转换为数组 / Convert to array
            arr = FeatureExtractor.to_array(features)
            assert arr.shape == (23,), f"数组形状应为(23,), 实际: {arr.shape}"
            assert arr.dtype == np.float32, f"数据类型应为float32"
            
            logger.info(f"  ✅ 特征提取成功: {len(features)} 维")
            logger.info(f"  ✅ 数组转换成功: shape={arr.shape}")
            
            self.results['feature_extractor'] = True
            return True
            
        except Exception as e:
            logger.error(f"  ❌ 特征提取失败: {e}")
            self.results['feature_extractor'] = False
            return False
    
    def test_ml_inference(self) -> bool:
        """测试ML推理引擎 / Test ML Inference Engine"""
        logger.info("\n" + "=" * 50)
        logger.info("🤖 测试2: ML推理引擎 / Test 2: ML Inference Engine")
        logger.info("=" * 50)
        
        try:
            from backend.ml_inference import ModelInference, get_inference_engine
            from ml.features import FeatureExtractor
            
            # 获取引擎 / Get engine
            engine = ModelInference()
            version = engine.get_model_version()
            
            logger.info(f"  模型版本 / Model version: {version or '未加载'}")
            
            if version is None:
                logger.warning("  ⚠️ 模型未加载 (可能尚未训练)")
                logger.info("  → 跳过推理测试, 请先训练模型")
                self.results['ml_inference'] = 'skipped'
                return True
            
            # 生成随机特征 / Generate random features
            mock_features = {
                name: float(np.random.randn()) 
                for name in FeatureExtractor.FEATURE_NAMES
            }
            
            # 执行推理 / Run inference
            result = engine.predict_roi_distribution(mock_features)
            
            if result is None:
                logger.warning("  ⚠️ 推理返回None")
                self.results['ml_inference'] = False
                return False
            
            # 验证结果 / Validate result
            required_keys = ['prob_loss', 'prob_minor', 'prob_good', 'prob_excellent', 'expected_roi']
            for key in required_keys:
                assert key in result, f"缺少字段: {key}"
            
            # 概率和应为1 / Probabilities should sum to 1
            prob_sum = result['prob_loss'] + result['prob_minor'] + result['prob_good'] + result['prob_excellent']
            assert 0.99 <= prob_sum <= 1.01, f"概率和应为1, 实际: {prob_sum}"
            
            logger.info(f"  ✅ 推理成功!")
            logger.info(f"  - P(亏损): {result['prob_loss']:.2%}")
            logger.info(f"  - P(微利): {result['prob_minor']:.2%}")
            logger.info(f"  - P(良好): {result['prob_good']:.2%}")
            logger.info(f"  - P(优秀): {result['prob_excellent']:.2%}")
            logger.info(f"  - 期望ROI: {result['expected_roi']:.2%}")
            
            # 性能测试 / Performance test
            if not self.quick_mode:
                logger.info("\n  ⏱️ 性能测试 (100样本)...")
                start = time.time()
                for _ in range(100):
                    engine.predict_roi_distribution(mock_features)
                elapsed = time.time() - start
                avg_ms = elapsed / 100 * 1000
                logger.info(f"  - 平均延迟: {avg_ms:.2f}ms/样本")
                
                if avg_ms < 2.0:
                    logger.info(f"  ✅ 性能达标 (<2ms)")
                else:
                    logger.warning(f"  ⚠️ 性能未达标 (>2ms)")
            
            self.results['ml_inference'] = True
            return True
            
        except Exception as e:
            logger.error(f"  ❌ 推理测试失败: {e}")
            import traceback
            traceback.print_exc()
            self.results['ml_inference'] = False
            return False
    
    def test_data_validation(self) -> bool:
        """测试数据验证 / Test Data Validation"""
        logger.info("\n" + "=" * 50)
        logger.info("🔍 测试3: 数据验证工具 / Test 3: Data Validation")
        logger.info("=" * 50)
        
        try:
            from ml.validate_data import DataValidator
            from ml.features import FeatureExtractor
            
            # 创建测试数据 / Create test data
            n_samples = 100
            test_data = {
                name: np.random.randn(n_samples)
                for name in FeatureExtractor.FEATURE_NAMES
            }
            test_data['label'] = np.random.randint(0, 4, n_samples)
            
            # 保存临时文件 / Save temp file
            temp_path = Path(__file__).parent / '_temp_test_data.parquet'
            df = pd.DataFrame(test_data)
            df.to_parquet(temp_path)
            
            # 运行验证 / Run validation
            validator = DataValidator(str(temp_path))
            result = validator.run_full_validation()
            
            # 清理 / Cleanup
            temp_path.unlink()
            
            logger.info(f"\n  验证结果: {result['status']}")
            
            self.results['data_validation'] = True
            return True
            
        except Exception as e:
            logger.error(f"  ❌ 数据验证测试失败: {e}")
            self.results['data_validation'] = False
            return False
    
    def test_execution_engine_integration(self) -> bool:
        """测试执行引擎集成 / Test Execution Engine Integration"""
        logger.info("\n" + "=" * 50)
        logger.info("🚀 测试4: 执行引擎集成 / Test 4: Execution Engine Integration")
        logger.info("=" * 50)
        
        try:
            # 只测试导入和初始化,不实际连接TWS / Only test import and init, don't connect TWS
            from backend.execution_engine import ExecutionEngine
            
            logger.info("  正在初始化执行引擎 (不连接TWS)...")
            
            # 检查关键方法存在 / Check key methods exist
            engine = ExecutionEngine.__new__(ExecutionEngine)
            
            assert hasattr(ExecutionEngine, '_filter_ai_candidates'), "缺少_filter_ai_candidates方法"
            assert hasattr(ExecutionEngine, '_extract_features_from_analysis'), "缺少_extract_features_from_analysis方法"
            assert hasattr(ExecutionEngine, 'run_daily_execution'), "缺少run_daily_execution方法"
            
            logger.info("  ✅ 执行引擎结构验证通过")
            logger.info("    - _filter_ai_candidates() ✓")
            logger.info("    - _extract_features_from_analysis() ✓")
            logger.info("    - run_daily_execution() ✓")
            
            self.results['execution_engine'] = True
            return True
            
        except Exception as e:
            logger.error(f"  ❌ 执行引擎测试失败: {e}")
            self.results['execution_engine'] = False
            return False
    
    def test_training_data_exists(self) -> bool:
        """检查训练数据是否存在 / Check if training data exists"""
        logger.info("\n" + "=" * 50)
        logger.info("📁 测试5: 训练数据检查 / Test 5: Training Data Check")
        logger.info("=" * 50)
        
        data_path = Path(__file__).parent / 'training_data_deep.parquet'
        
        if data_path.exists():
            df = pd.read_parquet(data_path)
            logger.info(f"  ✅ 训练数据存在: {data_path.name}")
            logger.info(f"    - 样本数: {len(df)}")
            logger.info(f"    - 列数: {len(df.columns)}")
            self.results['training_data'] = True
            return True
        else:
            logger.warning(f"  ⚠️ 训练数据不存在")
            logger.info(f"    → 请运行: python ml/generate_simulated_data.py")
            self.results['training_data'] = 'not_exists'
            return True  # 不阻止测试通过 / Don't block test
    
    def run_all_tests(self):
        """运行所有测试 / Run all tests"""
        logger.info("=" * 60)
        logger.info("🧪 ButterQuant 端到端测试 / End-to-End Tests")
        logger.info("=" * 60)
        logger.info(f"模式 / Mode: {'快速' if self.quick_mode else '完整'}")
        
        tests = [
            ('特征提取', self.test_feature_extractor),
            ('ML推理', self.test_ml_inference),
            ('数据验证', self.test_data_validation),
            ('执行引擎', self.test_execution_engine_integration),
            ('训练数据', self.test_training_data_exists),
        ]
        
        passed = 0
        failed = 0
        skipped = 0
        
        for name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"测试 '{name}' 异常: {e}")
                failed += 1
        
        # 统计跳过的测试 / Count skipped tests
        for k, v in self.results.items():
            if v == 'skipped' or v == 'not_exists':
                skipped += 1
                passed -= 1  # 不算通过 / Don't count as passed
        
        # 汇总报告 / Summary
        elapsed = time.time() - self.start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("📋 测试报告汇总 / Test Summary")
        logger.info("=" * 60)
        logger.info(f"  通过 / Passed:  {passed}")
        logger.info(f"  失败 / Failed:  {failed}")
        logger.info(f"  跳过 / Skipped: {skipped}")
        logger.info(f"  耗时 / Time:    {elapsed:.1f}s")
        
        if failed == 0:
            logger.info("\n✅ 所有测试通过! / All tests passed!")
            return True
        else:
            logger.error(f"\n❌ {failed} 个测试失败! / {failed} tests failed!")
            return False


def main():
    parser = argparse.ArgumentParser(description='ButterQuant 端到端测试')
    parser.add_argument('--quick', action='store_true', help='快速测试模式 (跳过性能测试)')
    args = parser.parse_args()
    
    tester = EndToEndTest(quick_mode=args.quick)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
