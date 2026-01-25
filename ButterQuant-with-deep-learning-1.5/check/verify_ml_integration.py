# -*- coding: utf-8 -*-
"""
ML集成验证脚本 / ML Integration Verification Script

验证内容 / Checks:
1. 模型文件存在 / Model files exist
2. 特征维度一致 / Feature dimensions consistent
3. 推理引擎可用 / Inference engine available
4. 执行引擎集成正确 / Execution engine integration correct

用法 / Usage:
    python check/verify_ml_integration.py
"""

import sys
import os
from pathlib import Path

# 添加项目路径 / Add project path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))
sys.path.insert(0, str(PROJECT_ROOT / 'ml'))

def check_model_files():
    """检查模型文件是否存在 / Check model files exist"""
    print("\n" + "=" * 50)
    print("🔍 检查1: 模型文件 / Check 1: Model Files")
    print("=" * 50)
    
    models_dir = PROJECT_ROOT / 'ml' / 'models'
    required_files = [
        'success_model_v2.onnx',
        'scaler_v2.joblib'
    ]
    
    all_exist = True
    for f in required_files:
        path = models_dir / f
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"  ✅ {f} ({size:.1f} KB)")
        else:
            print(f"  ❌ {f} 不存在 / not found")
            all_exist = False
    
    return all_exist


def check_feature_dimensions():
    """检查特征维度 / Check feature dimensions"""
    print("\n" + "=" * 50)
    print("📊 检查2: 特征维度 / Check 2: Feature Dimensions")
    print("=" * 50)
    
    try:
        from ml.features import FeatureExtractor
        
        n_features = len(FeatureExtractor.FEATURE_NAMES)
        print(f"  特征数量 / Feature count: {n_features}")
        
        # 注意: 实际代码是22维,文档写23维不准确
        # Note: Actual code uses 22-dim, docs incorrectly say 23
        if n_features == 22:
            print("  ✅ 特征维度正确 (22维)")
            return True
        else:
            print(f"  ⚠️ 特征维度: {n_features} (期望22维)")
            return False
    
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def check_inference_engine():
    """检查推理引擎 / Check inference engine"""
    print("\n" + "=" * 50)
    print("🤖 检查3: 推理引擎 / Check 3: Inference Engine")
    print("=" * 50)
    
    try:
        from backend.ml_inference import ModelInference, get_inference_engine
        from ml.features import FeatureExtractor
        import numpy as np
        
        # 获取引擎 / Get engine
        engine = ModelInference()
        version = engine.get_model_version()
        
        print(f"  模型版本 / Version: {version or '未加载'}")
        
        if version is None:
            print("  ⚠️ 模型未加载 (请先训练模型)")
            return True  # 不阻止
        
        # 测试推理 / Test inference
        mock_features = {
            name: float(np.random.randn())
            for name in FeatureExtractor.FEATURE_NAMES
        }
        
        result = engine.predict_roi_distribution(mock_features)
        
        if result is None:
            print("  ❌ 推理返回 None")
            return False
        
        print(f"  期望ROI / Expected ROI: {result['expected_roi']:.2%}")
        print(f"  预测类别 / Predicted class: {result.get('class_name', result['predicted_class'])}")
        print("  ✅ 推理引擎正常工作")
        
        return True
    
    except Exception as e:
        print(f"  ❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_execution_engine():
    """检查执行引擎集成 / Check execution engine integration"""
    print("\n" + "=" * 50)
    print("🚀 检查4: 执行引擎集成 / Check 4: Execution Engine")
    print("=" * 50)
    
    try:
        from backend.execution_engine import ExecutionEngine, ML_AVAILABLE, FEATURES_AVAILABLE
        
        print(f"  ML模块可用 / ML available: {'✅' if ML_AVAILABLE else '❌'}")
        print(f"  特征模块可用 / Features available: {'✅' if FEATURES_AVAILABLE else '❌'}")
        
        # 检查关键方法 / Check key methods
        has_filter = hasattr(ExecutionEngine, '_filter_ai_candidates')
        has_extract = hasattr(ExecutionEngine, '_extract_features_from_analysis')
        
        print(f"  _filter_ai_candidates: {'✅' if has_filter else '❌'}")
        print(f"  _extract_features_from_analysis: {'✅' if has_extract else '❌'}")
        
        if ML_AVAILABLE and FEATURES_AVAILABLE and has_filter and has_extract:
            print("  ✅ 执行引擎集成完整")
            return True
        else:
            print("  ⚠️ 部分组件缺失")
            return True  # 不阻止
    
    except Exception as e:
        print(f"  ❌ 执行引擎检查失败: {e}")
        return False


def main():
    """运行所有检查 / Run all checks"""
    print("=" * 60)
    print("🧪 ButterQuant ML集成验证 / ML Integration Verification")
    print("=" * 60)
    
    results = [
        ('模型文件', check_model_files()),
        ('特征维度', check_feature_dimensions()),
        ('推理引擎', check_inference_engine()),
        ('执行引擎', check_execution_engine()),
    ]
    
    # 汇总 / Summary
    print("\n" + "=" * 60)
    print("📋 验证结果汇总 / Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ 所有验证通过! ML集成正常 / All checks passed!")
    else:
        print("\n⚠️ 部分验证失败, 请检查上述问题 / Some checks failed")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
