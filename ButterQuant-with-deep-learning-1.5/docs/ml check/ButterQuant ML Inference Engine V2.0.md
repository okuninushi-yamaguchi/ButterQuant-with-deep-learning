"""
ButterQuant ML Inference Engine V2.0
4分类推理引擎 + 期望ROI计算

核心功能:
1. 4分类概率分布预测
2. 期望ROI计算: E[ROI] = Σ(p_i × roi_i)
3. ONNX Runtime优化 (CPU/CUDA)
4. 高性能推理 (<2ms/样本)
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional
import logging

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("⚠️ onnxruntime未安装")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLInferenceEngine:
    """ML推理引擎 V2.0"""
    
    # ROI分级定义
    ROI_VALUES = [0.0, 0.05, 0.20, 0.40]  # 对应0/1/2/3类
    
    def __init__(self, model_dir: str = "ml", use_cuda: bool = True):
        self.model_dir = Path(model_dir)
        self.use_cuda = use_cuda and ONNX_AVAILABLE
        
        self._session = None
        self._scaler = None
        
        from ml.features import FeatureExtractor
        self.feature_names = FeatureExtractor.FEATURE_NAMES
        
        self.load_model()
    
    def load_model(self):
        """加载模型和预处理器"""
        logger.info("🔄 加载ML推理引擎...")
        
        # 加载Scaler
        scaler_path = self.model_dir / "scaler_v2.pkl"
        if not scaler_path.exists():
            scaler_path = self.model_dir / "scaler.pkl"
        
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler加载成功: {scaler_path}")
        else:
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        # 加载ONNX模型
        onnx_path = self.model_dir / "success_model_v2.onnx"
        if ONNX_AVAILABLE and onnx_path.exists():
            providers = []
            if self.use_cuda:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            self._session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            actual_provider = self._session.get_providers()[0]
            logger.info(f"✅ ONNX模型加载成功: {onnx_path}")
            logger.info(f"   Provider: {actual_provider}")
            
            if 'CUDA' in actual_provider:
                logger.info(f"   🚀 GPU加速已启用")
        else:
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    def predict_roi_distribution(self, features_dict: Dict) -> Optional[Dict]:
        """
        预测ROI概率分布
        
        返回:
            {
                'prob_loss': float,
                'prob_minor': float,
                'prob_good': float,
                'prob_excellent': float,
                'expected_roi': float,
                'confidence': float,
                'predicted_class': int,
                'class_name': str
            }
        """
        if self._session is None or self._scaler is None:
            logger.error("❌ 模型未加载")
            return None
        
        try:
            # 构建特征向量
            X = np.array([
                features_dict.get(name, 0.0) 
                for name in self.feature_names
            ], dtype=np.float32).reshape(1, -1)
            
            # 处理异常值
            if not np.isfinite(X).all():
                logger.warning("⚠️ 特征包含NaN/Inf")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 标准化
            X_scaled = self._scaler.transform(X).astype(np.float32)
            
            # ONNX推理
            input_name = self._session.get_inputs()[0].name
            logits = self._session.run(None, {input_name: X_scaled})[0][0]
            
            # Softmax
            probs = self._softmax(logits)
            
            # 计算期望ROI
            expected_roi = np.dot(probs, self.ROI_VALUES)
            
            # 预测类别
            predicted_class = int(np.argmax(probs))
            confidence = float(np.max(probs))
            
            return {
                'prob_loss': float(probs[0]),
                'prob_minor': float(probs[1]),
                'prob_good': float(probs[2]),
                'prob_excellent': float(probs[3]),
                'expected_roi': float(expected_roi),
                'confidence': confidence,
                'predicted_class': predicted_class,
                'class_name': self._get_class_name(predicted_class)
            }
        
        except Exception as e:
            logger.error(f"❌ 推理失败: {e}")
            return None
    
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
    
    @staticmethod
    def _get_class_name(class_idx: int) -> str:
        """获取类别名称"""
        names = ['loss', 'minor', 'good', 'excellent']
        return names[class_idx] if 0 <= class_idx < 4 else 'unknown'


# ==================== 全局单例 ====================

_global_engine = None

def get_inference_engine(model_dir: str = "ml", use_cuda: bool = True) -> MLInferenceEngine:
    """获取全局推理引擎 (单例模式)"""
    global _global_engine
    
    if _global_engine is None:
        _global_engine = MLInferenceEngine(model_dir, use_cuda)
    
    return _global_engine


def predict_roi(features_dict: Dict) -> Optional[Dict]:
    """快捷函数: 直接预测ROI分布"""
    engine = get_inference_engine()
    return engine.predict_roi_distribution(features_dict)


def should_execute_trade(features_dict: Dict, min_expected_roi: float = 0.15) -> bool:
    """判断是否应该执行交易"""
    result = predict_roi(features_dict)
    
    if result is None:
        return False
    
    return result['expected_roi'] >= min_expected_roi


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("ML推理引擎 V2.0 - 测试")
    print("=" * 60)
    
    # 初始化引擎
    engine = MLInferenceEngine(use_cuda=True)
    
    # 测试推理
    print("\n🧪 测试推理...")
    
    from ml.features import FeatureExtractor
    mock_features = {name: np.random.randn() for name in FeatureExtractor.FEATURE_NAMES}
    
    result = engine.predict_roi_distribution(mock_features)
    
    if result:
        print("\n✅ 推理成功!")
        print(f"\n预测结果:")
        print(f"  P(亏损):   {result['prob_loss']:.2%}")
        print(f"  P(微利):   {result['prob_minor']:.2%}")
        print(f"  P(良好):   {result['prob_good']:.2%}")
        print(f"  P(优秀):   {result['prob_excellent']:.2%}")
        print(f"\n  期望ROI:   {result['expected_roi']:.2%}")
        print(f"  预测类别:  {result['class_name']} (置信度: {result['confidence']:.2%})")
        
        if result['expected_roi'] >= 0.15:
            print(f"\n  ✅ 建议执行 (期望ROI > 15%)")
        else:
            print(f"\n  ❌ 不建议执行 (期望ROI < 15%)")
    else:
        print("\n❌ 推理失败")