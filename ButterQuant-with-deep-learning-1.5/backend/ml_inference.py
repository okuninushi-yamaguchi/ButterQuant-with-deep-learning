# -*- coding: utf-8 -*-
"""
ButterQuant ML Inference Engine V2.0 / ButterQuant ML 推理引擎 V2.0
4分类推理引擎 + 期望ROI计算 / 4-class inference + Expected ROI calculation

核心功能 / Core Features:
1. 从二分类概率 → 4分类概率分布 / Binary → 4-class probability distribution
2. 计算期望ROI: E[ROI] = Σ(p_i × roi_i) / Calculate expected ROI
3. 支持ONNX Runtime (CPU/CUDA) / ONNX Runtime support
4. 高性能推理 (<2ms/样本) / High-performance inference

使用示例 / Example:
    engine = MLInferenceEngine()
    result = engine.predict_roi_distribution(features_dict)
    
    if result['expected_roi'] > 0.15:
        execute_trade()  # 期望ROI > 15%, 执行交易 / Execute trade
"""

import numpy as np
import joblib
import logging
import os
from pathlib import Path
from typing import Dict, Optional, List

# 使用 onnxruntime 进行推理 / Use onnxruntime for inference
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# 设置日志 / Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """
    ML推理引擎 (单例模式) / ML Inference Engine (Singleton)
    
    支持两种模型 / Supports two models:
    - V2 (4分类): 输出概率分布和期望ROI / Outputs probability distribution and expected ROI
    - V1 (二分类): 向后兼容,输出成功概率 / Backward compatible, outputs success probability
    """
    
    # ROI分级定义 (对应0/1/2/3类别) / ROI levels (for classes 0/1/2/3)
    ROI_VALUES = [-0.10, 0.05, 0.20, 0.40]  # 亏损, 微利5%, 良好20%, 优秀40%
    
    _instance = None
    _session_v2 = None
    _scaler_v2 = None
    _session_v1 = None
    _scaler_v1 = None
    _model_version = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelInference, cls).__new__(cls)
            cls._instance._load_resources()
        return cls._instance

    def _load_resources(self):
        """加载模型和缩放器资源 / Load model and scaler resources"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 尝试加载V2模型 (4分类) / Try loading V2 model (4-class)
        onnx_path_v2 = os.path.join(base_dir, 'ml', 'models', 'success_model_v2.onnx')
        scaler_path_v2 = os.path.join(base_dir, 'ml', 'models', 'scaler_v2.joblib')
        
        # 备用scaler路径 / Alternative scaler path
        scaler_path_v2_alt = os.path.join(base_dir, 'ml', 'models', 'scaler_v2.pkl')
        
        # 尝试加载V1模型 (二分类) / Try loading V1 model (binary)
        onnx_path_v1 = os.path.join(base_dir, 'ml', 'models', 'success_model.onnx')
        scaler_path_v1 = os.path.join(base_dir, 'ml', 'models', 'scaler.joblib')
        
        try:
            # 优先加载V2 / Prefer V2
            if os.path.exists(onnx_path_v2):
                # 尝试多个scaler路径 / Try multiple scaler paths
                if os.path.exists(scaler_path_v2):
                    self._scaler_v2 = joblib.load(scaler_path_v2)
                elif os.path.exists(scaler_path_v2_alt):
                    self._scaler_v2 = joblib.load(scaler_path_v2_alt)
                
                if self._scaler_v2 is not None and HAS_ONNX:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
                    self._session_v2 = ort.InferenceSession(onnx_path_v2, providers=providers)
                    self._model_version = 'V2'
                    logger.info(f"✅ V2模型加载成功 / V2 Model loaded with {self._session_v2.get_providers()[0]}")
            
            # 同时加载V1用于向后兼容 / Also load V1 for backward compatibility
            if os.path.exists(onnx_path_v1) and os.path.exists(scaler_path_v1):
                self._scaler_v1 = joblib.load(scaler_path_v1)
                
                if HAS_ONNX:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
                    self._session_v1 = ort.InferenceSession(onnx_path_v1, providers=providers)
                    
                    if self._model_version is None:
                        self._model_version = 'V1'
                        logger.info(f"✅ V1模型加载成功 / V1 Model loaded (fallback)")
            
            if self._model_version is None:
                logger.warning("⚠️ 未找到任何ML模型,进入回退模式 / No ML model found, running in fallback mode")
                    
        except Exception as e:
            logger.error(f"加载 ML 资源失败: {e} / Failed to load ML resources")
            self._session_v2 = None
            self._session_v1 = None

    def predict_roi_distribution(self, features_dict: Dict) -> Optional[Dict]:
        """
        预测4个等级的概率分布 (V2模型) / Predict 4-class probability distribution (V2 model)
        
        输入 / Input: 特征字典 (23维) / Feature dictionary (23-dim)
        输出 / Output: {
            'prob_loss': float,      # P(亏损) / P(Loss)
            'prob_minor': float,     # P(微利) / P(Minor profit)
            'prob_good': float,      # P(良好) / P(Good)
            'prob_excellent': float, # P(优秀) / P(Excellent)
            'expected_roi': float,   # 期望ROI / Expected ROI
            'confidence': float,     # 预测置信度 / Prediction confidence
            'predicted_class': int,  # 预测类别 / Predicted class
            'class_name': str        # 类别名称 / Class name
        }
        """
        if self._session_v2 is None or self._scaler_v2 is None:
            # V2模型不可用,尝试回退到V1 / V2 not available, fallback to V1
            prob = self.predict_success_probability(features_dict)
            if prob is not None:
                # 将二分类结果转换为4分类近似 / Convert binary to 4-class approximation
                return {
                    'prob_loss': 1.0 - prob,
                    'prob_minor': prob * 0.4,
                    'prob_good': prob * 0.4,
                    'prob_excellent': prob * 0.2,
                    'expected_roi': prob * 0.15,  # 近似 / Approximate
                    'confidence': max(prob, 1.0 - prob),
                    'predicted_class': 2 if prob > 0.5 else 0,
                    'class_name': 'good' if prob > 0.5 else 'loss'
                }
            return None

        try:
            # 导入特征名列表 / Import feature names
            try:
                from ml.features import FeatureExtractor
                feature_cols = FeatureExtractor.FEATURE_NAMES
            except ImportError:
                # 回退到硬编码列表 / Fallback to hardcoded list
                feature_cols = [
                    'trend_slope', 'dominant_period', 'period_strength', 'forecast_price',
                    'predicted_vol', 'current_iv', 'vol_mispricing', 'iv_percentile',
                    'delta', 'gamma', 'vega', 'theta', 'max_profit', 'max_loss',
                    'profit_ratio', 'prob_profit',
                    'skew_estimate', 'momentum_7d', 'vol_concentration', 
                    'dte_factor', 'price_stability', 'gamma_theta_ratio'
                ]
            
            # 构建特征向量 / Build feature vector
            feature_vector = []
            for col in feature_cols:
                val = features_dict.get(col, 0.0)
                if val is None:
                    val = 0.0
                # 处理无穷值 / Handle infinite values
                if not np.isfinite(val):
                    val = 0.0
                feature_vector.append(float(val))
            
            # 变形和缩放 / Reshape & Scale
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self._scaler_v2.transform(X).astype(np.float32)
            
            # ONNX推理 / ONNX Inference
            inputs = {self._session_v2.get_inputs()[0].name: X_scaled}
            outputs = self._session_v2.run(None, inputs)
            
            # 输出是logits,需要softmax / Output is logits, need softmax
            logits = outputs[0][0]  # shape: (4,)
            exp_logits = np.exp(logits - np.max(logits))  # 数值稳定 / Numerical stability
            probs = exp_logits / np.sum(exp_logits)
            
            # 计算期望ROI / Calculate expected ROI
            expected_roi = float(np.dot(probs, self.ROI_VALUES))
            
            # 预测类别和置信度 / Predicted class and confidence
            predicted_class = int(np.argmax(probs))
            confidence = float(np.max(probs))
            
            return {
                'prob_loss': float(probs[0]),
                'prob_minor': float(probs[1]),
                'prob_good': float(probs[2]),
                'prob_excellent': float(probs[3]),
                'expected_roi': expected_roi,
                'confidence': confidence,
                'predicted_class': predicted_class,
                'class_name': self._get_class_name(predicted_class)
            }
            
        except Exception as e:
            logger.error(f"V2预测失败: {e} / V2 Prediction failed")
            return None

    def predict_success_probability(self, features_dict: Dict) -> Optional[float]:
        """
        [向后兼容] 预测蝶式策略的成功概率 (V1模型) / [Backward compatible] Predict success probability
        
        输入 / Input: 特征字典 / Feature dictionary
        输出 / Output: 浮点数 (0.0 to 1.0) / Float (0.0 to 1.0)
        """
        # 如果只有V2可用,从V2结果计算 / If only V2 available, compute from V2 result
        if self._session_v1 is None and self._session_v2 is not None:
            result = self.predict_roi_distribution(features_dict)
            if result:
                # 成功概率 = 1 - P(亏损) / Success prob = 1 - P(Loss)
                return 1.0 - result['prob_loss']
            return None
        
        if self._session_v1 is None or self._scaler_v1 is None:
            return None

        try:
            # V1特征顺序 (17维,包含total_score) / V1 feature order (17-dim, includes total_score)
            feature_cols = [
                'trend_slope', 'dominant_period', 'period_strength', 'forecast_price',
                'predicted_vol', 'current_iv', 'vol_mispricing', 'iv_percentile',
                'delta', 'gamma', 'vega', 'theta', 'max_profit', 'max_loss',
                'profit_ratio', 'prob_profit', 'total_score'
            ]
            
            # 按正确顺序将字典转换为数组 / Convert dictionary to array in correct order
            feature_vector = []
            for col in feature_cols:
                val = features_dict.get(col, 0.0) 
                if val is None:
                    val = 0.0
                feature_vector.append(float(val))

            # 变形和缩放 / Reshape & Scale
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self._scaler_v1.transform(X).astype(np.float32)
            
            # 执行推理 / Run Inference
            inputs = {self._session_v1.get_inputs()[0].name: X_scaled}
            outputs = self._session_v1.run(None, inputs)
            
            prob = float(outputs[0][0][0])
            return prob
            
        except Exception as e:
            logger.error(f"V1预测失败: {e} / V1 Prediction failed")
            return None
    
    def batch_predict(self, features_list: List[Dict]) -> List[Optional[Dict]]:
        """
        批量推理 / Batch inference
        
        参数 / Parameters:
            features_list: List[Dict] - 特征字典列表 / List of feature dicts
        
        返回 / Returns:
            List[Dict] - 预测结果列表 / List of prediction results
        """
        return [self.predict_roi_distribution(f) for f in features_list]
    
    def benchmark(self, n_samples: int = 1000):
        """
        性能基准测试 / Performance benchmark
        
        参数 / Parameters:
            n_samples: 测试样本数 / Number of test samples
        """
        import time
        
        logger.info(f"\n⏱️ 性能基准测试 ({n_samples} 样本) / Performance benchmark")
        
        # 生成随机特征 / Generate random features
        try:
            from ml.features import FeatureExtractor
            feature_names = FeatureExtractor.FEATURE_NAMES
        except ImportError:
            feature_names = ['trend_slope', 'dominant_period', 'period_strength', 'forecast_price',
                           'predicted_vol', 'current_iv', 'vol_mispricing', 'iv_percentile',
                           'delta', 'gamma', 'vega', 'theta', 'max_profit', 'max_loss',
                           'profit_ratio', 'prob_profit', 'skew_estimate', 'momentum_7d',
                           'vol_concentration', 'dte_factor', 'price_stability', 'gamma_theta_ratio']
        
        dummy_features = {name: np.random.randn() for name in feature_names}
        
        # 预热 / Warmup
        for _ in range(10):
            self.predict_roi_distribution(dummy_features)
        
        # 测试 / Test
        start = time.time()
        for _ in range(n_samples):
            self.predict_roi_distribution(dummy_features)
        elapsed = time.time() - start
        
        avg_time = elapsed / n_samples * 1000  # ms
        throughput = n_samples / elapsed
        
        logger.info(f"   总耗时 / Total: {elapsed:.2f}s")
        logger.info(f"   平均延迟 / Avg latency: {avg_time:.2f}ms/样本")
        logger.info(f"   吞吐量 / Throughput: {throughput:.0f} 样本/秒")
        
        if avg_time < 2.0:
            logger.info(f"   ✅ 性能达标 (目标<2ms) / Performance OK")
        else:
            logger.warning(f"   ⚠️ 性能未达标 (目标<2ms) / Performance below target")
    
    def get_model_version(self) -> Optional[str]:
        """获取当前加载的模型版本 / Get loaded model version"""
        return self._model_version
    
    @staticmethod
    def _get_class_name(class_idx: int) -> str:
        """获取类别名称 / Get class name"""
        names = ['loss', 'minor', 'good', 'excellent']
        return names[class_idx] if 0 <= class_idx < 4 else 'unknown'


class ModelInferenceWithCache(ModelInference):
    """
    带缓存的推理引擎 / Inference engine with caching
    
    对于相同的特征, 直接返回缓存结果 / Returns cached results for same features
    """
    
    def __init__(self, cache_size: int = 1000):
        super().__new__(ModelInference)  # 使用父类单例 / Use parent singleton
        self._cache = {}
        self._cache_size = cache_size
    
    def predict_roi_distribution(self, features_dict: Dict) -> Optional[Dict]:
        """带缓存的推理 / Cached inference"""
        # 生成缓存key / Generate cache key
        cache_key = self._make_cache_key(features_dict)
        
        # 检查缓存 / Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 推理 / Inference
        result = super().predict_roi_distribution(features_dict)
        
        # 存入缓存 / Store in cache
        if result is not None:
            if len(self._cache) >= self._cache_size:
                # LRU: 删除最早的项 / Delete oldest item
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
        
        return result
    
    def _make_cache_key(self, features_dict: Dict) -> str:
        """生成缓存key / Generate cache key"""
        try:
            from ml.features import FeatureExtractor
            feature_names = FeatureExtractor.FEATURE_NAMES
        except ImportError:
            feature_names = list(features_dict.keys())
        values = tuple(features_dict.get(name, 0.0) for name in feature_names)
        return str(hash(values))
    
    def clear_cache(self):
        """清空缓存 / Clear cache"""
        self._cache.clear()
        logger.info("🗑️ 缓存已清空 / Cache cleared")


# ==================== 便捷函数 / Convenience Functions ====================

_global_engine = None


def get_inference_engine(use_cache: bool = False) -> ModelInference:
    """
    获取全局推理引擎 (单例模式) / Get global inference engine (singleton)
    
    使用示例 / Example:
        engine = get_inference_engine()
        result = engine.predict_roi_distribution(features)
    """
    global _global_engine
    
    if _global_engine is None:
        if use_cache:
            _global_engine = ModelInferenceWithCache()
        else:
            _global_engine = ModelInference()
    
    return _global_engine


def predict_roi(features_dict: Dict) -> Optional[Dict]:
    """
    快捷函数: 直接预测ROI分布 / Shortcut: Predict ROI distribution directly
    
    参数 / Parameters:
        features_dict: 特征字典 / Feature dictionary
    
    返回 / Returns:
        预测结果字典 / Prediction result dictionary
    """
    engine = get_inference_engine()
    return engine.predict_roi_distribution(features_dict)


def should_execute_trade(features_dict: Dict, min_expected_roi: float = 0.15) -> bool:
    """
    判断是否应该执行交易 / Determine if trade should be executed
    
    参数 / Parameters:
        features_dict: 特征字典 / Feature dictionary
        min_expected_roi: 最小期望ROI阈值 (默认15%) / Min expected ROI threshold (default 15%)
    
    返回 / Returns:
        bool: True表示应该执行 / True means should execute
    """
    result = predict_roi(features_dict)
    
    if result is None:
        return False
    
    return result['expected_roi'] >= min_expected_roi


# 单例实例 / Singleton instance
ml_engine = ModelInference()
