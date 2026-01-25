"""
ButterQuant ML Feature Engineering Module V2.0
特征工程模块 - 扩展到23维特征向量

关键改动:
1. ❌ 移除 total_score (数据泄露)
2. ✅ 新增 6 个低成本特征
3. ✅ 确保所有特征无NaN/Inf

特征列表 (23维):
-------------------
原有特征 (16个):
  - Fourier分析: trend_slope, dominant_period, period_strength
  - ARIMA预测: forecast_price
  - GARCH波动率: predicted_vol, current_iv, vol_mispricing, iv_percentile
  - Greeks: delta, gamma, vega, theta
  - 策略参数: max_profit, max_loss, profit_ratio, prob_profit

新增特征 (6个):
  - skew_estimate: IV偏度估计
  - momentum_7d: 7日价格动量
  - vol_concentration: 成交量集中度
  - dte_factor: DTE归一化因子
  - price_stability: 价格稳定性
  - gamma_theta_ratio: Gamma/Theta比率
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """特征提取器 - V2.0"""
    
    # 定义特征顺序 (重要: 训练和推理必须一致)
    FEATURE_NAMES = [
        # Fourier (3)
        'trend_slope',
        'dominant_period',
        'period_strength',
        
        # ARIMA (1)
        'forecast_price',
        
        # GARCH (4)
        'predicted_vol',
        'current_iv',
        'vol_mispricing',
        'iv_percentile',
        
        # Greeks (4)
        'delta',
        'gamma',
        'vega',
        'theta',
        
        # Strategy (4)
        'max_profit',
        'max_loss',
        'profit_ratio',
        'prob_profit',
        
        # New Features (6)
        'skew_estimate',
        'momentum_7d',
        'vol_concentration',
        'dte_factor',
        'price_stability',
        'gamma_theta_ratio'
    ]
    
    @staticmethod
    def extract_from_analysis(analysis_result: Dict, price_history: Optional[pd.DataFrame] = None) -> Dict:
        """
        从ButterflyAnalyzer结果提取特征
        
        参数:
            analysis_result: ButterflyAnalyzer.full_analysis()的返回字典
            price_history: 可选的价格历史DataFrame
        
        返回:
            features_dict: 23个特征的字典
        """
        
        # 解包分析结果
        fourier = analysis_result.get('fourier', {})
        arima = analysis_result.get('arima', {})
        garch = analysis_result.get('garch', {})
        greeks = analysis_result.get('greeks', {})
        butterfly = analysis_result.get('butterfly', {})
        
        # 基础特征 (16个)
        base_features = {
            # Fourier分析
            'trend_slope': _safe_get(fourier, 'trend_slope', 0.0),
            'dominant_period': _safe_get(fourier, 'dominant_period_days', 0.0),
            'period_strength': _safe_get(fourier, 'period_strength', 0.0),
            
            # ARIMA预测
            'forecast_price': _safe_get(arima, 'mean_forecast', 0.0),
            
            # GARCH波动率
            'predicted_vol': _safe_get(garch, 'predicted_vol', 0.0),
            'current_iv': _safe_get(garch, 'current_iv', 0.0),
            'vol_mispricing': _safe_get(garch, 'vol_mispricing', 0.0),
            'iv_percentile': _safe_get(garch, 'iv_percentile', 0.5),
            
            # Greeks
            'delta': _safe_get(greeks, 'delta', 0.0),
            'gamma': _safe_get(greeks, 'gamma', 0.0),
            'vega': _safe_get(greeks, 'vega', 0.0),
            'theta': _safe_get(greeks, 'theta', 0.0),
            
            # 蝴蝶策略参数
            'max_profit': _safe_get(butterfly, 'max_profit', 0.0),
            'max_loss': _safe_get(butterfly, 'max_loss', 0.0),
            'profit_ratio': _safe_get(butterfly, 'profit_ratio', 0.0),
            'prob_profit': _safe_get(butterfly, 'prob_profit', 0.5)
        }
        
        # 新增特征 (6个)
        new_features = FeatureExtractor._extract_new_features(
            analysis_result, 
            price_history
        )
        
        # 合并并验证
        features = {**base_features, **new_features}
        features = FeatureExtractor._validate_features(features)
        
        return features
    
    @staticmethod
    def _extract_new_features(analysis_result: Dict, price_history: Optional[pd.DataFrame]) -> Dict:
        """提取6个新增特征"""
        
        garch = analysis_result.get('garch', {})
        greeks = analysis_result.get('greeks', {})
        butterfly = analysis_result.get('butterfly', {})
        arima = analysis_result.get('arima', {})
        
        # 1. IV Skew估计
        skew_estimate = _calculate_skew_estimate(garch)
        
        # 2. 价格动量
        momentum_7d = _calculate_momentum(price_history, days=7)
        
        # 3. 成交量集中度
        vol_concentration = _calculate_volume_concentration(price_history, window=5)
        
        # 4. DTE因子
        dte = _safe_get(butterfly, 'dte', 30)
        dte_factor = min(dte / 30.0, 2.0)
        
        # 5. 价格稳定性
        price_stability = _calculate_price_stability(arima, price_history)
        
        # 6. Gamma/Theta比率
        gamma_theta_ratio = _calculate_gamma_theta_ratio(greeks)
        
        return {
            'skew_estimate': skew_estimate,
            'momentum_7d': momentum_7d,
            'vol_concentration': vol_concentration,
            'dte_factor': dte_factor,
            'price_stability': price_stability,
            'gamma_theta_ratio': gamma_theta_ratio
        }
    
    @staticmethod
    def _validate_features(features: Dict) -> Dict:
        """验证并清理特征"""
        validated = {}
        
        for name in FeatureExtractor.FEATURE_NAMES:
            value = features.get(name, 0.0)
            
            if not np.isfinite(value):
                logger.warning(f"特征 {name} 包含非有限值: {value}, 替换为0")
                value = 0.0
            
            validated[name] = float(value)
        
        return validated


# ==================== 辅助函数 ====================

def _safe_get(dictionary: Dict, key: str, default=0.0):
    """安全获取字典值"""
    value = dictionary.get(key, default)
    return value if np.isfinite(value) else default


def _calculate_skew_estimate(garch: Dict) -> float:
    """IV Skew估计"""
    vol_mispricing = _safe_get(garch, 'vol_mispricing', 0.0)
    return vol_mispricing * 100.0


def _calculate_momentum(price_history: Optional[pd.DataFrame], days: int = 7) -> float:
    """价格动量"""
    if price_history is None or len(price_history) < days:
        return 0.0
    
    try:
        current_price = price_history['Close'].iloc[-1]
        past_price = price_history['Close'].iloc[-days]
        
        if past_price == 0:
            return 0.0
        
        momentum = (current_price - past_price) / past_price
        return float(np.clip(momentum, -1.0, 1.0))
    except Exception as e:
        logger.debug(f"计算动量失败: {e}")
        return 0.0


def _calculate_volume_concentration(price_history: Optional[pd.DataFrame], window: int = 5) -> float:
    """成交量集中度"""
    if price_history is None or len(price_history) < window:
        return 1.0
    
    try:
        recent_volume = price_history['Volume'].iloc[-window:]
        
        if recent_volume.mean() == 0:
            return 1.0
        
        concentration = recent_volume.max() / recent_volume.mean()
        return float(np.clip(concentration, 0.5, 10.0))
    except Exception as e:
        logger.debug(f"计算成交量集中度失败: {e}")
        return 1.0


def _calculate_price_stability(arima: Dict, price_history: Optional[pd.DataFrame]) -> float:
    """价格稳定性"""
    # 方法1: ARIMA置信区间
    ci_width = _safe_get(arima, 'confidence_interval_width', None)
    if ci_width is not None and ci_width > 0:
        return 1.0 / (ci_width + 1e-6)
    
    # 方法2: 历史波动率
    if price_history is not None and len(price_history) >= 10:
        try:
            price_std = price_history['Close'].std()
            price_mean = price_history['Close'].mean()
            
            if price_mean > 0:
                cv = price_std / price_mean
                return 1.0 / (cv + 1e-6)
        except Exception:
            pass
    
    return 1.0


def _calculate_gamma_theta_ratio(greeks: Dict) -> float:
    """Gamma/Theta比率"""
    gamma = _safe_get(greeks, 'gamma', 0.0)
    theta = _safe_get(greeks, 'theta', -0.01)
    
    theta_abs = abs(theta)
    
    if theta_abs < 1e-6:
        return 0.0
    
    ratio = abs(gamma) / theta_abs
    return float(np.clip(ratio, 0.0, 100.0))


# ==================== 便捷函数 ====================

def extract_features_v2(analysis_result: Dict, price_history: Optional[pd.DataFrame] = None) -> Dict:
    """快捷函数: 提取特征"""
    return FeatureExtractor.extract_from_analysis(analysis_result, price_history)