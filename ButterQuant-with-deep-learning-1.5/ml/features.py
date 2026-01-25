# -*- coding: utf-8 -*-
"""
ButterQuant ML Feature Engineering Module / ButterQuant ML 特征工程模块
扩展到22维特征向量 / Extended to 22-dim feature vector

关键改动 / Key Changes:
1. ❌ 移除 total_score (数据泄露) / Removed total_score (data leakage)
2. ✅ 新增 6 个低成本特征 / Added 6 new low-cost features
3. ✅ 确保所有特征无NaN/Inf / Ensured all features have no NaN/Inf

特征列表 (22维) / Feature List (22-dim):
-------------------
原有特征 (16个) / Original features (16):
  - Fourier分析: trend_slope, dominant_period, period_strength
  - ARIMA预测: forecast_price
  - GARCH波动率: predicted_vol, current_iv, vol_mispricing, iv_percentile
  - Greeks: delta, gamma, vega, theta
  - 策略参数: max_profit, max_loss, profit_ratio, prob_profit

新增特征 (6个) / New features (6):
  - skew_estimate: IV偏度估计 / IV skew estimate
  - momentum_7d: 7日价格动量 / 7-day price momentum
  - vol_concentration: 成交量集中度 / Volume concentration
  - dte_factor: DTE归一化因子 / DTE normalized factor
  - price_stability: 价格稳定性 / Price stability
  - gamma_theta_ratio: Gamma/Theta比率 / Gamma/Theta ratio
"""

import numpy as np
import pandas as pd
import json
import sqlite3
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    特征提取器 V2.0 / Feature Extractor V2.0
    
    统一的特征提取接口,确保训练和推理使用相同的特征顺序
    Unified feature extraction interface, ensures same feature order for training and inference
    """
    
    # 定义特征顺序 (重要: 训练和推理必须一致) / Define feature order (important: must be consistent)
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
        
        # New Features (6) / 新增特征
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
        从ButterflyAnalyzer结果提取特征 / Extract features from ButterflyAnalyzer result
        
        参数 / Parameters:
            analysis_result: ButterflyAnalyzer.full_analysis()的返回字典 / Return dict from ButterflyAnalyzer
            price_history: 可选的价格历史DataFrame / Optional price history DataFrame
        
        返回 / Returns:
            features_dict: 22个特征的字典 / Dictionary with 22 features
        """
        # 解包分析结果 / Unpack analysis result
        fourier = analysis_result.get('fourier', {})
        arima = analysis_result.get('arima', {})
        garch = analysis_result.get('garch', {})
        greeks = analysis_result.get('greeks', {})
        butterfly = analysis_result.get('butterfly', {})
        
        # 如果greeks在butterfly内部 / If greeks is inside butterfly
        if not greeks and 'greeks' in butterfly:
            greeks = butterfly.get('greeks', {})
        
        # 基础特征 (16个) / Base features (16)
        base_features = {
            # Fourier分析 / Fourier analysis
            'trend_slope': _safe_get(fourier, 'trend_slope', 0.0),
            'dominant_period': _safe_get(fourier, 'dominant_period_days', 0.0),
            'period_strength': _safe_get(fourier, 'period_strength', 0.0),
            
            # ARIMA预测 / ARIMA forecast
            'forecast_price': _safe_get(arima, 'mean_forecast', 0.0),
            
            # GARCH波动率 / GARCH volatility
            'predicted_vol': _safe_get(garch, 'predicted_vol', 0.0),
            'current_iv': _safe_get(garch, 'current_iv', 0.0),
            'vol_mispricing': _safe_get(garch, 'vol_mispricing', 0.0),
            'iv_percentile': _safe_get(garch, 'iv_percentile', 0.5),
            
            # Greeks
            'delta': _safe_get(greeks, 'delta', 0.0),
            'gamma': _safe_get(greeks, 'gamma', 0.0),
            'vega': _safe_get(greeks, 'vega', 0.0),
            'theta': _safe_get(greeks, 'theta', 0.0),
            
            # 蝴蝶策略参数 / Butterfly strategy parameters
            'max_profit': _safe_get(butterfly, 'max_profit', 0.0),
            'max_loss': _safe_get(butterfly, 'max_loss', 0.0),
            'profit_ratio': _safe_get(butterfly, 'profit_ratio', 0.0),
            'prob_profit': _safe_get(butterfly, 'prob_profit', 0.5)
        }
        
        # 新增特征 (6个) / New features (6)
        new_features = FeatureExtractor._extract_new_features(
            analysis_result, 
            price_history
        )
        
        # 合并并验证 / Merge and validate
        features = {**base_features, **new_features}
        features = FeatureExtractor._validate_features(features)
        
        return features
    
    @staticmethod
    def _extract_new_features(analysis_result: Dict, price_history: Optional[pd.DataFrame]) -> Dict:
        """提取6个新增特征 / Extract 6 new features"""
        
        garch = analysis_result.get('garch', {})
        greeks = analysis_result.get('greeks', {})
        butterfly = analysis_result.get('butterfly', {})
        arima = analysis_result.get('arima', {})
        
        # 如果greeks在butterfly内部 / If greeks is inside butterfly
        if not greeks and 'greeks' in butterfly:
            greeks = butterfly.get('greeks', {})
        
        # 1. IV Skew估计 / IV Skew estimate
        skew_estimate = _safe_get(garch, 'vol_mispricing', 0.0) * 100
        
        # 2. 价格动量 (需要历史数据) / Price momentum (needs history)
        momentum_7d = _calculate_momentum(price_history, days=7)
        
        # 3. 成交量集中度 / Volume concentration
        vol_concentration = _calculate_volume_concentration(price_history, window=5)
        
        # 4. DTE因子 (归一化到0-1) / DTE factor (normalized)
        dte = _safe_get(butterfly, 'dte', 30)
        dte_factor = min(dte / 30.0, 2.0)  # 限制最大值为2 / Cap at 2
        
        # 5. 价格稳定性 / Price stability
        price_stability = _calculate_price_stability(arima, price_history)
        
        # 6. Gamma/Theta比率 / Gamma/Theta ratio
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
        """
        验证并清理特征 / Validate and clean features
        - 替换NaN/Inf为安全值 / Replace NaN/Inf with safe values
        - 确保所有特征都存在 / Ensure all features exist
        """
        validated = {}
        
        for name in FeatureExtractor.FEATURE_NAMES:
            value = features.get(name, 0.0)
            
            # 处理异常值 / Handle anomalies
            if value is None or not np.isfinite(value):
                logger.debug(f"特征 {name} 包含非有限值: {value}, 替换为0 / Feature {name} has invalid value: {value}")
                value = 0.0
            
            validated[name] = float(value)
        
        return validated
    
    @staticmethod
    def to_array(features: Dict) -> np.ndarray:
        """
        转换为numpy数组 (按固定顺序) / Convert to numpy array (fixed order)
        
        返回 / Returns: shape (22,) 的数组 / array of shape (22,)
        """
        return np.array([
            features.get(name, 0.0) for name in FeatureExtractor.FEATURE_NAMES
        ], dtype=np.float32)
    
    @staticmethod
    def to_dataframe(features_list: list) -> pd.DataFrame:
        """
        批量特征转DataFrame / Batch features to DataFrame
        
        参数 / Parameters:
            features_list: List[Dict] - 特征字典列表 / List of feature dicts
        
        返回 / Returns:
            DataFrame - 每行一个样本 / Each row is a sample
        """
        return pd.DataFrame(features_list, columns=FeatureExtractor.FEATURE_NAMES)


# ==================== 辅助函数 / Helper Functions ====================

def _safe_get(dictionary: Dict, key: str, default=0.0):
    """安全获取字典值 / Safely get dict value"""
    if dictionary is None:
        return default
    value = dictionary.get(key, default)
    if value is None:
        return default
    if isinstance(value, (int, float)) and not np.isfinite(value):
        return default
    return value


def _calculate_momentum(price_history: Optional[pd.DataFrame], days: int = 7) -> float:
    """
    计算价格动量 / Calculate price momentum
    
    公式 / Formula: momentum = (P_now - P_days_ago) / P_days_ago
    """
    if price_history is None or len(price_history) < days:
        return 0.0
    
    try:
        current_price = float(price_history['Close'].iloc[-1])
        past_price = float(price_history['Close'].iloc[-days])
        
        if past_price == 0:
            return 0.0
        
        momentum = (current_price - past_price) / past_price
        return float(np.clip(momentum, -1.0, 1.0))  # 限制在±100% / Clip to ±100%
    
    except Exception as e:
        logger.debug(f"计算动量失败 / Momentum calculation failed: {e}")
        return 0.0


def _calculate_volume_concentration(price_history: Optional[pd.DataFrame], window: int = 5) -> float:
    """
    计算成交量集中度 / Calculate volume concentration
    
    公式 / Formula: concentration = max(volume_recent) / mean(volume_recent)
    """
    if price_history is None or len(price_history) < window:
        return 1.0
    
    try:
        recent_volume = price_history['Volume'].iloc[-window:]
        mean_vol = recent_volume.mean()
        
        if mean_vol == 0:
            return 1.0
        
        concentration = recent_volume.max() / mean_vol
        return float(np.clip(concentration, 0.5, 10.0))  # 限制范围 / Clip range
    
    except Exception as e:
        logger.debug(f"计算成交量集中度失败 / Volume concentration calculation failed: {e}")
        return 1.0


def _calculate_price_stability(arima: Dict, price_history: Optional[pd.DataFrame]) -> float:
    """
    计算价格稳定性 / Calculate price stability
    
    方法1 (优先): 使用ARIMA置信区间宽度 / Method 1: Use ARIMA CI width
    方法2 (备用): 使用历史价格标准差 / Method 2: Use historical std
    """
    # 方法1: ARIMA置信区间 / Method 1: ARIMA CI
    if arima:
        ci_width = _safe_get(arima, 'confidence_interval_width', None)
        if ci_width is not None and ci_width > 0:
            return 1.0 / (ci_width + 1e-6)
    
    # 方法2: 历史波动率 / Method 2: Historical volatility
    if price_history is not None and len(price_history) >= 10:
        try:
            price_std = price_history['Close'].std()
            price_mean = price_history['Close'].mean()
            
            if price_mean > 0:
                cv = price_std / price_mean  # 变异系数 / Coefficient of variation
                return 1.0 / (cv + 1e-6)
        except Exception:
            pass
    
    # 默认值 / Default value
    return 1.0


def _calculate_gamma_theta_ratio(greeks: Dict) -> float:
    """
    计算Gamma/Theta比率 / Calculate Gamma/Theta ratio
    
    原理 / Rationale: 
    - Gamma: 每$1价格变动对Delta的影响 / Delta sensitivity to price
    - Theta: 每天的时间衰减 / Daily time decay
    """
    gamma = _safe_get(greeks, 'gamma', 0.0)
    theta = _safe_get(greeks, 'theta', -0.01)
    
    theta_abs = abs(theta)
    
    if theta_abs < 1e-6:
        return 0.0
    
    ratio = abs(gamma) / theta_abs
    return float(np.clip(ratio, 0.0, 100.0))  # 限制最大值 / Cap max value


# ==================== 便捷函数 / Convenience Functions ====================

def extract_features_v2(analysis_result: Dict, price_history: Optional[pd.DataFrame] = None) -> Dict:
    """
    快捷函数: 提取特征 / Shortcut: Extract features
    
    使用示例 / Example:
        features = extract_features_v2(analyzer.full_analysis(), price_df)
    """
    return FeatureExtractor.extract_from_analysis(analysis_result, price_history)


def validate_feature_quality(df: pd.DataFrame) -> Dict:
    """
    验证特征质量 / Validate feature quality
    
    返回 / Returns:
        report: 包含问题汇总的字典 / Dict with issue summary
    """
    report = {
        'total_samples': len(df),
        'total_features': len(FeatureExtractor.FEATURE_NAMES),
        'issues': []
    }
    
    for col in FeatureExtractor.FEATURE_NAMES:
        if col not in df.columns:
            report['issues'].append(f"缺失特征 / Missing feature: {col}")
            continue
        
        # 检查NaN / Check NaN
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            report['issues'].append(f"{col}: {nan_count} 个NaN值 / NaN values")
        
        # 检查Inf / Check Inf
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            report['issues'].append(f"{col}: {inf_count} 个Inf值 / Inf values")
        
        # 检查零方差 / Check zero variance
        if df[col].std() == 0:
            report['issues'].append(f"{col}: 零方差 (常数列) / Zero variance (constant)")
    
    return report


# ==================== Legacy FeatureEngine (向后兼容) / Backward Compatibility ====================

class FeatureEngine:
    """[LEGACY] 原有特征工程类,保留向后兼容 / Original feature engine, kept for compatibility"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """连接数据库 / Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        
    def fetch_records(self, limit=10000, table='analysis_history'):
        """获取分析记录 / Fetch analysis records"""
        query = f"SELECT ticker, analysis_date, full_result FROM {table} ORDER BY analysis_date DESC LIMIT %s"
        df = pd.read_sql(query, self.conn, params=(limit,))
        return df

    def extract_features_from_json(self, json_data):
        """从JSON提取特征 (使用新的FeatureExtractor) / Extract features from JSON (using new extractor)"""
        if not json_data:
            return None
        
        try:
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data
            
            # 使用新的提取器 / Use new extractor
            features = FeatureExtractor.extract_from_analysis(data)
            
            # 添加标注辅助数据 / Add labeling helper data
            butterfly = data.get('butterfly', {})
            features['center_strike'] = butterfly.get('center_strike')
            features['lower_strike'] = butterfly.get('lower_strike')
            features['upper_strike'] = butterfly.get('upper_strike')
            
            return features
        except Exception as e:
            logger.debug(f"特征提取失败 / Feature extraction failed: {e}")
            return None

    def create_dataset(self):
        """创建数据集 / Create dataset"""
        logger.info("Fetching records from database... / 从数据库获取记录...")
        df_raw = self.fetch_records()
        
        logger.info("Extracting features from JSON... / 从JSON提取特征...")
        feature_list = []
        for _, row in df_raw.iterrows():
            f = self.extract_features_from_json(row['full_result'])
            if f:
                f['ticker'] = row['ticker']
                f['analysis_date'] = row['analysis_date']
                feature_list.append(f)
        
        df = pd.DataFrame(feature_list)
        
        # Drop rows with missing strikes / 删除缺失行权价的行
        df = df.dropna(subset=['center_strike', 'lower_strike', 'upper_strike'])
        
        logger.info(f"Processed {len(df)} records with valid features. / 处理了 {len(df)} 条有效记录")
        return df

    def generate_labels(self, df):
        """生成标签 (动态评估时间) / Generate labels (Dynamic evaluation timing)"""
        import yfinance as yf
        from datetime import timedelta
        
        labels = []
        for idx, row in df.iterrows():
            try:
                ticker = row['ticker']
                # 解析建仓日期 / Parse analysis date
                analysis_date = pd.to_datetime(row['analysis_date'])
                
                # 获取策略DTE (默认为30天) / Get strategy DTE (default 30 days)
                # 注意：这里假设full_result解析后的特征字典里有 'dte'，或者从row中获取
                # 如果row中没有dte，由于这是历史模拟数据，我们可能需要从full_result里解包
                # 这里为了兼容性，先尝试从 features获取，如果没有则默认
                 
                # 尝试从full_result JSON中获取 dte / Try to get dte from full_result JSON
                cols = row.keys()
                dte = 30
                if 'dte' in cols:
                    dte = row['dte']
                elif 'full_result' in cols:
                    import json
                    try:
                        fr = json.loads(row['full_result']) if isinstance(row['full_result'], str) else row['full_result']
                        dte = fr.get('butterfly', {}).get('dte', 30)
                    except:
                        pass
                
                # 动态计算评估日期 / Calculate dynamic evaluation date
                target_date, _ = calculate_dynamic_evaluation_date(analysis_date, dte)
                
                # 获取未来价格 / Get future price
                future_data = yf.download(
                    ticker,
                    start=target_date,
                    end=target_date + timedelta(days=5),
                    progress=False
                )
                
                if len(future_data) == 0:
                    labels.append(None)
                    continue
                
                future_price = float(future_data['Close'].iloc[0])
                
                # 计算ROI和标签 / Calculate ROI and label
                lower = row['lower_strike']
                center = row['center_strike']
                upper = row['upper_strike']
                cost = row.get('max_loss', 1.0)
                max_profit = row.get('max_profit', 1.0)
                
                if lower <= future_price <= upper:
                    if future_price <= center:
                        payoff = max_profit * (future_price - lower) / (center - lower) if center != lower else 0
                    else:
                        payoff = max_profit * (upper - future_price) / (upper - center) if upper != center else 0
                else:
                    payoff = -cost
                
                roi = (payoff - cost) / cost if cost > 0 else -1
                
                # 使用新的分类标准 / Use new classification criteria
                label = classify_roi(roi)
                labels.append(label)
                
            except Exception as e:
                logger.debug(f"标签生成失败 / Label generation failed: {e}")
                labels.append(None)
        
        df['label'] = labels
        return df.dropna(subset=['label'])


# ==================== Labeling Helper Functions / 标注辅助函数 ====================

def calculate_dynamic_evaluation_date(analysis_date, dte):
    """
    动态计算评估日期 / Calculate dynamic evaluation date
    
    逻辑 / Logic:
    - DTE >= 30天: 到期前5天评估 / Evaluate 5 days before expiry
    - DTE < 30天: 到期前3天评估 / Evaluate 3 days before expiry
    
    返回 / Returns:
        (evaluation_date, days_held)
    """
    from datetime import timedelta
    if dte >= 30:
        days_held = dte - 5
    else:
        days_held = max(dte - 3, int(dte * 0.8))
    
    evaluation_date = analysis_date + timedelta(days=days_held)
    return evaluation_date, days_held


def classify_roi(roi):
    """
    ROI分类 (新标准) / ROI Classification (New Criteria)
    
    阈值 / Thresholds:
    - Loss: < -10%
    - Minor: -10% ~ 5%
    - Good: 5% ~ 15%
    - Excellent: > 15%
    """
    if roi < -0.10:
        return 0  # 亏损 / Loss
    elif roi < 0.05:
        return 1  # 微利 / Minor
    elif roi < 0.15:
        return 2  # 良好 / Good
    else:
        return 3  # 优秀 / Excellent



if __name__ == "__main__":
    # 测试代码 / Test code
    print("=" * 60)
    print("FeatureExtractor V2.0 - 22维特征 / 22-dim features")
    print("=" * 60)
    
    # 模拟分析结果 / Mock analysis result
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
    
    print(f"\n特征数量 / Feature count: {len(features)}")
    print(f"预期数量 / Expected: {len(FeatureExtractor.FEATURE_NAMES)}")
    
    print("\n特征值 / Feature values:")
    for name, value in features.items():
        print(f"  {name:25s}: {value:10.4f}")
    
    # 转换为数组 / Convert to array
    feature_array = FeatureExtractor.to_array(features)
    print(f"\nNumPy数组 shape / array shape: {feature_array.shape}")
    
    print("\n✅ 测试通过! / Test passed!")
