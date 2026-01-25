# -*- coding: utf-8 -*-
"""
ButterQuant ML Training Data Generator / ButterQuant ML 训练数据生成器
历史回测模拟生成训练数据 - Phase 1 / Historical simulation for training data - Phase 1

功能 / Features:
1. VIX分层采样 (低波/常态/高波) / VIX-based stratified sampling (low/normal/high volatility)
2. 历史蝴蝶策略模拟 (as-of分析) / Historical butterfly strategy simulation (as-of analysis)
3. 简化IV Proxy计算 / Simplified IV Proxy calculation
4. 14天前向标注 (4分类) / 14-day forward labeling (4-class)

预计耗时 / Estimated time: 2-3小时 / 2-3 hours
输出 / Output: ml/training_data_deep.parquet
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 添加项目路径 / Add project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))
import random
from typing import Dict, List, Tuple, Optional
from ml.features import calculate_dynamic_evaluation_date, classify_roi
import warnings
warnings.filterwarnings('ignore')

# 配置日志 / Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)  # Suppress yfinance warnings
logging.getLogger('peewee').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


class HistoricalDataGenerator:
    """历史回测数据生成器 / Historical simulation data generator"""
    
    def __init__(self, output_dir: str = None):
        # 输出目录 / Output directory
        if output_dir is None:
            self.output_dir = Path(__file__).parent
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 采样配置 / Sampling configuration
        self.sample_config = {
            'LOW_VOL': 2000,    # Target sample count
            'NORMAL': 3000,
            'HIGH_VOL': 2000
        }
        
        # VIX阈值定义 / VIX threshold definitions
        self.vix_thresholds = {
            'LOW_VOL': (0, 15),
            'NORMAL': (15, 25),
            'HIGH_VOL': (25, 100)
        }
        
        # 时间范围 / Time range
        self.start_date = "2023-01-01"
        self.end_date = "2025-01-31"
        
        # 高流动性股票池 (按行业分散) / High liquidity stock pool (diversified by sector)
        self.ticker_pool = [
            # 高流动性ETF (重要: 提供市场Beta特征) / High liquidity ETFs (Important: Provide market Beta)
            'SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'SLV', 'EEM', 'XLE', 'XLF', 'XLK', 'XLV',
            # 科技 & 半导体 / Tech & Semi
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
            'ADBE', 'ORCL', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'IBM', 'NOW', 'UBER', 'ABNB',
            'PLTR', 'SNOW', 'PANW', 'FTNT',
            # 金融 & 支付 / Finance & Payments
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA',
            'PYPL', 'SQ', 'COIN', 'HOOD',
            # 医疗 & 制药 / Healthcare & Pharma
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'LLY', 'BMY', 'AMGN', 'GILD',
            'CVS', 'CI', 'ISRG',
            # 消费 & 零售 / Consumer & Retail
            'WMT', 'HD', 'NKE', 'MCD', 'COST', 'TGT', 'SBUX', 'LOW', 'PG', 'KO', 'PEP',
            'CL', 'EL', 'LULU', 'CMG',
            # 能源 & 原材料 / Energy & Materials
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'LIN', 'FCX', 'NEM',
            # 工业 & 国防 / Industrial & Defense
            'BA', 'CAT', 'GE', 'UPS', 'HON', 'RTX', 'LMT', 'DE', 'UNP', 'LUV', 'DAL',
            # 通讯 & 媒体 / Telecom & Media
            'DIS', 'NFLX', 'CMCSA', 'TMUS', 'VZ', 'T',
            # 房地产 & 公用事业 / Real Estate & Utilities
            'PLD', 'AMT', 'CCI', 'O', 'NEE', 'DUK', 'SO'
        ]
        
    def download_vix_data(self) -> pd.DataFrame:
        """下载VIX历史数据 / Download VIX historical data"""
        logger.info("📥 下载VIX历史数据... / Downloading VIX historical data...")
        vix = yf.download("^VIX", start=self.start_date, end=self.end_date, progress=False)
        logger.info(f"✅ 获取 {len(vix)} 天VIX数据 / Retrieved {len(vix)} days of VIX data")
        return vix
    
    def stratified_sampling(self, vix_data: pd.DataFrame) -> Dict[str, List[datetime]]:
        """基于VIX的分层采样 / VIX-based stratified sampling"""
        logger.info("🎲 执行分层采样... / Executing stratified sampling...")
        
        samples = {}
        for regime, (low, high) in self.vix_thresholds.items():
            # 筛选符合条件的日期 / Filter eligible dates
            mask = (vix_data['Close'] >= low) & (vix_data['Close'] < high)
            eligible_dates = vix_data[mask].index.tolist()
            
            # 计算需要采样的天数 (每天50个标的) / Calculate days to sample (50 tickers per day)
            n_days = self.sample_config[regime] // 50
            
            if len(eligible_dates) < n_days:
                logger.warning(f"⚠️ {regime} 可用日期不足 / Insufficient eligible dates: {len(eligible_dates)} < {n_days}")
                sampled = eligible_dates
            else:
                sampled = random.sample(eligible_dates, k=n_days)
            
            samples[regime] = sorted(sampled)
            logger.info(f"  {regime}: 采样 {len(sampled)} 天 (VIX {low}-{high}) / Sampled {len(sampled)} days")
        
        return samples
    
    def get_top_tickers(self, date: datetime, n: int = 50) -> List[str]:
        """
        获取指定日期的Top N标的 / Get Top N tickers for specified date
        
        简化版: 使用固定的流动性好的股票池 / Simplified: Use fixed high-liquidity stock pool
        生产版: 可以从历史市值/成交量数据筛选 / Production: Filter by historical market cap/volume
        """
        # 随机抽取n个 (模拟不同日期的热门股) / Random sample n (simulate popular stocks on different dates)
        return random.sample(self.ticker_pool, min(n, len(self.ticker_pool)))
    
    def get_iv_proxy(self, strike: float, spot: float, hv: float) -> float:
        """
        简化版IV代理 / Simplified IV Proxy
        
        基于Moneyness调整历史波动率 / Adjust historical volatility based on Moneyness:
        - OTM Put (K < 0.95S): 1.25x HV (恐慌溢价 / Panic premium)
        - ATM (0.95S ≤ K ≤ 1.05S): 1.15x HV
        - OTM Call (K > 1.05S): 1.10x HV
        """
        moneyness = strike / spot
        
        if moneyness < 0.95:
            multiplier = 1.25
        elif moneyness > 1.05:
            multiplier = 1.10
        else:
            multiplier = 1.15
        
        return hv * multiplier
    
    def simulate_butterfly_analysis(self, ticker: str, as_of_date: datetime) -> Optional[Dict]:
        """
        模拟蝴蝶策略分析 (as-of时刻) / Simulate butterfly strategy analysis (as-of moment)
        
        注意: 这里简化了分析逻辑,实际可集成ButterflyAnalyzer / Note: Simplified analysis logic
        """
        try:
            # 1. 获取截至该日期的历史价格 / Get historical prices up to that date
            hist = yf.download(
                ticker, 
                end=as_of_date, 
                period="90d",
                progress=False,
                threads=False,  # Reduce rate limit/errors
                ignore_tz=True  # Fix timezone issues
            )
            
            if len(hist) < 30:
                return None
            
            spot = float(hist['Close'].iloc[-1])
            returns = hist['Close'].pct_change().dropna()
            hv = float(returns.std() * np.sqrt(252))
            
            # 2. 构造蝴蝶策略参数 / Construct butterfly strategy parameters
            dte = 30  # 假设30天到期 / Assume 30 days to expiry
            
            # 确定行权价间隔 / Determine strike interval
            if spot < 50:
                strike_step = 2.5
            elif spot < 100:
                strike_step = 5
            elif spot < 200:
                strike_step = 5
            else:
                strike_step = 10
            
            center_strike = round(spot / strike_step) * strike_step
            wing_width = strike_step * 2  # 2个间隔的翼宽 / 2 intervals wing width
            
            lower_strike = center_strike - wing_width
            upper_strike = center_strike + wing_width
            
            # 3. 使用IV Proxy计算期权价格 / Calculate option prices using IV Proxy
            iv_lower = self.get_iv_proxy(lower_strike, spot, hv)
            iv_center = self.get_iv_proxy(center_strike, spot, hv)
            iv_upper = self.get_iv_proxy(upper_strike, spot, hv)
            
            # 简化BS定价 / Simplified BS pricing
            T = dte / 365
            r = 0.045  # 无风险利率 / Risk-free rate
            
            def simple_call_price(S, K, iv, T):
                """简化的Call期权价格 / Simplified Call option price"""
                from scipy.stats import norm
                if T <= 0 or iv <= 0:
                    return max(S - K, 0)
                d1 = (np.log(S/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
                d2 = d1 - iv*np.sqrt(T)
                return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            
            price_lower = simple_call_price(spot, lower_strike, iv_lower, T)
            price_center = simple_call_price(spot, center_strike, iv_center, T)
            price_upper = simple_call_price(spot, upper_strike, iv_upper, T)
            
            # 蝴蝶组合成本 / Butterfly spread cost
            net_debit = price_lower - 2*price_center + price_upper
            net_debit = max(0.10, net_debit)  # 最低成本 / Minimum cost
            max_profit = wing_width - net_debit
            
            # 4. 计算特征 / Calculate features
            returns_arr = returns.values
            forecast_price = spot * (1 + float(returns.mean()) * dte)
            predicted_vol = hv * 0.9  # GARCH预测通常略低 / GARCH prediction usually slightly lower
            
            # 傅里叶相关特征 (简化) / Fourier-related features (simplified)
            if len(returns_arr) >= 20:
                trend_slope = float((hist['Close'].iloc[-1] - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20] * 100)
            else:
                trend_slope = 0.0
            
            # Greeks (简化计算) / Greeks (simplified calculation)
            delta = np.random.normal(0.0, 0.01)  # 加上微小扰动 / Add small noise
            gamma = 0.05 / spot
            vega = wing_width * 0.01
            theta = -net_debit / dte
            
            # 动量和成交量特征 / Momentum and volume features
            if len(hist) >= 7:
                momentum_7d = float((hist['Close'].iloc[-1] - hist['Close'].iloc[-7]) / hist['Close'].iloc[-7])
                vol_recent = hist['Volume'].iloc[-5:]
                vol_concentration = float(vol_recent.max() / (vol_recent.mean() + 1e-6))
            else:
                momentum_7d = 0.0
                vol_concentration = 1.0
            
            return {
                'ticker': ticker,
                'analysis_date': as_of_date,
                'spot_price': spot,
                'hv': hv,
                
                'butterfly': {
                    'lower_strike': lower_strike,
                    'center_strike': center_strike,
                    'upper_strike': upper_strike,
                    'net_debit': net_debit,
                    'max_profit': max_profit,
                    'max_loss': net_debit,
                    'profit_ratio': max_profit / net_debit if net_debit > 0 else 0,
                    'prob_profit': 0.5,  # 简化 / Simplified
                    'dte': dte
                },
                
                'fourier': {
                    'trend_slope': trend_slope,
                    'dominant_period_days': 21,
                    'period_strength': 0.3
                },
                
                'arima': {
                    'mean_forecast': forecast_price,
                    'confidence_interval_width': spot * 0.1
                },
                
                'garch': {
                    'predicted_vol': predicted_vol,
                    'current_iv': iv_center,
                    'vol_mispricing': (iv_center - predicted_vol) / predicted_vol if predicted_vol > 0 else 0,
                    'iv_percentile': 50.0 + np.random.normal(0, 5)  # 模拟IV分位数的随机性 / Simulate IV percentile randomness
                },
                
                'greeks': {
                    'delta': delta,
                    'gamma': gamma,
                    'vega': vega,
                    'theta': theta
                },
                
                # 额外特征 / Extra features
                'momentum_7d': momentum_7d,
                'vol_concentration': vol_concentration
            }
            
        except Exception as e:
            logger.debug(f"⚠️ {ticker} @ {as_of_date}: {e}")
            return None
    
    def calculate_label(self, analysis: Dict, future_price: float) -> int:
        """
        计算4分类标签 / Calculate 4-class label
        
        基于14天后的实际价格计算ROI / Calculate ROI based on actual price after 14 days:
        - 0: 亏损 (ROI < 0) / Loss
        - 1: 微利 (0% ≤ ROI < 10%) / Minor profit
        - 2: 良好 (10% ≤ ROI < 30%) / Good
        - 3: 优秀 (ROI ≥ 30%) / Excellent
        """
        bf = analysis['butterfly']
        lower = bf['lower_strike']
        center = bf['center_strike']
        upper = bf['upper_strike']
        cost = bf['net_debit']
        max_profit = bf['max_profit']
        
        # 计算payoff / Calculate payoff
        if lower <= future_price <= upper:
            if future_price <= center:
                # 左翼 / Left wing
                payoff = max_profit * (future_price - lower) / (center - lower) if center != lower else 0
            else:
                # 右翼 / Right wing
                payoff = max_profit * (upper - future_price) / (upper - center) if upper != center else 0
        else:
            # 超出区间,损失全部成本 / Outside range, lose all cost
            payoff = -cost
        
        roi = (payoff - cost) / cost if cost > 0 else -1
        
        # 使用统一的新分类标准 / Use unified new classification criteria
        label = classify_roi(roi)
        return label, roi
    
    def extract_features(self, analysis: Dict) -> Dict:
        """
        提取23维特征向量 / Extract 23-dim feature vector
        
        包括 / Includes:
        - 原有16个特征 (移除total_score) / Original 16 features (removed total_score)
        - 新增6个低成本特征 / 6 new low-cost features
        """
        bf = analysis['butterfly']
        fourier = analysis['fourier']
        arima = analysis['arima']
        garch = analysis['garch']
        greeks = analysis['greeks']
        
        return {
            # 原有特征 (16个) / Original features (16)
            'trend_slope': fourier.get('trend_slope', 0),
            'dominant_period': fourier.get('dominant_period_days', 0),
            'period_strength': fourier.get('period_strength', 0),
            'forecast_price': arima.get('mean_forecast', 0),
            'predicted_vol': garch.get('predicted_vol', 0),
            'current_iv': garch.get('current_iv', 0),
            'vol_mispricing': garch.get('vol_mispricing', 0),
            'iv_percentile': garch.get('iv_percentile', 0),
            'delta': greeks.get('delta', 0),
            'gamma': greeks.get('gamma', 0),
            'vega': greeks.get('vega', 0),
            'theta': greeks.get('theta', 0),
            'max_profit': bf.get('max_profit', 0),
            'max_loss': bf.get('max_loss', 0),
            'profit_ratio': bf.get('profit_ratio', 0),
            'prob_profit': bf.get('prob_profit', 0.5),
            
            # 新增特征 (6个) / New features (6)
            'skew_estimate': garch.get('vol_mispricing', 0) * 100,
            'momentum_7d': analysis.get('momentum_7d', 0),
            'vol_concentration': analysis.get('vol_concentration', 1.0),
            'dte_factor': bf.get('dte', 30) / 30.0,
            'price_stability': 1.0 / (arima.get('confidence_interval_width', 1.0) + 1e-6),
            'gamma_theta_ratio': abs(greeks.get('gamma', 0) / (greeks.get('theta', -0.01) + 1e-6))
        }
    
    def generate_dataset(self, limit: int = None) -> pd.DataFrame:
        """主数据生成流程 / Main data generation process"""
        
        # Step 1: VIX分层采样 / VIX stratified sampling
        vix_data = self.download_vix_data()
        sampled_dates = self.stratified_sampling(vix_data)
        
        # Step 2: 遍历所有采样日期 / Iterate through all sampled dates
        all_samples = []
        total_dates = sum(len(dates) for dates in sampled_dates.values())
        
        logger.info(f"🚀 开始生成数据: 共 {total_dates} 天 / Starting data generation: {total_dates} days total")
        
        processed = 0
        total_collected = 0
        
        for regime, dates in sampled_dates.items():
            if limit and total_collected >= limit:
                break
                
            logger.info(f"\n📊 处理 {regime} 市场 ({len(dates)} 天) / Processing {regime} market ({len(dates)} days)")
            
            for date in dates:
                if limit and total_collected >= limit:
                    break
                    
                processed += 1
                if processed % 5 == 0:
                    logger.info(f"  [{processed}/{total_dates}] {date.date() if hasattr(date, 'date') else date}")
                
                # 获取Top 50标的 / Get Top 50 tickers
                tickers = self.get_top_tickers(date)
                
                for ticker in tickers:
                    if limit and total_collected >= limit:
                        break
                        
                    try:
                        # 历史模拟分析 / Historical simulation analysis
                        analysis = self.simulate_butterfly_analysis(ticker, date)
                        if analysis is None:
                            continue
                        
                        # 获取蝴蝶参数 / Get butterfly params
                        bf = analysis['butterfly']
                        dte = bf['dte']
                        
                        # 动态计算评估日期 / Calculate dynamic evaluation date
                        future_date, _ = calculate_dynamic_evaluation_date(date, dte)
                        
                        # 获取评估日期的实际价格 / Get actual price at evaluation date
                        future_data = yf.download(
                            ticker,
                            start=future_date,
                            end=future_date + timedelta(days=5),
                            progress=False,
                            threads=False,
                            ignore_tz=True
                        )
                        
                        if len(future_data) == 0:
                            continue
                        
                        # Fix for multi-level columns
                        close_data = future_data['Close']
                        if isinstance(close_data, pd.DataFrame):
                            future_price = float(close_data.iloc[0, 0])
                        else:
                            future_price = float(close_data.iloc[0])
                        
                        # 计算标签 / Calculate label (using new classify_roi)
                        label, roi_val = self.calculate_label(analysis, future_price)
                        
                        # 提取特征 / Extract features
                        features = self.extract_features(analysis)
                        
                        # 合并 / Merge
                        sample = {
                            **features,
                            'label': label,
                            '_ticker': ticker,
                            '_date': str(date.date() if hasattr(date, 'date') else date),
                            '_regime': regime,
                            '_spot': analysis['spot_price'],
                            '_future_price': future_price,
                            '_debug_roi': roi_val,
                            # Save strikes for future safety
                            'lower_strike': bf['lower_strike'],
                            'center_strike': bf['center_strike'],
                            'upper_strike': bf['upper_strike'],
                            'net_debit': bf['net_debit'],
                            'dte': dte
                        }
                        
                        all_samples.append(sample)
                        total_collected += 1
                        
                    except Exception as e:
                        logger.debug(f"    ⚠️ {ticker}: {e}")
                        continue
                
                if processed % 10 == 0:
                    logger.info(f"  ✅ 已收集 {len(all_samples)} 个样本 / Collected {len(all_samples)} samples")
        
        # Step 3: 转换为DataFrame / Convert to DataFrame
        df = pd.DataFrame(all_samples)
        logger.info(f"\n✅ 数据生成完成: {len(df)} 个样本 / Data generation complete: {len(df)} samples")
        
        return df
    
    def validate_dataset(self, df: pd.DataFrame):
        """数据质量验证 / Data quality validation"""
        logger.info("\n🔍 数据质量检查 / Data Quality Check:")
        
        # 1. 样本量 / Sample count
        logger.info(f"  总样本数 / Total samples: {len(df)}")
        
        # 2. 标签分布 / Label distribution
        label_dist = df['label'].value_counts(normalize=True).sort_index()
        logger.info(f"  标签分布 / Label distribution:")
        label_names = ['Loss/亏损', 'Minor/微利', 'Good/良好', 'Excellent/优秀']
        for label, pct in label_dist.items():
            logger.info(f"    Class {label} ({label_names[label]}): {pct:.1%}")
        
        # 3. 缺失值 / Missing values
        missing = df.isnull().sum().sum()
        logger.info(f"  缺失值 / Missing values: {missing}")
        
        # 4. 特征范围 / Feature range
        feature_cols = [col for col in df.columns if not col.startswith('_') and col != 'label']
        logger.info(f"  特征数量 / Feature count: {len(feature_cols)}")
        
        # 检查异常值 / Check for anomalies
        for col in feature_cols:
            if np.isinf(df[col]).any():
                logger.warning(f"  ⚠️ {col} 包含无穷值 / contains infinite values")
            if df[col].std() == 0:
                logger.warning(f"  ⚠️ {col} 无方差 / has no variance")
    
    def run(self, limit: int = None) -> pd.DataFrame:
        """执行完整流程 / Execute complete pipeline"""
        logger.info("=" * 60)
        logger.info("🦋 ButterQuant ML 训练数据生成器 / Training Data Generator")
        logger.info("=" * 60)
        
        # 生成数据 / Generate data
        df = self.generate_dataset(limit=limit)
        
        if len(df) == 0:
            logger.error("❌ 未生成任何数据! / No data generated!")
            return df
        
        # 验证 / Validate
        self.validate_dataset(df)
        
        # 保存 / Save
        try:
            output_path = self.output_dir / "training_data_deep.parquet"
            if limit:
                output_path = self.output_dir / "training_data_deep_test.parquet"
            
            df.to_parquet(output_path, index=False)
            logger.info(f"\n💾 数据已保存(Parquet) / Data saved: {output_path}")
        except ImportError:
            logger.warning("⚠️ 缺少pyarrow/fastparquet，回退到CSV / Missing parquet lib, fallback to CSV")
            output_path = self.output_dir / "training_data_deep.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"\n💾 数据已保存(CSV) / Data saved: {output_path}")
        except Exception as e:
            logger.error(f"❌ 保存失败 / Save failed: {e}")
            # 尝试强制保存CSV / Try force CSV
            output_path = self.output_dir / "training_data_deep_backup.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"💾 已强制保存备份 / Backup saved: {output_path}")
        
        # 同时保存CSV预览 / Also save CSV preview
        csv_path = self.output_dir / "training_data_deep.csv"
        df.head(100).to_csv(csv_path, index=False)
        logger.info(f"💾 样本预览已保存 / Sample preview saved: {csv_path}")
        
        # 样本预览 / Sample preview
        logger.info("\n📋 样本预览 / Sample Preview:")
        print(df.head())
        
        return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ButterQuant ML Data Generator")
    parser.add_argument('--limit', type=int, default=None, help='Limit total samples for testing')
    args = parser.parse_args()
    
    generator = HistoricalDataGenerator()
    generator.run(limit=args.limit)


if __name__ == "__main__":
    main()
