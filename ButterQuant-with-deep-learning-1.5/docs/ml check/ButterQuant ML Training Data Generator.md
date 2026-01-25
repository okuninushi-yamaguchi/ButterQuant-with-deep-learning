"""
ButterQuant ML Training Data Generator
历史回测模拟生成训练数据

功能:
1. VIX分层采样 (低波/常态/高波)
2. 历史蝴蝶策略模拟 (as-of分析)
3. 简化IV Proxy计算
4. 动态评估时间点 (DTE-5天)
5. 4分类标注

预计耗时: 2-3小时
输出: ml/training_data_deep.parquet
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataGenerator:
    """历史回测数据生成器"""
    
    def __init__(self, output_dir: str = "ml"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 采样配置
        self.sample_config = {
            'LOW_VOL': 2000,
            'NORMAL': 3000,
            'HIGH_VOL': 2000
        }
        
        # VIX阈值
        self.vix_thresholds = {
            'LOW_VOL': (0, 15),
            'NORMAL': (15, 25),
            'HIGH_VOL': (25, 100)
        }
        
        # 时间范围
        self.start_date = "2023-01-01"
        self.end_date = "2025-01-31"
        
    def download_vix_data(self) -> pd.DataFrame:
        """下载VIX历史数据"""
        logger.info("📥 下载VIX历史数据...")
        vix = yf.download("^VIX", start=self.start_date, end=self.end_date, progress=False)
        logger.info(f"✅ 获取 {len(vix)} 天VIX数据")
        return vix
    
    def stratified_sampling(self, vix_data: pd.DataFrame) -> dict:
        """基于VIX的分层采样"""
        logger.info("🎲 执行分层采样...")
        
        samples = {}
        for regime, (low, high) in self.vix_thresholds.items():
            mask = (vix_data['Close'] >= low) & (vix_data['Close'] < high)
            eligible_dates = vix_data[mask].index.tolist()
            
            n_days = self.sample_config[regime] // 50
            
            if len(eligible_dates) < n_days:
                logger.warning(f"⚠️ {regime} 可用日期不足: {len(eligible_dates)} < {n_days}")
                sampled = eligible_dates
            else:
                sampled = random.sample(eligible_dates, k=n_days)
            
            samples[regime] = sorted(sampled)
            logger.info(f"  {regime}: 采样 {len(sampled)} 天 (VIX {low}-{high})")
        
        return samples
    
    def get_top_tickers(self, date: datetime, n: int = 50) -> list:
        """获取Top N标的"""
        ticker_pool = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW',
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'LLY',
            'WMT', 'HD', 'NKE', 'MCD', 'COST', 'TGT', 'SBUX',
            'XOM', 'CVX', 'COP', 'SLB',
            'BA', 'CAT', 'GE', 'UPS', 'HON',
            'DIS', 'NFLX', 'V', 'MA', 'PYPL', 'SQ', 'UBER'
        ]
        return random.sample(ticker_pool, min(n, len(ticker_pool)))
    
    def get_iv_proxy(self, strike: float, spot: float, hv: float) -> float:
        """简化版IV代理"""
        moneyness = strike / spot
        
        if moneyness < 0.95:
            multiplier = 1.25  # OTM Put
        elif moneyness > 1.05:
            multiplier = 1.10  # OTM Call
        else:
            multiplier = 1.15  # ATM
        
        return hv * multiplier
    
    def calculate_evaluation_date(self, analysis_date: datetime, dte: int) -> tuple:
        """动态计算评估日期"""
        if dte >= 30:
            days_to_eval = dte - 5
        else:
            days_to_eval = max(dte - 3, int(dte * 0.8))
        
        evaluation_date = analysis_date + timedelta(days=days_to_eval)
        return evaluation_date, days_to_eval
    
    def calculate_butterfly_roi(self, future_price, lower, center, upper, cost, max_profit):
        """计算蝴蝶策略ROI"""
        if lower <= future_price <= upper:
            if future_price <= center:
                payoff = max_profit * (future_price - lower) / (center - lower)
            else:
                payoff = max_profit * (upper - future_price) / (upper - center)
        else:
            payoff = -cost
        
        roi = payoff / cost if cost > 0 else -1.0
        return roi
    
    def classify_roi(self, roi: float) -> int:
        """ROI分类 (4分类)"""
        if roi < -0.10:
            return 0  # 亏损
        elif roi < 0.05:
            return 1  # 微利
        elif roi < 0.15:
            return 2  # 良好
        else:
            return 3  # 优秀
    
    def simulate_butterfly_analysis(self, ticker: str, as_of_date: datetime) -> dict:
        """模拟蝴蝶策略分析"""
        try:
            # 获取历史数据
            hist = yf.download(ticker, end=as_of_date, period="90d", progress=False)
            
            if len(hist) < 30:
                return None
            
            spot = hist['Close'].iloc[-1]
            hv = hist['Close'].pct_change().std() * np.sqrt(252)
            
            # 构造策略
            dte = 30
            center_strike = round(spot, 0)
            wing_width = max(spot * 0.05, 5)
            
            lower_strike = center_strike - wing_width
            upper_strike = center_strike + wing_width
            
            # IV Proxy
            iv_lower = self.get_iv_proxy(lower_strike, spot, hv)
            iv_center = self.get_iv_proxy(center_strike, spot, hv)
            iv_upper = self.get_iv_proxy(upper_strike, spot, hv)
            
            # 简化定价
            def simple_option_price(strike, iv):
                intrinsic = max(strike - spot, 0)
                time_value = iv * np.sqrt(dte/365) * spot * 0.1
                return intrinsic + time_value
            
            price_lower = simple_option_price(lower_strike, iv_lower)
            price_center = simple_option_price(center_strike, iv_center)
            price_upper = simple_option_price(upper_strike, iv_upper)
            
            net_debit = price_lower - 2*price_center + price_upper
            max_profit = wing_width - net_debit
            
            # 模拟分析结果
            returns = hist['Close'].pct_change().dropna()
            forecast_price = spot * (1 + returns.mean() * dte)
            predicted_vol = returns.std() * np.sqrt(252)
            
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
                    'dte': dte
                },
                'fourier': {
                    'trend_slope': returns.mean() * 252,
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
                    'vol_mispricing': (iv_center - predicted_vol) / predicted_vol,
                    'iv_percentile': 0.5
                },
                'greeks': {
                    'delta': 0.0,
                    'gamma': 0.05,
                    'vega': wing_width * 0.1,
                    'theta': -net_debit / dte
                }
            }
        except Exception as e:
            logger.warning(f"⚠️ {ticker} @ {as_of_date}: {e}")
            return None
    
    def generate_dataset(self) -> pd.DataFrame:
        """主数据生成流程"""
        # VIX采样
        vix_data = self.download_vix_data()
        sampled_dates = self.stratified_sampling(vix_data)
        
        all_samples = []
        total_dates = sum(len(dates) for dates in sampled_dates.values())
        
        logger.info(f"🚀 开始生成数据: 共 {total_dates} 天")
        
        processed = 0
        for regime, dates in sampled_dates.items():
            logger.info(f"\n📊 处理 {regime} 市场 ({len(dates)} 天)")
            
            for date in dates:
                processed += 1
                logger.info(f"  [{processed}/{total_dates}] {date.date()}")
                
                tickers = self.get_top_tickers(date)
                
                for ticker in tickers:
                    try:
                        # 模拟分析
                        analysis = self.simulate_butterfly_analysis(ticker, date)
                        if analysis is None:
                            continue
                        
                        # 计算评估日期
                        dte = analysis['butterfly']['dte']
                        eval_date, _ = self.calculate_evaluation_date(date, dte)
                        
                        # 获取未来价格
                        future_data = yf.download(
                            ticker,
                            start=eval_date,
                            end=eval_date + timedelta(days=3),
                            progress=False
                        )
                        
                        if len(future_data) == 0:
                            continue
                        
                        future_price = future_data['Close'].iloc[0]
                        
                        # 计算ROI和标签
                        roi = self.calculate_butterfly_roi(
                            future_price,
                            analysis['butterfly']['lower_strike'],
                            analysis['butterfly']['center_strike'],
                            analysis['butterfly']['upper_strike'],
                            analysis['butterfly']['net_debit'],
                            analysis['butterfly']['max_profit']
                        )
                        
                        label = self.classify_roi(roi)
                        
                        # 提取特征 (需要导入features模块)
                        from ml.features import extract_features_v2
                        features = extract_features_v2(analysis)
                        
                        # 合并
                        sample = {
                            **features,
                            'label': label,
                            '_ticker': ticker,
                            '_date': date,
                            '_regime': regime,
                            '_debug_roi': roi
                        }
                        
                        all_samples.append(sample)
                        
                    except Exception as e:
                        logger.debug(f"    ⚠️ {ticker}: {e}")
                        continue
                
                if processed % 10 == 0:
                    logger.info(f"  ✅ 已收集 {len(all_samples)} 个样本")
        
        df = pd.DataFrame(all_samples)
        logger.info(f"\n✅ 数据生成完成: {len(df)} 个样本")
        
        return df
    
    def validate_dataset(self, df: pd.DataFrame):
        """数据质量验证"""
        logger.info("\n🔍 数据质量检查:")
        logger.info(f"  总样本数: {len(df)}")
        
        # 标签分布
        label_dist = df['label'].value_counts(normalize=True).sort_index()
        logger.info(f"  标签分布:")
        for label, pct in label_dist.items():
            logger.info(f"    Class {label}: {pct:.1%}")
        
        # 缺失值
        missing = df.isnull().sum().sum()
        logger.info(f"  缺失值: {missing}")
        
        # ROI统计
        roi = df['_debug_roi'].values
        logger.info(f"\n  ROI统计:")
        logger.info(f"    均值: {np.mean(roi):.2%}")
        logger.info(f"    中位数: {np.median(roi):.2%}")
        logger.info(f"    P25: {np.percentile(roi, 25):.2%}")
        logger.info(f"    P75: {np.percentile(roi, 75):.2%}")
    
    def run(self):
        """执行完整流程"""
        logger.info("=" * 70)
        logger.info("🦋 ButterQuant ML 训练数据生成器")
        logger.info("=" * 70)
        
        df = self.generate_dataset()
        self.validate_dataset(df)
        
        output_path = self.output_dir / "training_data_deep.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"\n💾 数据已保存: {output_path}")
        
        return df


if __name__ == "__main__":
    generator = HistoricalDataGenerator()
    df = generator.run()