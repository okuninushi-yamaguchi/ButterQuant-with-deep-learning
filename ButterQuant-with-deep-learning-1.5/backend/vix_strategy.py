"""
VIX波动率交易策略分析器 / VIX Volatility Trading Strategy Analyzer
基于VIX Z-Score进行SPX跨式期权交易信号生成 / SPX straddle trading signal generation based on VIX Z-Score
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VIXStraddleStrategy:
    """VIX驱动的跨式期权策略分析器 / VIX-driven straddle option strategy analyzer"""
    
    def __init__(self, lookback_days: int = 300):  # 300天 / 300 days
        self.lookback_days = lookback_days
        self.vix_ticker = yf.Ticker('^VIX')
        self.spx_ticker = yf.Ticker('^SPX')
        
    def get_vix_indicators(self) -> Dict:
        """获取VIX指标数据 / Get VIX indicator data"""
        try:
            # 获取VIX历史数据 / Get VIX historical data
            # 股市每年约252个交易日 / About 252 trading days per year
            # 我们使用 1.6 倍加上 30 天的缓冲 / Use 1.6x plus 30-day buffer
            calendar_days = int(self.lookback_days * 1.6) + 30
            end_date = datetime.now()
            start_date = end_date - timedelta(days=calendar_days)
            
            vix_data = self.vix_ticker.history(start=start_date, end=end_date)
            
            actual_points = len(vix_data)
            if actual_points < self.lookback_days:
                logger.warning(f"VIX数据不足 / Insufficient VIX data (Requested {self.lookback_days}, Got {actual_points})")
            
            # 计算指标 / Calculate indicators
            # 如果实际点数少于lookback_days，则使用所有可用点数 / Use all points if fewer than lookback_days
            use_days = min(actual_points, self.lookback_days)
            recent_data = vix_data['Close'].tail(use_days)
            current_vix = vix_data['Close'].iloc[-1]
            vix_sma = recent_data.mean()
            vix_std = recent_data.std()
            
            # 计算Z-Score / Calculate Z-Score
            z_score = (current_vix - vix_sma) / vix_std if vix_std > 0 else 0
            
            # VIX百分位数 / VIX Percentile
            vix_percentile = (recent_data < current_vix).sum() / len(recent_data) * 100
            
            return {
                'current_vix': float(current_vix),
                'vix_sma': float(vix_sma),
                'vix_std': float(vix_std),
                'z_score': float(z_score),
                'vix_percentile': float(vix_percentile),
                'data_points': len(recent_data),
                'last_update': vix_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"获取VIX指标失败: {e} / Failed to get VIX indicators")
            return {}
    
    def get_spx_options_chain(self, target_dte: int = 30) -> Dict:
        """获取SPX期权链数据 / Get SPX options chain data"""
        try:
            # 获取当前SPX价格 / Get current SPX price
            spx_data = self.spx_ticker.history(period='1d')
            current_spx = spx_data['Close'].iloc[-1]
            
            # 获取期权到期日 / Get option expirations
            expirations = self.spx_ticker.options
            if not expirations:
                return {'error': '无法获取SPX期权到期日 / Unable to get SPX option expirations'}
            
            # 找到最接近目标DTE的到期日 / Find expiry closest to target DTE
            target_date = datetime.now() + timedelta(days=target_dte)
            best_expiry = min(expirations, 
                            key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
            
            # 获取期权链 / Get option chain
            option_chain = self.spx_ticker.option_chain(best_expiry)
            calls = option_chain.calls
            puts = option_chain.puts
            
            # 找到最接近平值的期权 / Find ATM strike
            atm_strike = self._find_atm_strike(calls, current_spx)
            
            # 获取平值看涨和看跌期权
            atm_call = calls[calls['strike'] == atm_strike].iloc[0] if len(calls[calls['strike'] == atm_strike]) > 0 else None
            atm_put = puts[puts['strike'] == atm_strike].iloc[0] if len(puts[puts['strike'] == atm_strike]) > 0 else None
            
            if atm_call is None or atm_put is None:
                return {'error': f'无法找到平值期权 / Unable to find ATM options for {atm_strike}'}
            
            # 计算跨式期权数据 / Calculate straddle metrics
            straddle_data = self._calculate_straddle_metrics(atm_call, atm_put, current_spx, best_expiry)
            
            return {
                'spx_price': float(current_spx),
                'expiry_date': best_expiry,
                'strike': float(atm_strike),
                'call_data': {
                    'price': float(atm_call['lastPrice']),
                    'bid': float(atm_call['bid']),
                    'ask': float(atm_call['ask']),
                    'volume': int(atm_call['volume']) if not pd.isna(atm_call['volume']) else 0,
                    'iv': float(atm_call['impliedVolatility']) if not pd.isna(atm_call['impliedVolatility']) else 0.0
                },
                'put_data': {
                    'price': float(atm_put['lastPrice']),
                    'bid': float(atm_put['bid']),
                    'ask': float(atm_put['ask']),
                    'volume': int(atm_put['volume']) if not pd.isna(atm_put['volume']) else 0,
                    'iv': float(atm_put['impliedVolatility']) if not pd.isna(atm_put['impliedVolatility']) else 0.0
                },
                'straddle': straddle_data
            }
            
        except Exception as e:
            logger.error(f"获取SPX期权链失败: {e} / Failed to get SPX option chain")
            return {'error': str(e)}
    
    def _find_atm_strike(self, options_df: pd.DataFrame, current_price: float) -> float:
        """找到最接近平值的行权价 / Find the strike closest to ATM"""
        return options_df.iloc[(options_df['strike'] - current_price).abs().argsort()[:1]]['strike'].iloc[0]
    
    def _calculate_straddle_metrics(self, call_data: pd.Series, put_data: pd.Series, 
                                  spx_price: float, expiry: str) -> Dict:
        """计算跨式期权指标 / Calculate straddle metrics"""
        try:
            # 使用中间价计算 / Use mid price calculation
            call_mid = (call_data['bid'] + call_data['ask']) / 2 if call_data['bid'] > 0 and call_data['ask'] > 0 else call_data['lastPrice']
            put_mid = (put_data['bid'] + put_data['ask']) / 2 if put_data['bid'] > 0 and put_data['ask'] > 0 else put_data['lastPrice']
            
            # 跨式期权总成本 / Total straddle cost
            total_cost = call_mid + put_mid
            
            # 盈亏平衡点 / Breakeven points
            upper_breakeven = call_data['strike'] + total_cost
            lower_breakeven = call_data['strike'] - total_cost
            
            # 到期天数 / Days to expiry
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
            days_to_expiry = (expiry_date - datetime.now()).days
            
            # 最大损失 / Max loss
            max_loss_long = total_cost
            
            # 目标收益 / Target profit
            target_profit_long = total_cost * 2.5
            
            # 计算隐含波动率平均值 / Calculate avg implied volatility
            avg_iv = (call_data['impliedVolatility'] + put_data['impliedVolatility']) / 2
            
            return {
                'total_cost': float(total_cost),
                'call_mid_price': float(call_mid),
                'put_mid_price': float(put_mid),
                'upper_breakeven': float(upper_breakeven),
                'lower_breakeven': float(lower_breakeven),
                'days_to_expiry': int(days_to_expiry) if not pd.isna(days_to_expiry) else 0,
                'max_loss_long': float(max_loss_long),
                'target_profit_long': float(target_profit_long),
                'profit_loss_ratio': float(target_profit_long / max_loss_long),
                'avg_implied_volatility': float(avg_iv),
                'breakeven_move_pct': float(total_cost / spx_price * 100)  # 需要移动的百分比
            }
            
        except Exception as e:
            logger.error(f"计算跨式期权指标失败: {e} / Failed to calculate straddle metrics")
            return {}
    
    def generate_signal(self, z_score: float) -> Dict:
        """基于VIX Z-Score生成交易信号 / Generate trading signal based on VIX Z-Score"""
        if z_score <= -1.0:
            signal = 'LONG_STRADDLE'
            confidence = min(abs(z_score) * 0.3, 1.0)  # Z-Score越低，信心越强 / Lower Z-Score, stronger confidence
            description = f'VIX相对较低(Z-Score: {z_score:.2f})，建议做多跨式期权 / VIX low, suggest Long Straddle'
        elif z_score >= 1.0:
            signal = 'SHORT_STRADDLE'
            confidence = min(abs(z_score) * 0.3, 1.0)
            description = f'VIX相对较高(Z-Score: {z_score:.2f})，建议做空跨式期权 / VIX high, suggest Short Straddle'
        else:
            signal = 'HOLD'
            confidence = 0.1
            description = f'VIX处于中性区间(Z-Score: {z_score:.2f}) / VIX neutral'
        
        return {
            'signal': signal,
            'confidence': float(confidence),
            'z_score': float(z_score),
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_complete_analysis(self) -> Dict:
        """获取完整的VIX策略分析 / Get complete VIX strategy analysis"""
        try:
            # 获取VIX指标 / Get VIX indicators
            vix_indicators = self.get_vix_indicators()
            if not vix_indicators:
                return {'error': '无法获取VIX数据 / Unable to get VIX data'}
            
            # 获取SPX期权数据 / Get SPX options data
            options_data = self.get_spx_options_chain()
            if 'error' in options_data:
                return {'error': f'期权数据获取失败 / Option data failed: {options_data["error"]}'}
            
            # 生成交易信号 / Generate trading signal
            signal_data = self.generate_signal(vix_indicators['z_score'])
            
            # 组合完整分析结果 / Combine complete analysis results
            return {
                'vix_analysis': vix_indicators,
                'options_analysis': options_data,
                'trading_signal': signal_data,
                'strategy_summary': {
                    'current_recommendation': signal_data['signal'],
                    'confidence_level': signal_data['confidence'],
                    'risk_reward_ratio': options_data.get('straddle', {}).get('profit_loss_ratio', 0),
                    'breakeven_move_required': options_data.get('straddle', {}).get('breakeven_move_pct', 0),
                    'days_to_expiry': options_data.get('straddle', {}).get('days_to_expiry', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"完整分析失败: {e} / Complete analysis failed")
            return {'error': str(e)}

# 测试函数 / Test function
if __name__ == "__main__":
    strategy = VIXStraddleStrategy()
    result = strategy.get_complete_analysis()
    
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))