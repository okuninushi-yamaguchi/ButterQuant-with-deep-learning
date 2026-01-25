# -*- coding: utf-8 -*-
"""
Execution Engine (Alpaca) - Alpaca 自动化执行引擎
"""

import os
import sys
import json
import logging
import time
from datetime import datetime

# 确保可以导入 backend 和当前目录的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from alpaca_trader import AlpacaTrader
from backend.database import DatabaseManager

# 导入ML推理引擎
try:
    from backend.ml_inference import get_inference_engine
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# 导入特征提取器
try:
    from backend.ml.features import FeatureExtractor, extract_features_v2
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AlpacaExecutionEngine')

class ExecutionEngine:
    def __init__(self):
        self.trader = AlpacaTrader()
        self.db = DatabaseManager()
        # 排名文件仍在 backend/data 下
        self.ranking_file = os.path.join(project_root, 'backend/data/rankings_combined.json')
        
        self.MAX_PORTFOLIO_SIZE = 100
        self.ALLOCATION_PER_STRATEGY = 1000
        self.AGGRESSION_OFFSET = 0.05 
        self.USE_MARKET_ORDER = True
        self.EXPECTED_ROI_THRESHOLD = 0.15 
        self.ML_THRESHOLD = 0.70 
        
        self.ml_engine = None
        if ML_AVAILABLE:
            try:
                self.ml_engine = get_inference_engine()
                logger.info(f"✅ ML推理引擎已加载 (版本: {self.ml_engine.get_model_version()})")
            except Exception as e:
                logger.warning(f"⚠️ ML引擎加载失败: {e}")

    def load_rankings(self):
        if not os.path.exists(self.ranking_file):
            logger.error(f"找不到榜单文件: {self.ranking_file}")
            return []
        with open(self.ranking_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_features_from_analysis(self, analysis_result: dict) -> dict:
        if FEATURES_AVAILABLE:
            return extract_features_v2(analysis_result)
        # Fallback logic...
        return {} # Simplified for now

    def run_daily_execution(self):
        logger.info(f"==== 开始 Alpaca 每日自动化执行系统 ====")
        if not self.trader.connect():
            logger.error("无法连接到 Alpaca")
            return
        
        try:
            current_symbol_set = self.trader.get_active_symbols()
            all_candidates = self.load_rankings()
            if not all_candidates:
                return

            # Track A: AI
            ai_candidates = self._filter_ai_candidates(all_candidates)
            sorted_ai_list = sorted(ai_candidates, key=lambda x: x.get('expected_roi', 0), reverse=True)
            logger.info(f"🤖 [AI Track] 识别出 {len(sorted_ai_list)} 个高胜率机会")
            self._execute_batch(sorted_ai_list, current_symbol_set, 'AI', max_count=5)

            # Track B: Baseline
            sorted_baseline_list = sorted(all_candidates, key=lambda x: (x.get('score') or 0), reverse=True)
            sorted_baseline_list = [c for c in sorted_baseline_list if (c.get('score') or 0) > 60]
            logger.info(f"📊 [Baseline Track] 选取 Top 5 传统技术高分机会")
            self._execute_batch(sorted_baseline_list, current_symbol_set, 'Baseline', max_count=5)

        finally:
            self.trader.disconnect()

    def _filter_ai_candidates(self, all_candidates: list) -> list:
        ai_candidates = []
        for c in all_candidates:
            ticker = c.get('ticker', '')
            expected_roi = c.get('ml_expected_roi') or 0
            if expected_roi >= self.EXPECTED_ROI_THRESHOLD:
                ai_candidates.append({**c, 'expected_roi': expected_roi})
        return ai_candidates

    def _execute_batch(self, candidates, current_symbol_set, strategy_label, max_count=5):
        count = 0
        for item in candidates:
            if count >= max_count: break
            ticker = item['ticker']
            if ticker in current_symbol_set: continue
            
            analysis = self.db.get_latest_analysis(ticker)
            if not analysis: continue
            butterfly = analysis.get('butterfly', {})
            if not butterfly: continue
            
            expiry = butterfly.get('expiry')
            details = {
                'lower': butterfly.get('lower_strike'),
                'center': butterfly.get('center_strike'),
                'upper': butterfly.get('upper_strike'),
                'expiry': expiry,
                'type': butterfly.get('butterfly_type', 'CALL').upper(),
                'price': butterfly.get('net_debit', 0.50),
                'is_credit_strategy': butterfly.get('is_credit_strategy', False)
            }
            
            logger.info(f"[{strategy_label}] 尝试下单: {ticker}")
            res = self.trader.place_butterfly_order(ticker, details, strategy_type=strategy_label)
            if res['status'] == 'submitted':
                count += 1
            time.sleep(1)

if __name__ == "__main__":
    engine = ExecutionEngine()
    engine.run_daily_execution()
