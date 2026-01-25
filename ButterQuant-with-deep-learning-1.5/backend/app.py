# -*- coding: utf-8 -*-

import os
import json
import time
import logging
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
import redis

# 引入之前的模块 / Import previous modules
from analyzer import ButterflyAnalyzer
from database import DatabaseManager
from daily_scanner import DailyScanner
from vix_strategy import VIXStraddleStrategy

# ==================== 配置 / Configuration ====================
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
CACHE_TTL = int(os.getenv('CACHE_TTL', 300))  # 5分钟 / 5 minutes
DEBUG_MODE = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

import numpy as np
import math

def sanitize_data(data):
    """递归将 NaN/Inf 转换为 None (JSON null) / Recursively convert NaN/Inf to None (JSON null)"""
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
    return data

# ==================== 初始化 / Initialization ====================
app = Flask(__name__)
CORS(app)

# 配置日志 / Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化数据库管理器 / Initialize Database Manager
db_manager = DatabaseManager()

# Redis 连接池（复用连接） / Redis connection pool (connection reuse)
try:
    redis_pool = redis.ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        max_connections=50,
        decode_responses=True,
        socket_timeout=1
    )
    redis_client = redis.Redis(connection_pool=redis_pool)
    redis_client.ping() # 立即测试连接 / Test connection immediately
    redis_available = True
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Running in no-cache mode.")
    redis_client = None
    redis_available = False

# ==================== 缓存装饰器 / Cache Decorator ====================
def cache_result(cache_key_func, ttl=CACHE_TTL):
    """通用缓存装饰器 (带优雅降级) / General cache decorator (with graceful degradation)"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 如果 Redis 不可用，直接执行函数 / If Redis is not available, execute function directly
            if not redis_available or not redis_client:
                return func(*args, **kwargs)

            try:
                # 生成缓存键 / Generate cache key
                cache_key = cache_key_func(*args, **kwargs)
                
                # 尝试从缓存获取 / Try to get from cache
                cached = redis_client.get(cache_key)
                if cached:
                    logger.info(f"Cache HIT: {cache_key}")
                    return json.loads(cached), True  # 返回数据 + 缓存标记 / Return data + cache flag
            except Exception as e:
                logger.error(f"Redis read error: {e}")
            
            # 缓存未命中，执行函数 / Cache miss, execute function
            logger.info(f"Cache MISS: {cache_key}")
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                # 如果函数执行出错，直接抛出 / Raise if function execution fails
                raise e
            
            # 存入缓存 / Store in cache
            try:
                if result:
                    redis_client.setex(
                        cache_key,
                        ttl,
                        json.dumps(result, ensure_ascii=False)
                    )
            except Exception as e:
                logger.error(f"Redis write error: {e}")
            
            return result, False
        
        return wrapper
    return decorator

# ==================== 业务逻辑 / Business Logic ====================

@app.route('/api/analyze', methods=['POST'])
@cache_result(
    cache_key_func=lambda: f"analysis:{request.json.get('ticker', 'AAPL').upper()}",
    ttl=CACHE_TTL
)
def analyze():
    """主分析接口（带缓存） / Main analysis interface (with cache)"""
    start_time = time.time()
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL').upper()
        
        if not ticker or len(ticker) > 10:
             return jsonify({
                'success': False,
                'error': 'Invalid ticker'
            }), 400

        logger.info(f"Starting analysis for {ticker}")
        
        # 实例化分析器 / Instantiate analyzer
        analyzer = ButterflyAnalyzer(ticker)
        
        # --- 耗时操作 / Time-consuming operations ---
        result = analyzer.full_analysis()
        
        try:
             db_manager.save_analysis(result)
             db_manager.save_daily_metrics(ticker, result)
        except Exception as db_e:
             logger.error(f"DB Save Error: {db_e}")
             # DB 错误不应该导致 API 失败 / DB errors should not cause API failure
        
        logger.info(f"Analysis finished in {time.time() - start_time:.2f}s")
        
        # 净化数据，防止 NaN 导致前端崩溃 / Sanitize data to prevent NaN crashing frontend
        clean_result = sanitize_data(result)
        
        return jsonify({
            'success': True,
            'data': clean_result,
            'from_cache': False
        })

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.after_request
def after_request(response):
    """标准化响应格式 / Standardize response format"""
    if response.status_code == 200 and response.is_json:
        try:
            data = response.get_json()
            pass 
        except:
            pass
    return response

@app.route('/api/trades', methods=['GET'])
def get_trades_endpoint():
    """获取交易历史记录 / Get transaction history"""
    try:
        limit = request.args.get('limit', default=50, type=int)
        trades = db_manager.get_trades(limit=limit)
        return jsonify({
            'success': True,
            'data': trades
        })
    except Exception as e:
        logger.error(f"Failed to fetch trades: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/vix-strategy', methods=['GET'])
@cache_result(
    cache_key_func=lambda: "vix_strategy_analysis",
    ttl=300  # 5分钟缓存 / 5 minutes cache
)
def vix_strategy_analysis():
    """VIX波动率交易策略分析接口 / VIX volatility trading strategy analysis interface"""
    start_time = time.time()
    try:
        logger.info("Starting VIX strategy analysis")
        
        # 实例化VIX策略分析器 / Instantiate VIX strategy analyzer
        vix_analyzer = VIXStraddleStrategy()
        
        # 执行完整分析 / Execute full analysis
        result = vix_analyzer.get_complete_analysis()
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
        logger.info(f"VIX analysis finished in {time.time() - start_time:.2f}s")
        
        # 净化数据 / Sanitize data
        clean_result = sanitize_data(result)
        
        return jsonify({
            'success': True,
            'data': clean_result,
            'from_cache': False
        })

    except Exception as e:
        logger.error(f"VIX strategy analysis failed: {e}", exc_info=True)
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/vix-indicators', methods=['GET'])
@cache_result(
    cache_key_func=lambda: "vix_indicators",
    ttl=60  # 1分钟缓存，更频繁更新 / 1 minute cache, updated more frequently
)
def vix_indicators():
    """获取VIX指标数据 / Get VIX indicator data"""
    try:
        vix_analyzer = VIXStraddleStrategy()
        indicators = vix_analyzer.get_vix_indicators()
        
        if not indicators:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch VIX indicators'
            }), 500
        
        return jsonify({
            'success': True,
            'data': indicators,
            'from_cache': False
        })

    except Exception as e:
        logger.error(f"VIX indicators failed: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/spx-options', methods=['GET'])
@cache_result(
    cache_key_func=lambda: f"spx_options_{request.args.get('dte', 30)}",
    ttl=300  # 5分钟缓存 / 5 minutes cache
)
def spx_options():
    """获取SPX期权链数据 / Get SPX options chain data"""
    try:
        dte = int(request.args.get('dte', 30))  # 默认30天到期 / Default 30 days to maturity
        
        vix_analyzer = VIXStraddleStrategy()
        options_data = vix_analyzer.get_spx_options_chain(target_dte=dte)
        
        if 'error' in options_data:
            return jsonify({
                'success': False,
                'error': options_data['error']
            }), 500
        
        return jsonify({
            'success': True,
            'data': options_data,
            'from_cache': False
        })

    except Exception as e:
        logger.error(f"SPX options failed: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口 / Health check interface"""
    status = {
        'status': 'ok',
        'timestamp': time.time(),
        'redis': 'connected' if redis_available else 'disconnected'
    }
    return jsonify(status)

@app.route('/api/rankings', methods=['GET'])
def get_rankings():
    """获取排行榜 / Get rankings"""
    try:
        # 获取 limit 参数 / Get limit parameter
        limit = int(request.args.get('limit', 1000))
        # 使用优化后的 DB manager / Use optimized DB manager
        rankings = db_manager.get_latest_ranking(limit=limit)
        clean_rankings = sanitize_data(rankings)
        return jsonify({
            'success': True,
            'data': clean_rankings
        })
    except Exception as e:
        logger.error(f"Rankings error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scan', methods=['POST'])
def start_scan():
    """触发扫描 (依然使用线程，但改为非阻塞) / Trigger scan (still use thread, but non-blocking)"""
    try:
        # 检查是否已有扫描在运行 / Check if scan is already running
        # 这里简化处理，直接启动新线程 / Simplified processing, start new thread directly
        from threading import Thread
        
        def run_scan_job():
            scanner = DailyScanner()
            scanner.run()
            
        thread = Thread(target=run_scan_job)
        thread.start()
        
        return jsonify({
            'success': True, 
            'message': 'Scan started in background'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # 开发模式 / Development mode
    logger.info("Starting Flask server on port 5001...")
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
