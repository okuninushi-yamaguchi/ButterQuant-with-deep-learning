import sqlite3
import psycopg2
from psycopg2 import extras
import json
from datetime import datetime
from contextlib import contextmanager
import threading
import logging
import os
from ticker_utils import get_tickers_with_tags

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理器（线程安全 + 性能优化版） / Database Manager (thread-safe + performance optimized)"""
    
    def __init__(self, db_path='data/history.db', research_db_path='data/market_research.db'):
        # 确保路径是绝对路径或相对于 backend 目录 / Ensure path is absolute or relative to backend directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if not os.path.isabs(db_path):
            self.db_path = os.path.join(base_dir, db_path)
        else:
            self.db_path = db_path
            
        if not os.path.isabs(research_db_path):
            self.research_db_path = os.path.join(base_dir, research_db_path)
        else:
            self.research_db_path = research_db_path
            
        self._local = threading.local()  # 每个线程独立连接 / Independent connection for each thread
        
        # PostgreSQL 配置 / PostgreSQL configuration
        self.use_pg = os.getenv('DB_HOST') is not None
        self.pg_params = {
            "dbname": os.getenv('DB_NAME', 'butterquant'),
            "user": os.getenv('DB_USER', 'postgres'),
            "password": os.getenv('DB_PASSWORD', 'butterquant_pass'),
            "host": os.getenv('DB_HOST', 'localhost'),
            "port": int(os.getenv('DB_PORT', 5432))
        }
        
        # 确保目录存在 / Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # --- 预加载 Ticker 标签 --- / --- Preload Ticker Tags ---
        # 假设 doc 目录在 backend 的上一级的 doc 目录 / Assume doc directory is at project root
        project_root = os.path.dirname(base_dir) 
        doc_dir = os.path.join(project_root, 'docs')
        
        file_map = {
            'NASDAQ': os.path.join(doc_dir, 'nas100.md'),
            'SP500': os.path.join(doc_dir, 'sp500.md')
        }
        
        try:
            self.ticker_tags = get_tickers_with_tags(file_map)
            logger.info(f"Loaded tags for {len(self.ticker_tags)} tickers")
        except Exception as e:
            logger.error(f"Failed to load ticker tags: {e}")
            self.ticker_tags = {}
        
        # 初始化数据库 / Initialize database
        self.init_db()
        self.create_indexes()
    
    @contextmanager
    def get_connection(self, db_path=None):
        """获取数据库连接（优先使用 PostgreSQL，回退到 SQLite） / Get database connection (prioritize PostgreSQL, fallback to SQLite)"""
        if self.use_pg:
            if not hasattr(self._local, 'pg_conn') or self._local.pg_conn.closed:
                try:
                    conn = psycopg2.connect(**self.pg_params)
                    conn.autocommit = False # 我们手动 commit / Manual commit
                    self._local.pg_conn = conn
                except Exception as e:
                    logger.error(f"Failed to connect to PostgreSQL: {e}. Falling back to SQLite.")
                    self.use_pg = False # 暂时停用 PG / Temporarily disable PG
            
            if self.use_pg:
                conn = self._local.pg_conn
                try:
                    yield conn
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.error(f"PostgreSQL error: {e}")
                    raise
                return

        target_path = db_path or self.db_path
        
        # 每个线程维护独立连接 / Each thread maintains its own connection
        # 使用 target_path 作为 key 来支持多数据库 / Use target_path as key to support multiple databases
        if not hasattr(self._local, 'conns'):
            self._local.conns = {}
            
        if target_path not in self._local.conns:
            conn = sqlite3.connect(
                target_path,
                check_same_thread=False,
                timeout=30.0  # 30秒超时，等待锁释放 / 30s timeout, wait for lock release
            )
            conn.row_factory = sqlite3.Row  # 返回字典式结果 / Return dict-like results
            
            # 性能优化配置 / Performance optimization configuration
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging (关键优化 / Key optimization)
            conn.execute("PRAGMA synchronous=NORMAL")  # 平衡安全与性能 / Balance safety and performance
            conn.execute("PRAGMA cache_size=-64000")  # 64MB 缓存 / 64MB Cache
            conn.execute("PRAGMA temp_store=MEMORY")  # 临时表放内存 / Temporary tables in memory
            
            self._local.conns[target_path] = conn
        
        conn = self._local.conns[target_path]
        
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error on {target_path}: {e}")
            raise

    def init_db(self):
        """初始化数据库表 / Initialize database tables"""
        if self.use_pg:
            # PostgreSQL 初始化逻辑在 init-db.sql 中处理了，但可以这里确保 / PostgreSQL init logic is handled in init-db.sql, but can be ensured here
            return

        # 主历史表 / Main history table
        with self.get_connection(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    total_score REAL,
                    butterfly_type TEXT,
                    recommendation TEXT,
                    full_result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        # 深度分析表 (用于深度学习训练) / Deep analysis table (for deep learning training)
        with self.get_connection(self.research_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    current_price REAL,
                    
                    -- 傅立叶分析 / Fourier Analysis
                    trend_direction TEXT,
                    trend_slope REAL,
                    dominant_period REAL,
                    
                    -- ARIMA 预测 / ARIMA Forecast
                    predicted_price REAL,
                    prediction_lower REAL,
                    prediction_upper REAL,
                    price_stability REAL,
                    
                    -- GARCH 波动率 / GARCH Volatility
                    predicted_vol REAL,
                    current_iv REAL,
                    vol_mispricing REAL,
                    iv_percentile REAL,
                    
                    -- Greeks
                    delta REAL,
                    gamma REAL,
                    vega REAL,
                    theta REAL,
                    
                    -- 策略指标 / Strategy Metrics
                    butterfly_type TEXT,
                    max_profit REAL,
                    max_loss REAL,
                    profit_ratio REAL,
                    prob_profit REAL,
                    
                    -- 评分 / Scoring
                    total_score REAL,
                    recommendation TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- 唯一约束（防止重复扫描） / Unique constraint (prevent duplicate scans)
                    UNIQUE(ticker, analysis_date)
                )
            ''')
            
            # 交易执行记录表 / Trade execution history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    order_id INTEGER,
                    order_ref TEXT, -- ButterAI 或 ButterBaseline / Order reference
                    strategy_type TEXT,
                    butterfly_type TEXT,
                    strikes TEXT, -- JSON 字符串 / JSON string
                    expiry TEXT,
                    quantity INTEGER,
                    status TEXT, -- SUBMITTED, FILLED, CANCELLED, ERROR / Order status
                    theoretical_price REAL, -- 理想成交价 / Theoretical price
                    price REAL, -- 实际成交价 (Fill Price) / Actual fill price
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def create_indexes(self):
        """创建索引（提升查询性能） / Create indexes (improve query performance)"""
        if self.use_pg:
            return # PostgreSQL 索引在 init-db.sql 中处理 / Handled in init-db.sql
        
        indexes = [
            # 主历史表索引 / Main history table indexes
            ("idx_ticker", "analysis_history", "ticker"),
            ("idx_date", "analysis_history", "analysis_date"),
            ("idx_score", "analysis_history", "total_score"),
            ("idx_ticker_date", "analysis_history", "ticker, analysis_date"),
        ]
        
        research_indexes = [
            # 深度分析表索引 / Deep analysis table indexes
            ("idx_research_ticker", "daily_metrics", "ticker"),
            ("idx_research_date", "daily_metrics", "analysis_date"),
            ("idx_research_score", "daily_metrics", "total_score"),
            ("idx_research_recommendation", "daily_metrics", "recommendation")
        ]
        
        # 主历史表 / Main history table
        with self.get_connection(self.db_path) as conn:
            for idx_name, table, columns in indexes:
                try:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({columns})")
                except sqlite3.OperationalError:
                    pass 
            
        # 深度分析表 / Deep analysis table
        with self.get_connection(self.research_db_path) as conn:
            for idx_name, table, columns in research_indexes:
                try:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({columns})")
                except sqlite3.OperationalError:
                    pass

    def save_analysis(self, result):
        """保存分析结果到主历史表 / Save analysis results to main history table"""
        with self.get_connection(self.db_path) as conn:
            if self.use_pg:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO analysis_history (
                        ticker, analysis_date, total_score, 
                        butterfly_type, recommendation, full_result
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                ''', (
                    result.get('ticker'),
                    datetime.now(),
                    result.get('score', {}).get('total', 0),
                    result.get('fourier', {}).get('butterfly_type', 'UNKNOWN'),
                    result.get('trade_suggestion', {}).get('action', 'NEUTRAL'),
                    json.dumps(result, ensure_ascii=False)
                ))
            else:
                conn.execute('''
                    INSERT INTO analysis_history (
                        ticker, analysis_date, total_score, 
                        butterfly_type, recommendation, full_result
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result.get('ticker'),
                    datetime.now().isoformat(),
                    result.get('score', {}).get('total', 0),
                    result.get('fourier', {}).get('butterfly_type', 'UNKNOWN'),
                    result.get('trade_suggestion', {}).get('action', 'NEUTRAL'),
                    json.dumps(result, ensure_ascii=False)
                ))

    def save_trade(self, trade_data):
        """记录交易执行 / Log trade execution"""
        with self.get_connection(self.db_path) as conn:
            if self.use_pg:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades_history (
                        ticker, order_id, order_ref, strategy_type,
                        butterfly_type, strikes, expiry, quantity, status, theoretical_price, price
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    trade_data.get('ticker'),
                    trade_data.get('order_id'),
                    trade_data.get('order_ref'),
                    trade_data.get('strategy_type'),
                    trade_data.get('butterfly_type'),
                    json.dumps(trade_data.get('strikes', {}), ensure_ascii=False),
                    trade_data.get('expiry'),
                    trade_data.get('quantity', 1),
                    trade_data.get('status', 'SUBMITTED'),
                    trade_data.get('theoretical_price', 0),
                    trade_data.get('price', 0)
                ))
            else:
                conn.execute('''
                    INSERT INTO trades_history (
                        ticker, order_id, order_ref, strategy_type,
                        butterfly_type, strikes, expiry, quantity, status, theoretical_price, price
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data.get('ticker'),
                    trade_data.get('order_id'),
                    trade_data.get('order_ref'),
                    trade_data.get('strategy_type'),
                    trade_data.get('butterfly_type'),
                    json.dumps(trade_data.get('strikes', {}), ensure_ascii=False),
                    trade_data.get('expiry'),
                    trade_data.get('quantity', 1),
                    trade_data.get('status', 'SUBMITTED'),
                    trade_data.get('theoretical_price', 0),
                    trade_data.get('price', 0)
                ))

    def update_trade_status(self, order_id, status, fill_price=None):
        """更新订单状态和最终成交价 / Update order status and final fill price"""
        with self.get_connection(self.db_path) as conn:
            if self.use_pg:
                cursor = conn.cursor()
                if fill_price is not None:
                    cursor.execute('UPDATE trades_history SET status = %s, price = %s WHERE order_id = %s', (status, fill_price, order_id))
                else:
                    cursor.execute('UPDATE trades_history SET status = %s WHERE order_id = %s', (status, order_id))
            else:
                if fill_price is not None:
                    conn.execute('UPDATE trades_history SET status = ?, price = ? WHERE order_id = ?', (status, fill_price, order_id))
                else:
                    conn.execute('UPDATE trades_history SET status = ? WHERE order_id = ?', (status, order_id))

    def get_trades(self, limit=50):
        """获取交易历史记录 / Get trade execution history"""
        with self.get_connection(self.db_path) as conn:
            if self.use_pg:
                cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
                cursor.execute('SELECT * FROM trades_history ORDER BY timestamp DESC LIMIT %s', (limit,))
                return cursor.fetchall()
            else:
                cursor = conn.execute('SELECT * FROM trades_history ORDER BY timestamp DESC LIMIT ?', (limit,))
                return [dict(row) for row in cursor.fetchall()]

            
    def get_latest_ranking(self, limit=100):
        """获取最新排行榜 (优先使用预生成的 JSON 缓存) / Get latest ranking (prioritize pre-generated JSON cache)"""
        # 1. 尝试读取 JSON 缓存 / Try to read JSON cache
        try:
            json_path = os.path.join(os.path.dirname(self.db_path), 'rankings_combined.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 确保是列表 / Ensure it is a list
                    if isinstance(data, list):
                        return data[:limit]
        except Exception as e:
            logger.error(f"Failed to load rankings from JSON: {e}")

        # 2. 数据库回退方案 (如果 JSON 不存在)
        try:
            with self.get_connection(self.db_path) as conn:
                # 使用 MAX(id) 和 GROUP BY ticker 来获取每个 ticker 最新的记录
                # 这样可以避免同一个 ticker 出现多次
                cursor = conn.execute('''
                    SELECT full_result
                    FROM analysis_history
                    WHERE id IN (
                        SELECT MAX(id)
                        FROM analysis_history
                        WHERE created_at >= datetime('now', '-30 days')
                        GROUP BY ticker
                    )
                    ORDER BY total_score DESC
                    LIMIT ?
                ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    try:
                        data = json.loads(row['full_result'])
                        
                        # --- 字段映射修复 --- / --- Field Mapping Fix ---
                        # 前端依赖 'strategy' 和 'recommendation' 字段 / Frontend depends on 'strategy' and 'recommendation' fields
                        # 如果顶层没有，从嵌套对象中提取 / If not at top level, extract from nested objects
                        
                        # 1. Strategy
                        if 'strategy' not in data:
                            # 尝试从 fourier 或 butterfly 中获取 / Try to get from fourier or butterfly
                            # analyzer.py 返回的 key 是 'fourier' 和 'butterfly' / analyzer.py returns 'fourier' and 'butterfly'
                            fourier = data.get('fourier', data.get('fourier_analysis', {}))
                            butterfly = data.get('butterfly', data.get('butterfly_strategy', {}))
                            
                            if 'butterfly_type' in fourier:
                                data['strategy'] = fourier['butterfly_type']
                            elif 'type' in butterfly:
                                data['strategy'] = butterfly['type']
                            else:
                                data['strategy'] = 'UNKNOWN'
                                
                        # 2. Recommendation
                        if 'recommendation' not in data:
                            # analyzer.py 返回 'score' 和 'trade_suggestion' / analyzer.py returns 'score' and 'trade_suggestion'
                            score = data.get('score', data.get('scoring', {}))
                            trade = data.get('trade_suggestion', {})
                            
                            if 'recommendation' in score:
                                data['recommendation'] = score['recommendation']
                            elif 'action' in trade:
                                data['recommendation'] = trade['action']
                            else:
                                data['recommendation'] = 'NEUTRAL'
                            
                        # 3. Tags (动态获取) / 3. Tags (dynamically fetched)
                        if 'tags' not in data:
                            ticker = data.get('ticker', '')
                            # 从预加载的映射中获取标签 / Get tags from preloaded mapping
                            tags = self.ticker_tags.get(ticker, [])
                            
                            # 保留ETF的特殊处理 / Special handling for ETFs
                            if ticker in ['SPY', 'QQQ', 'IWM', 'DIA']:
                                if 'ETF' not in tags:
                                    tags.append('ETF')
                                    
                            data['tags'] = tags

                        results.append(data)
                    except:
                        continue
                return results
        except Exception as e:
            logger.error(f"Error getting rankings: {e}")
            return []

    def save_daily_metrics(self, ticker, result):
        """保存到深度分析表（展开的数据，用于AI训练） / Save to deep analysis table (flattened data for AI training)"""
        try:
            # 安全地提取嵌套字典 / Safely extract nested dictionaries
            fourier = result.get('fourier', {})
            arima = result.get('arima', {})
            garch = result.get('garch', {})
            butterfly = result.get('butterfly', {})
            greeks = butterfly.get('greeks', {})
            score = result.get('score', {})
            
            with self.get_connection(self.research_db_path) as conn:
                if self.use_pg:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO daily_metrics (
                            ticker, analysis_date, current_price,
                            trend_direction, trend_slope, dominant_period,
                            predicted_price, prediction_lower, prediction_upper, price_stability,
                            predicted_vol, current_iv, vol_mispricing, iv_percentile,
                            delta, gamma, vega, theta,
                            butterfly_type, max_profit, max_loss, profit_ratio, prob_profit,
                            total_score, recommendation
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, analysis_date) DO UPDATE SET
                            current_price = EXCLUDED.current_price,
                            total_score = EXCLUDED.total_score,
                            recommendation = EXCLUDED.recommendation
                    ''', (
                        ticker,
                        datetime.now().date(),
                        result.get('current_price', 0),
                        fourier.get('trend_direction'),
                        fourier.get('trend_slope'),
                        fourier.get('dominant_period_days'),
                        arima.get('mean_forecast'),
                        arima.get('lower_bound', [0])[0] if isinstance(arima.get('lower_bound'), list) else arima.get('lower_bound', 0),
                        arima.get('upper_bound', [0])[0] if isinstance(arima.get('upper_bound'), list) else arima.get('upper_bound', 0),
                        score.get('components', {}).get('stability', 0),
                        garch.get('predicted_vol'),
                        garch.get('current_iv'),
                        garch.get('vol_mispricing'),
                        garch.get('iv_percentile'),
                        greeks.get('delta'),
                        greeks.get('gamma'),
                        greeks.get('vega'),
                        greeks.get('theta'),
                        fourier.get('butterfly_type'),
                        butterfly.get('max_profit'),
                        butterfly.get('max_loss'),
                        butterfly.get('profit_ratio'),
                        butterfly.get('prob_profit'),
                        score.get('total'),
                        result.get('trade_suggestion', {}).get('action')
                    ))
                else:
                    conn.execute('''
                        INSERT OR REPLACE INTO daily_metrics (
                            ticker, analysis_date, current_price,
                            trend_direction, trend_slope, dominant_period,
                            predicted_price, prediction_lower, prediction_upper, price_stability,
                            predicted_vol, current_iv, vol_mispricing, iv_percentile,
                            delta, gamma, vega, theta,
                            butterfly_type, max_profit, max_loss, profit_ratio, prob_profit,
                            total_score, recommendation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker,
                        datetime.now().date().isoformat(),
                        result.get('current_price', 0),
                        fourier.get('trend_direction'),
                        fourier.get('trend_slope'),
                        fourier.get('dominant_period_days'),
                        arima.get('mean_forecast'),
                        arima.get('lower_bound', [0])[0] if isinstance(arima.get('lower_bound'), list) else arima.get('lower_bound', 0),
                        arima.get('upper_bound', [0])[0] if isinstance(arima.get('upper_bound'), list) else arima.get('upper_bound', 0),
                        score.get('components', {}).get('stability', 0),
                        garch.get('predicted_vol'),
                        garch.get('current_iv'),
                        garch.get('vol_mispricing'),
                        garch.get('iv_percentile'),
                        greeks.get('delta'),
                        greeks.get('gamma'),
                        greeks.get('vega'),
                        greeks.get('theta'),
                        fourier.get('butterfly_type'),
                        butterfly.get('max_profit'),
                        butterfly.get('max_loss'),
                        butterfly.get('profit_ratio'),
                        butterfly.get('prob_profit'),
                        score.get('total'),
                        result.get('trade_suggestion', {}).get('action')
                    ))
        except Exception as e:
            # 这里的错误不应该阻断主流程 / Errors here should not block the main process
            logger.error(f"Failed to save daily metrics for {ticker}: {e}")
            
    def get_latest_analysis(self, ticker):
        """获取指定 Ticker 的最新分析结果 / Get latest analysis for a specific Ticker"""
        query = "SELECT full_result FROM analysis_history WHERE ticker = ? ORDER BY analysis_date DESC LIMIT 1"
        if self.use_pg:
            query = query.replace('?', '%s')
            
        with self.get_connection(self.db_path) as conn:
            if self.use_pg:
                cursor = conn.cursor()
                cursor.execute(query, (ticker,))
                res = cursor.fetchone()
                return json.loads(res[0]) if res else None
            else:
                res = conn.execute(query, (ticker,)).fetchone()
                return json.loads(res[0]) if res else None

    def close_all(self):
        """关闭所有连接 / Close all connections"""
        if hasattr(self._local, 'conns'):
            for conn in self._local.conns.values():
                try:
                    conn.close()
                except:
                    pass
            self._local.conns = {}
        
        if hasattr(self._local, 'pg_conn'):
            try:
                self._local.pg_conn.close()
            except:
                pass
            del self._local.pg_conn
