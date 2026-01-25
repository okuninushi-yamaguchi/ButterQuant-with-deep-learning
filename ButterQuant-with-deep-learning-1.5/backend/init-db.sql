-- Create daily_metrics table
CREATE TABLE IF NOT EXISTS daily_metrics (
    id SERIAL,
    ticker TEXT NOT NULL,
    analysis_date TIMESTAMPTZ NOT NULL,
    current_price DOUBLE PRECISION,
    trend_direction TEXT,
    trend_slope DOUBLE PRECISION,
    dominant_period DOUBLE PRECISION,
    predicted_price DOUBLE PRECISION,
    prediction_lower DOUBLE PRECISION,
    prediction_upper DOUBLE PRECISION,
    price_stability DOUBLE PRECISION,
    predicted_vol DOUBLE PRECISION,
    current_iv DOUBLE PRECISION,
    vol_mispricing DOUBLE PRECISION,
    iv_percentile DOUBLE PRECISION,
    delta DOUBLE PRECISION,
    gamma DOUBLE PRECISION,
    vega DOUBLE PRECISION,
    theta DOUBLE PRECISION,
    butterfly_type TEXT,
    max_profit DOUBLE PRECISION,
    max_loss DOUBLE PRECISION,
    profit_ratio DOUBLE PRECISION,
    prob_profit DOUBLE PRECISION,
    total_score DOUBLE PRECISION,
    recommendation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (analysis_date, ticker, id)
);

-- Convert to hypertable for TimescaleDB optimization
SELECT create_hypertable('daily_metrics', 'analysis_date', if_not_exists => TRUE);

-- Create analysis_history for tracking (non-hypertable is fine, or hypertable if large)
CREATE TABLE IF NOT EXISTS analysis_history (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    analysis_date TIMESTAMPTZ NOT NULL,
    total_score DOUBLE PRECISION,
    butterfly_type TEXT,
    recommendation TEXT,
    full_result JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_metrics_ticker ON daily_metrics (ticker, analysis_date DESC);
CREATE INDEX idx_history_ticker ON analysis_history (ticker, analysis_date DESC);
