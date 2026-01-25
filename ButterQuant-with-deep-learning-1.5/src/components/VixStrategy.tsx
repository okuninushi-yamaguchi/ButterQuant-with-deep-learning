import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Minus, AlertTriangle, Activity, Target } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ReferenceLine } from 'recharts';

interface VixIndicators {
  current_vix: number;
  vix_sma: number;
  vix_std: number;
  z_score: number;
  vix_percentile: number;
  data_points: number;
  last_update: string;
}

interface StraddleData {
  total_cost: number;
  call_mid_price: number;
  put_mid_price: number;
  upper_breakeven: number;
  lower_breakeven: number;
  days_to_expiry: number;
  max_loss_long: number;
  target_profit_long: number;
  profit_loss_ratio: number;
  avg_implied_volatility: number;
  breakeven_move_pct: number;
}

interface OptionsData {
  spx_price: number;
  expiry_date: string;
  strike: number;
  call_data: {
    price: number;
    bid: number;
    ask: number;
    volume: number;
    iv: number;
  };
  put_data: {
    price: number;
    bid: number;
    ask: number;
    volume: number;
    iv: number;
  };
  straddle: StraddleData;
}

interface TradingSignal {
  signal: 'LONG_STRADDLE' | 'SHORT_STRADDLE' | 'HOLD';
  confidence: number;
  z_score: number;
  description: string;
  timestamp: string;
}

interface VixAnalysisData {
  vix_analysis: VixIndicators;
  options_analysis: OptionsData;
  trading_signal: TradingSignal;
  strategy_summary: {
    current_recommendation: string;
    confidence_level: number;
    risk_reward_ratio: number;
    breakeven_move_required: number;
    days_to_expiry: number;
  };
}

const VixStrategy: React.FC = () => {
  const { t, i18n } = useTranslation();
  const [data, setData] = useState<VixAnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchVixData = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/vix-strategy');
      const result = await response.json();

      if (result.success) {
        setData(result.data);
        setError(null);
      } else {
        setError(result.error || t('common.error'));
      }
    } catch (err) {
      setError('Network error: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVixData();
    // 每5分钟刷新一次数据
    const interval = setInterval(fetchVixData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'LONG_STRADDLE':
        return <TrendingUp className="w-5 h-5 text-green-500" />;
      case 'SHORT_STRADDLE':
        return <TrendingDown className="w-5 h-5 text-red-500" />;
      default:
        return <Minus className="w-5 h-5 text-gray-500" />;
    }
  };

  const getSignalLabel = (signal: string) => {
    return t(`vix_strategy.signal.${signal}`);
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'LONG_STRADDLE':
        return 'bg-green-50 border-green-200 text-green-800';
      case 'SHORT_STRADDLE':
        return 'bg-red-50 border-red-200 text-red-800';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat(i18n.language === 'zh' ? 'zh-CN' : 'en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
        <span className="ml-3 text-gray-600">{t('common.loading')}</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center">
          <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
          <span className="text-red-800">{error}</span>
        </div>
        <button
          onClick={fetchVixData}
          className="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
        >
          {t('vix_strategy.retry')}
        </button>
      </div>
    );
  }

  if (!data ||
    !data.vix_analysis ||
    !data.options_analysis ||
    !data.trading_signal ||
    !data.options_analysis.straddle ||
    !data.options_analysis.call_data ||
    !data.options_analysis.put_data) return null;

  const { vix_analysis, options_analysis, trading_signal, strategy_summary } = data;

  // 准备盈亏图数据
  const profitLossData = [];
  const strike = options_analysis.strike;
  const totalCost = options_analysis.straddle.total_cost;

  // Safety check to prevent infinite loop if totalCost is 0
  if (totalCost > 0) {
    for (let price = strike - totalCost * 2; price <= strike + totalCost * 2; price += totalCost * 0.1) {
      const callValue = Math.max(0, price - strike);
      const putValue = Math.max(0, strike - price);
      const totalValue = callValue + putValue;
      const profit = totalValue - totalCost;

      profitLossData.push({
        price: price,
        profit: profit
      });
    }
  }

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              <Activity className="w-6 h-6 text-indigo-600" />
              {t('vix_strategy.title')}
            </h1>
            <p className="text-gray-600 mt-1">
              {t('vix_strategy.subtitle')}
            </p>
          </div>
          <button
            onClick={fetchVixData}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            {t('vix_strategy.refresh')}
          </button>
        </div>
      </div>

      {/* Trading Signal Card */}
      <div className={`rounded-lg border-2 p-6 ${getSignalColor(trading_signal.signal)}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {getSignalIcon(trading_signal.signal)}
            <div>
              <h2 className="text-xl font-bold">
                {getSignalLabel(trading_signal.signal)}
              </h2>
              <p className="text-sm opacity-80">
                {t('vix_strategy.signal.confidence')}: {(trading_signal.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold">
              {t('vix_strategy.signal.z_score')}: {trading_signal.z_score.toFixed(2)}
            </div>
            <div className="text-sm opacity-80">
              {t('vix_strategy.signal.vix_value')}: {vix_analysis.current_vix.toFixed(2)}
            </div>
          </div>
        </div>
        <div className="mt-4 p-3 bg-white bg-opacity-50 rounded">
          <p className="text-sm">
            {t(`vix_strategy.signal.${trading_signal.signal}_DESC`, {
              z_score: trading_signal.z_score.toFixed(2),
              defaultValue: trading_signal.description
            })}
          </p>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">{t('vix_strategy.metrics.spx_price')}</p>
              <p className="text-2xl font-bold text-gray-900">
                {options_analysis.spx_price.toFixed(2)}
              </p>
            </div>
            <Target className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">{t('vix_strategy.metrics.straddle_cost')}</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatCurrency(options_analysis.straddle.total_cost)}
              </p>
            </div>
            <Activity className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">{t('vix_strategy.metrics.breakeven_move')}</p>
              <p className="text-2xl font-bold text-gray-900">
                ±{options_analysis.straddle.breakeven_move_pct.toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-orange-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">{t('vix_strategy.metrics.days_to_expiry')}</p>
              <p className="text-2xl font-bold text-gray-900">
                {options_analysis.straddle.days_to_expiry}
              </p>
            </div>
            <AlertTriangle className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* VIX Analysis Chart */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">{t('vix_strategy.charts.vix_analysis')}</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">{t('vix_strategy.charts.current_vix')}</span>
              <span className="font-semibold">{vix_analysis.current_vix.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">{t('vix_strategy.charts.sma_24m')}</span>
              <span className="font-semibold">{vix_analysis.vix_sma.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">{t('vix_strategy.charts.z_score')}</span>
              <span className={`font-semibold ${vix_analysis.z_score > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {vix_analysis.z_score.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">{t('vix_strategy.charts.vix_percentile')}</span>
              <span className="font-semibold">{vix_analysis.vix_percentile.toFixed(1)}%</span>
            </div>

            {/* Z-Score Visual Indicator */}
            <div className="mt-4">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>{t('vix_strategy.charts.low_vix')}</span>
                <span>{t('vix_strategy.charts.normal')}</span>
                <span>{t('vix_strategy.charts.high_vix')}</span>
              </div>
              <div className="relative h-4 bg-gradient-to-r from-green-200 via-yellow-200 to-red-200 rounded">
                <div
                  className="absolute top-0 w-2 h-4 bg-gray-800 rounded"
                  style={{
                    left: `${Math.max(0, Math.min(100, (vix_analysis.z_score + 2) / 4 * 100))}%`,
                    transform: 'translateX(-50%)'
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Profit/Loss Chart */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">{t('vix_strategy.charts.pl_chart_title')}</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={profitLossData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="price"
                tickFormatter={(value) => value.toFixed(0)}
              />
              <YAxis
                tickFormatter={(value) => `$${value.toFixed(0)}`}
              />
              <Tooltip
                formatter={(value: number) => [`$${value.toFixed(2)}`, 'P&L']}
                labelFormatter={(value) => `${t('vix_strategy.charts.spx_price_label')}: $${value.toFixed(2)}`}
              />
              <Line
                type="monotone"
                dataKey="profit"
                stroke="#4f46e5"
                strokeWidth={2}
                dot={false}
              />
              <ReferenceLine
                y={0}
                stroke="#ef4444"
                strokeDasharray="5 5"
                label={{
                  value: t('vix_strategy.summary.lower_be') + '/' + t('vix_strategy.summary.upper_be'),
                  position: 'insideBottomRight',
                  fill: '#ef4444',
                  fontSize: 10
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Options Details */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{t('vix_strategy.options.title')}</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Call Option */}
          <div className="border border-gray-200 rounded-lg p-4">
            <h4 className="font-semibold text-green-600 mb-3">{t('vix_strategy.options.call')}</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>{t('vix_strategy.options.strike')}:</span>
                <span className="font-semibold">{formatCurrency(options_analysis.strike)}</span>
              </div>
              <div className="flex justify-between">
                <span>{t('vix_strategy.options.last_price')}:</span>
                <span className="font-semibold">{formatCurrency(options_analysis.call_data.price)}</span>
              </div>
              <div className="flex justify-between">
                <span>{t('vix_strategy.options.bid_ask')}:</span>
                <span className="font-semibold">
                  {formatCurrency(options_analysis.call_data.bid)} / {formatCurrency(options_analysis.call_data.ask)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>{t('vix_strategy.options.volume')}:</span>
                <span className="font-semibold">{options_analysis.call_data.volume.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>{t('vix_strategy.options.iv')}:</span>
                <span className="font-semibold">{(options_analysis.call_data.iv * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Put Option */}
          <div className="border border-gray-200 rounded-lg p-4">
            <h4 className="font-semibold text-red-600 mb-3">{t('vix_strategy.options.put')}</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>{t('vix_strategy.options.strike')}:</span>
                <span className="font-semibold">{formatCurrency(options_analysis.strike)}</span>
              </div>
              <div className="flex justify-between">
                <span>{t('vix_strategy.options.last_price')}:</span>
                <span className="font-semibold">{formatCurrency(options_analysis.put_data.price)}</span>
              </div>
              <div className="flex justify-between">
                <span>{t('vix_strategy.options.bid_ask')}:</span>
                <span className="font-semibold">
                  {formatCurrency(options_analysis.put_data.bid)} / {formatCurrency(options_analysis.put_data.ask)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>{t('vix_strategy.options.volume')}:</span>
                <span className="font-semibold">{options_analysis.put_data.volume.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>{t('vix_strategy.options.iv')}:</span>
                <span className="font-semibold">{(options_analysis.put_data.iv * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Strategy Summary */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-semibold text-gray-900 mb-3">{t('vix_strategy.summary.title')}</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">{t('vix_strategy.summary.max_loss')}:</span>
              <div className="font-semibold text-red-600">
                {formatCurrency(options_analysis.straddle.max_loss_long)}
              </div>
            </div>
            <div>
              <span className="text-gray-600">{t('vix_strategy.summary.target_profit')}:</span>
              <div className="font-semibold text-green-600">
                {formatCurrency(options_analysis.straddle.target_profit_long)}
              </div>
            </div>
            <div>
              <span className="text-gray-600">{t('vix_strategy.summary.upper_be')}:</span>
              <div className="font-semibold">
                {formatCurrency(options_analysis.straddle.upper_breakeven)}
              </div>
            </div>
            <div>
              <span className="text-gray-600">{t('vix_strategy.summary.lower_be')}:</span>
              <div className="font-semibold">
                {formatCurrency(options_analysis.straddle.lower_breakeven)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VixStrategy;