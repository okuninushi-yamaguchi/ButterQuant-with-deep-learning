import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, Area, AreaChart, BarChart, Bar, ScatterChart, Scatter } from 'recharts';
import { TrendingUp, Zap, Target, AlertTriangle, Activity } from 'lucide-react';

const ButterflyTheoryDemo = () => {
  const [activeSection, setActiveSection] = useState('payoff');
  const [bsParams, setBsParams] = useState({
    S: 100,
    K: 100,
    T: 0.5,
    r: 0.05,
    sigma: 0.25
  });
  const [animateBS, setAnimateBS] = useState(false);
  
  // Section 1: Butterfly Payoff Diagram
  const generateButterflyPayoff = () => {
    const K1 = 470, K2 = 480, K3 = 490;
    const premium = 1.2;
    const data = [];
    
    for (let price = 450; price <= 510; price += 2) {
      const payoff = Math.max(0, price - K1) - 2 * Math.max(0, price - K2) + Math.max(0, price - K3) - premium;
      data.push({
        price,
        payoff: payoff,
        breakeven: price === K1 + premium || price === K3 - premium
      });
    }
    return data;
  };

  // Section 2: IV Skew
  const generateIVSkew = () => {
    return [
      { strike: 450, moneyness: 95, iv: 28, type: 'OTM Put' },
      { strike: 460, moneyness: 97, iv: 26.5, type: 'OTM Put' },
      { strike: 470, moneyness: 99, iv: 25.5, type: 'ITM' },
      { strike: 475, moneyness: 100, iv: 25, type: 'ATM' },
      { strike: 480, moneyness: 101, iv: 24.8, type: 'ATM' },
      { strike: 490, moneyness: 103, iv: 24, type: 'OTM Call' },
      { strike: 500, moneyness: 105, iv: 23, type: 'OTM Call' },
    ];
  };

  // Section 3: Fourier Transform (Price vs Detrended)
  const generateFourierData = () => {
    const days = 60;
    const data = [];
    let price = 475;
    
    for (let i = 0; i < days; i++) {
      const trend = i * 0.15;
      const cycle = 8 * Math.sin(2 * Math.PI * i / 28);
      const noise = (Math.random() - 0.5) * 3;
      
      price = 475 + trend + cycle + noise;
      const vwap = 475 + trend;
      
      data.push({
        day: i,
        price: price,
        vwap: vwap,
        detrended: price - vwap
      });
    }
    return data;
  };

  // Section 4: Scoring System
  const generateScoringDemo = (arimaMatch = 95, volPremium = 85, stability = 90, fourier = 100) => {
    return [
      { factor: 'ARIMA匹配', weight: 0.35, score: arimaMatch, contribution: arimaMatch * 0.35 },
      { factor: '波动率溢价', weight: 0.30, score: volPremium, contribution: volPremium * 0.30 },
      { factor: '价格稳定性', weight: 0.20, score: stability, contribution: stability * 0.20 },
      { factor: '傅立叶对齐', weight: 0.15, score: fourier, contribution: fourier * 0.15 },
    ];
  };

  // Section 5: Greeks Evolution
  const generateGreeksData = () => {
    const data = [];
    for (let price = 460; price <= 500; price += 2) {
      const K2 = 480;
      const delta = price < K2 ? (price - 460) * 0.02 : (500 - price) * 0.02;
      const gamma = Math.exp(-Math.pow((price - K2) / 15, 2)) * 0.15;
      const vega = -0.8 + Math.abs(price - K2) * 0.02;
      
      data.push({ price, delta, gamma, vega });
    }
    return data;
  };

  // Section 6: Black-Scholes Formula Animation
  const calculateBS = (S, K, T, r, sigma) => {
    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);
    
    const normCDF = (x) => {
      const t = 1 / (1 + 0.2316419 * Math.abs(x));
      const d = 0.3989423 * Math.exp(-x * x / 2);
      const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
      return x > 0 ? 1 - p : p;
    };
    
    const Nd1 = normCDF(d1);
    const Nd2 = normCDF(d2);
    
    const callPrice = S * Nd1 - K * Math.exp(-r * T) * Nd2;
    
    return { callPrice, d1, d2, Nd1, Nd2 };
  };

  const generateBSSurface = () => {
    const data = [];
    for (let S = 80; S <= 120; S += 4) {
      for (let sigma = 0.1; sigma <= 0.4; sigma += 0.03) {
        const { callPrice } = calculateBS(S, 100, 0.5, 0.05, sigma);
        data.push({
          S,
          sigma: sigma * 100,
          price: callPrice
        });
      }
    }
    return data;
  };

  // Section 7: GARCH Volatility Clustering
  const generateGARCHData = () => {
    const days = 250;
    const data = [];
    let price = 100;
    let vol = 0.02;
    
    // GARCH(1,1) parameters
    const omega = 0.000001;
    const alpha = 0.08;
    const beta = 0.90;
    
    for (let i = 0; i < days; i++) {
      const shock = (Math.random() - 0.5) * 2;
      const return_t = vol * shock;
      
      price = price * (1 + return_t);
      
      // GARCH variance equation
      vol = Math.sqrt(omega + alpha * (return_t * return_t) + beta * (vol * vol));
      
      // Add some volatility clustering events
      if (i > 50 && i < 70) vol *= 1.5;
      if (i > 150 && i < 180) vol *= 2.0;
      
      data.push({
        day: i,
        price: price,
        volatility: vol * 100,
        returns: return_t * 100,
        regime: vol > 0.03 ? 'High Vol' : 'Low Vol'
      });
    }
    return data;
  };

  const butterflyData = useMemo(() => generateButterflyPayoff(), []);
  const ivSkewData = useMemo(() => generateIVSkew(), []);
  const fourierData = useMemo(() => generateFourierData(), []);
  const [scoringData, setScoringData] = useState(generateScoringDemo());
  const greeksData = useMemo(() => generateGreeksData(), []);
  const bsSurfaceData = useMemo(() => generateBSSurface(), []);
  const garchData = useMemo(() => generateGARCHData(), []);

  const totalScore = scoringData.reduce((sum, item) => sum + item.contribution, 0);
  const bsResult = calculateBS(bsParams.S, bsParams.K, bsParams.T, bsParams.r, bsParams.sigma);

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="bg-white rounded-2xl shadow-2xl p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-800 mb-3">
            🦋 期权蝴蝶策略量化模型
          </h1>
          <p className="text-slate-600 text-lg">理论基础与核心机制可视化演示</p>
        </div>

        {/* Navigation */}
        <div className="flex gap-2 mb-8 overflow-x-auto pb-2">
          {[
            { id: 'payoff', label: '盈亏结构', icon: Target },
            { id: 'ivskew', label: 'IV Skew', icon: TrendingUp },
            { id: 'fourier', label: '傅立叶分析', icon: Activity },
            { id: 'bsformula', label: 'BS公式', icon: Zap },
            { id: 'garch', label: 'GARCH聚类', icon: AlertTriangle },
            { id: 'scoring', label: '评分系统', icon: Zap },
            { id: 'greeks', label: 'Greeks风控', icon: AlertTriangle }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveSection(id)}
              className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${
                activeSection === id
                  ? 'bg-blue-600 text-white shadow-lg scale-105'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              <Icon size={20} />
              {label}
            </button>
          ))}
        </div>

        {/* Content Sections */}
        <div className="bg-slate-50 rounded-xl p-6">
          {/* Section 1: Butterfly Payoff */}
          {activeSection === 'payoff' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">📈 蝴蝶策略盈亏结构</h2>
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                    <div className="text-green-700 font-bold">买入 1x Call</div>
                    <div className="text-2xl font-bold text-green-800">K₁ = $470</div>
                  </div>
                  <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
                    <div className="text-red-700 font-bold">卖出 2x Call</div>
                    <div className="text-2xl font-bold text-red-800">K₂ = $480</div>
                  </div>
                  <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                    <div className="text-green-700 font-bold">买入 1x Call</div>
                    <div className="text-2xl font-bold text-green-800">K₃ = $490</div>
                  </div>
                </div>
                
                <LineChart width={900} height={400} data={butterflyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis 
                    dataKey="price" 
                    label={{ value: '到期价格 ($)', position: 'insideBottom', offset: -5 }}
                    stroke="#64748b"
                  />
                  <YAxis 
                    label={{ value: '盈亏 ($)', angle: -90, position: 'insideLeft' }}
                    stroke="#64748b"
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }}
                    formatter={(value) => [`$${value.toFixed(2)}`, '盈亏']}
                  />
                  <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="5 5" />
                  <ReferenceLine x={470} stroke="#22c55e" strokeDasharray="3 3" label="K₁" />
                  <ReferenceLine x={480} stroke="#ef4444" strokeDasharray="3 3" label="K₂" />
                  <ReferenceLine x={490} stroke="#22c55e" strokeDasharray="3 3" label="K₃" />
                  <Line type="monotone" dataKey="payoff" stroke="#3b82f6" strokeWidth={3} dot={false} />
                </LineChart>
                
                <div className="mt-6 bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    💡 <strong>核心特征：</strong> 最大收益在价格 = K₂ ($480) 时实现，盈亏结构呈"山峰"状，
                    适合预期价格在区间内小幅波动的市场环境。
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Section 2: IV Skew */}
          {activeSection === 'ivskew' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">📊 隐含波动率偏斜 (IV Skew)</h2>
                
                <LineChart width={900} height={400} data={ivSkewData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis 
                    dataKey="strike" 
                    label={{ value: '行权价 ($)', position: 'insideBottom', offset: -5 }}
                    stroke="#64748b"
                  />
                  <YAxis 
                    label={{ value: '隐含波动率 (%)', angle: -90, position: 'insideLeft' }}
                    domain={[22, 29]}
                    stroke="#64748b"
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }}
                    formatter={(value, name) => [
                      name === 'iv' ? `${value}%` : value,
                      name === 'iv' ? 'IV' : name
                    ]}
                  />
                  <ReferenceLine y={25} stroke="#94a3b8" strokeDasharray="5 5" label="ATM IV" />
                  <Line type="monotone" dataKey="iv" stroke="#8b5cf6" strokeWidth={3} dot={{ fill: '#8b5cf6', r: 6 }} />
                </LineChart>

                <div className="grid grid-cols-3 gap-4 mt-6">
                  <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
                    <div className="text-purple-700 font-bold text-sm">OTM Put (K=$450)</div>
                    <div className="text-3xl font-bold text-purple-800">28%</div>
                    <div className="text-sm text-purple-600 mt-1">保护需求高 → IV升高</div>
                  </div>
                  <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
                    <div className="text-blue-700 font-bold text-sm">ATM (K=$475)</div>
                    <div className="text-3xl font-bold text-blue-800">25%</div>
                    <div className="text-sm text-blue-600 mt-1">基准波动率</div>
                  </div>
                  <div className="bg-indigo-50 border-2 border-indigo-300 rounded-lg p-4">
                    <div className="text-indigo-700 font-bold text-sm">OTM Call (K=$500)</div>
                    <div className="text-3xl font-bold text-indigo-800">23%</div>
                    <div className="text-sm text-indigo-600 mt-1">投机需求弱 → IV降低</div>
                  </div>
                </div>

                <div className="mt-6 bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    ⚠️ <strong>关键洞察：</strong> 如果对所有行权价使用相同的波动率（如 σ=25%），
                    会导致蝴蝶策略定价误差高达 <span className="text-red-600 font-bold">20%</span>！
                    必须根据钱性 (Moneyness) 调整每个期权的波动率。
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Section 3: Fourier Analysis */}
          {activeSection === 'fourier' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">🌊 傅立叶分析：去趋势的重要性</h2>
                
                <div className="mb-6">
                  <h3 className="text-lg font-bold text-slate-700 mb-3">原始价格序列 vs VWAP</h3>
                  <LineChart width={900} height={300} data={fourierData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="day" label={{ value: '交易日', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: '价格 ($)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
                    <Line type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={2} name="实际价格" dot={false} />
                    <Line type="monotone" dataKey="vwap" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" name="VWAP (趋势)" dot={false} />
                  </LineChart>
                </div>

                <div className="mb-6">
                  <h3 className="text-lg font-bold text-slate-700 mb-3">去趋势后的信号（可用于FFT）</h3>
                  <AreaChart width={900} height={300} data={fourierData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="day" label={{ value: '交易日', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: '偏移 ($)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
                    <ReferenceLine y={0} stroke="#94a3b8" />
                    <Area type="monotone" dataKey="detrended" stroke="#10b981" fill="#d1fae5" fillOpacity={0.6} />
                  </AreaChart>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
                    <div className="text-red-700 font-bold mb-2">❌ 错误做法</div>
                    <ul className="text-sm text-slate-700 space-y-1">
                      <li>• 直接对价格做FFT</li>
                      <li>• 趋势会产生虚假低频能量</li>
                      <li>• 无法区分真实周期 vs 随机游走</li>
                    </ul>
                  </div>
                  <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                    <div className="text-green-700 font-bold mb-2">✅ 正确做法</div>
                    <ul className="text-sm text-slate-700 space-y-1">
                      <li>• 使用 Price - VWAP 去趋势</li>
                      <li>• 或使用对数收益率 log(P(t)/P(t-1))</li>
                      <li>• 得到平稳信号后再做FFT</li>
                    </ul>
                  </div>
                </div>

                <div className="mt-6 bg-cyan-50 border-l-4 border-cyan-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    🎯 <strong>实战应用：</strong> 傅立叶不是用来"预测"价格，而是用来"侦测"市场节奏。
                    如果检测到 28 天周期，就选择 DTE=30 天的期权，让到期日对齐周期峰值。
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Section 4: Black-Scholes Formula Animation */}
          {activeSection === 'bsformula' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">📐 Black-Scholes 公式动态演示</h2>
                
                {/* Formula Display */}
                <div className="bg-gradient-to-r from-indigo-900 to-purple-900 text-white rounded-xl p-6 mb-6">
                  <div className="text-center mb-4">
                    <div className="text-2xl font-bold mb-2">Call Option Price</div>
                    <div className="text-4xl font-mono">C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)</div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
                    <div className="bg-white/10 rounded-lg p-3">
                      <div className="font-mono">d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)</div>
                      <div className="mt-2 text-blue-300">d₁ = {bsResult.d1.toFixed(4)}</div>
                    </div>
                    <div className="bg-white/10 rounded-lg p-3">
                      <div className="font-mono">d₂ = d₁ - σ√T</div>
                      <div className="mt-2 text-purple-300">d₂ = {bsResult.d2.toFixed(4)}</div>
                    </div>
                  </div>
                </div>

                {/* Interactive Parameter Controls */}
                <div className="grid grid-cols-2 gap-6 mb-6">
                  <div>
                    <h3 className="text-lg font-bold text-slate-700 mb-4">调整参数观察价格变化</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="text-sm font-medium text-slate-600 flex justify-between">
                          <span>标的价格 (S)</span>
                          <span className="text-blue-600 font-bold">${bsParams.S}</span>
                        </label>
                        <input
                          type="range"
                          min="80"
                          max="120"
                          value={bsParams.S}
                          onChange={(e) => setBsParams({...bsParams, S: Number(e.target.value)})}
                          className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
                        />
                      </div>
                      
                      <div>
                        <label className="text-sm font-medium text-slate-600 flex justify-between">
                          <span>行权价 (K)</span>
                          <span className="text-green-600 font-bold">${bsParams.K}</span>
                        </label>
                        <input
                          type="range"
                          min="80"
                          max="120"
                          value={bsParams.K}
                          onChange={(e) => setBsParams({...bsParams, K: Number(e.target.value)})}
                          className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer"
                        />
                      </div>
                      
                      <div>
                        <label className="text-sm font-medium text-slate-600 flex justify-between">
                          <span>到期时间 (T) 年</span>
                          <span className="text-purple-600 font-bold">{bsParams.T.toFixed(2)}</span>
                        </label>
                        <input
                          type="range"
                          min="0.1"
                          max="2"
                          step="0.1"
                          value={bsParams.T}
                          onChange={(e) => setBsParams({...bsParams, T: Number(e.target.value)})}
                          className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                        />
                      </div>
                      
                      <div>
                        <label className="text-sm font-medium text-slate-600 flex justify-between">
                          <span>波动率 (σ) %</span>
                          <span className="text-red-600 font-bold">{(bsParams.sigma * 100).toFixed(0)}%</span>
                        </label>
                        <input
                          type="range"
                          min="0.1"
                          max="0.5"
                          step="0.01"
                          value={bsParams.sigma}
                          onChange={(e) => setBsParams({...bsParams, sigma: Number(e.target.value)})}
                          className="w-full h-2 bg-red-200 rounded-lg appearance-none cursor-pointer"
                        />
                      </div>
                      
                      <div>
                        <label className="text-sm font-medium text-slate-600 flex justify-between">
                          <span>无风险利率 (r) %</span>
                          <span className="text-amber-600 font-bold">{(bsParams.r * 100).toFixed(1)}%</span>
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="0.1"
                          step="0.01"
                          value={bsParams.r}
                          onChange={(e) => setBsParams({...bsParams, r: Number(e.target.value)})}
                          className="w-full h-2 bg-amber-200 rounded-lg appearance-none cursor-pointer"
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-bold text-slate-700 mb-4">计算结果</h3>
                    <div className="space-y-3">
                      <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl p-6">
                        <div className="text-sm opacity-90">Call 期权价格</div>
                        <div className="text-5xl font-bold">${bsResult.callPrice.toFixed(2)}</div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-3">
                          <div className="text-xs text-slate-600">N(d₁)</div>
                          <div className="text-2xl font-bold text-blue-700">{bsResult.Nd1.toFixed(4)}</div>
                          <div className="text-xs text-slate-500 mt-1">Delta 近似值</div>
                        </div>
                        <div className="bg-purple-50 border-2 border-purple-200 rounded-lg p-3">
                          <div className="text-xs text-slate-600">N(d₂)</div>
                          <div className="text-2xl font-bold text-purple-700">{bsResult.Nd2.toFixed(4)}</div>
                          <div className="text-xs text-slate-500 mt-1">行权概率</div>
                        </div>
                      </div>
                      
                      <div className="bg-slate-50 border-2 border-slate-200 rounded-lg p-3">
                        <div className="text-xs text-slate-600 mb-2">钱性 (Moneyness)</div>
                        <div className="text-lg font-bold text-slate-800">
                          {bsParams.S / bsParams.K > 1.05 ? '💰 ITM (实值)' : 
                           bsParams.S / bsParams.K < 0.95 ? '📉 OTM (虚值)' : 
                           '🎯 ATM (平值)'}
                        </div>
                        <div className="text-sm text-slate-600 mt-1">
                          S/K = {(bsParams.S / bsParams.K).toFixed(3)}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 3D Surface Visualization */}
                <div className="mt-6">
                  <h3 className="text-lg font-bold text-slate-700 mb-4">期权价格曲面（标的价格 × 波动率）</h3>
                  <ScatterChart width={900} height={400} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis 
                      type="number"
                      dataKey="S" 
                      name="标的价格"
                      label={{ value: '标的价格 ($)', position: 'insideBottom', offset: -5 }}
                      domain={[80, 120]}
                    />
                    <YAxis 
                      type="number"
                      dataKey="sigma" 
                      name="波动率"
                      label={{ value: '波动率 (%)', angle: -90, position: 'insideLeft' }}
                      domain={[10, 40]}
                    />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }}
                      formatter={(value, name) => {
                        if (name === 'price') return [`${value.toFixed(2)}`, 'Call价格'];
                        return value;
                      }}
                    />
                    <Scatter 
                      data={bsSurfaceData} 
                      fill="#8884d8"
                    >
                      {bsSurfaceData.map((entry, index) => (
                        <circle 
                          key={index}
                          r={entry.price / 2}
                          fill={entry.price > 15 ? '#ef4444' : entry.price > 10 ? '#f59e0b' : entry.price > 5 ? '#3b82f6' : '#10b981'}
                          opacity={0.6}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </div>

                <div className="mt-6 bg-indigo-50 border-l-4 border-indigo-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    🔬 <strong>核心洞察：</strong> 期权价格对波动率（σ）极其敏感。
                    波动率从 20% 提升到 30%，期权价格可能上涨 <span className="text-indigo-600 font-bold">50%+</span>。
                    这就是为什么必须考虑 IV Skew，不同行权价的波动率不同！
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Section 5: GARCH Volatility Clustering */}
          {activeSection === 'garch' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">📊 GARCH 波动率聚类效应</h2>
                
                {/* Price Chart */}
                <div className="mb-6">
                  <h3 className="text-lg font-bold text-slate-700 mb-3">价格走势</h3>
                  <LineChart width={900} height={250} data={garchData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis 
                      dataKey="day" 
                      label={{ value: '交易日', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: '价格 ($)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
                    <Line type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  </LineChart>
                </div>

                {/* Volatility Chart with Clustering */}
                <div className="mb-6">
                  <h3 className="text-lg font-bold text-slate-700 mb-3">波动率聚类（GARCH 预测）</h3>
                  <AreaChart width={900} height={300} data={garchData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis 
                      dataKey="day" 
                      label={{ value: '交易日', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: '波动率 (%)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
                    <Area 
                      type="monotone" 
                      dataKey="volatility" 
                      stroke="#ef4444" 
                      fill="#fee2e2" 
                      fillOpacity={0.6}
                      strokeWidth={2}
                    />
                    <ReferenceLine y={3} stroke="#94a3b8" strokeDasharray="5 5" label="阈值" />
                  </AreaChart>
                </div>

                {/* Returns Distribution */}
                <div className="mb-6">
                  <h3 className="text-lg font-bold text-slate-700 mb-3">收益率分布（展示厚尾效应）</h3>
                  <BarChart width={900} height={300} data={garchData.slice(0, 100)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis 
                      dataKey="day" 
                      label={{ value: '时间', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: '日收益率 (%)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
                    <Bar dataKey="returns" radius={[4, 4, 0, 0]}>
                      {garchData.slice(0, 100).map((entry, index) => (
                        <cell key={index} fill={entry.regime === 'High Vol' ? '#ef4444' : '#3b82f6'} />
                      ))}
                    </Bar>
                  </BarChart>
                </div>

                {/* GARCH Model Explanation */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="bg-gradient-to-br from-red-50 to-orange-50 border-2 border-red-300 rounded-lg p-5">
                    <h4 className="text-red-700 font-bold text-lg mb-3">🔥 波动率聚类现象</h4>
                    <ul className="text-sm text-slate-700 space-y-2">
                      <li>• <strong>大波动后继续大波动</strong></li>
                      <li>• <strong>小波动后继续小波动</strong></li>
                      <li>• 这违反了 BS 模型的"恒定波动率"假设</li>
                      <li>• 市场存在"波动率状态"切换</li>
                    </ul>
                  </div>
                  
                  <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-300 rounded-lg p-5">
                    <h4 className="text-blue-700 font-bold text-lg mb-3">📐 GARCH(1,1) 模型</h4>
                    <div className="text-sm text-slate-700 space-y-2">
                      <div className="font-mono bg-white p-2 rounded">σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁</div>
                      <div className="mt-2">
                        <div>• <strong>ω</strong> = 长期均值水平</div>
                        <div>• <strong>α</strong> = 昨日冲击的影响</div>
                        <div>• <strong>β</strong> = 波动率持续性</div>
                      </div>
                      <div className="mt-2 text-xs text-slate-500">
                        典型值: α≈0.08, β≈0.90 → 高持续性
                      </div>
                    </div>
                  </div>
                </div>

                {/* Practical Implications */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                    <div className="text-green-700 font-bold mb-2">✅ 策略应用</div>
                    <ul className="text-sm text-slate-700 space-y-1">
                      <li>• 低波动期：卖出期权</li>
                      <li>• 高波动期：避免新仓位</li>
                      <li>• 波动率均值回归</li>
                    </ul>
                  </div>
                  
                  <div className="bg-yellow-50 border-2 border-yellow-300 rounded-lg p-4">
                    <div className="text-yellow-700 font-bold mb-2">⚠️ 风险识别</div>
                    <ul className="text-sm text-slate-700 space-y-1">
                      <li>• GARCH 预测 σ↑30%</li>
                      <li>• 立即提高止损阈值</li>
                      <li>• 减少仓位规模</li>
                    </ul>
                  </div>
                  
                  <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
                    <div className="text-purple-700 font-bold mb-2">📈 入场时机</div>
                    <ul className="text-sm text-slate-700 space-y-1">
                      <li>• IV 高位 + GARCH 预测回落</li>
                      <li>• 卖出蝴蝶获利概率高</li>
                      <li>• 波动率溢价最大化</li>
                    </ul>
                  </div>
                </div>

                <div className="mt-6 bg-red-50 border-l-4 border-red-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    💡 <strong>核心价值：</strong> GARCH 不是用来"预测价格"，而是用来<strong>预测波动率状态</strong>。
                    在模型中占 <span className="text-red-600 font-bold">30% 权重</span>，
                    帮助识别 "当前 IV 是否被高估/低估" 以及 "未来波动率会上升还是下降"。
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Section 6: Scoring System */}
          {activeSection === 'scoring' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">⚡ 多因子评分系统</h2>
                
                <BarChart width={900} height={350} data={scoringData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="factor" stroke="#64748b" />
                  <YAxis label={{ value: '贡献分数', angle: -90, position: 'insideLeft' }} stroke="#64748b" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }}
                    formatter={(value, name) => {
                      if (name === 'score') return [value.toFixed(1), '原始分数'];
                      if (name === 'contribution') return [value.toFixed(1), '加权贡献'];
                      return value;
                    }}
                  />
                  <Bar dataKey="contribution" fill="#3b82f6" radius={[8, 8, 0, 0]} />
                </BarChart>

                <div className="mt-6 grid grid-cols-4 gap-4">
                  {scoringData.map((item, idx) => (
                    <div key={idx} className="bg-slate-50 border-2 border-slate-200 rounded-lg p-3">
                      <div className="text-slate-600 text-xs font-medium mb-1">{item.factor}</div>
                      <div className="text-2xl font-bold text-slate-800">{item.score}</div>
                      <div className="text-xs text-slate-500 mt-1">权重: {(item.weight * 100).toFixed(0)}%</div>
                    </div>
                  ))}
                </div>

                <div className="mt-6 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl p-6 shadow-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium opacity-90">综合评分</div>
                      <div className="text-5xl font-bold">{totalScore.toFixed(1)}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold">
                        {totalScore >= 75 ? '🚀 STRONG BUY' : totalScore >= 60 ? '✅ BUY' : totalScore >= 45 ? '⚖️ NEUTRAL' : '🚫 AVOID'}
                      </div>
                      <div className="text-sm opacity-90 mt-1">
                        建议仓位: {totalScore >= 75 ? '3-5%' : totalScore >= 60 ? '2-3%' : totalScore >= 45 ? '1-2%' : '0%'}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    🧮 <strong>评分公式：</strong> Score = 35%×ARIMA匹配 + 30%×波动率溢价 + 20%×价格稳定性 + 15%×傅立叶对齐 - Delta惩罚
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Section 7: Greeks */}
          {activeSection === 'greeks' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">🛡️ Greeks 风险管理</h2>
                
                <LineChart width={900} height={400} data={greeksData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis 
                    dataKey="price" 
                    label={{ value: '标的价格 ($)', position: 'insideBottom', offset: -5 }}
                    stroke="#64748b"
                  />
                  <YAxis 
                    label={{ value: 'Greeks 值', angle: -90, position: 'insideLeft' }}
                    stroke="#64748b"
                  />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
                  <Legend />
                  <ReferenceLine x={480} stroke="#94a3b8" strokeDasharray="5 5" label="K₂" />
                  <Line type="monotone" dataKey="delta" stroke="#3b82f6" strokeWidth={2} name="Delta (方向风险)" />
                  <Line type="monotone" dataKey="gamma" stroke="#10b981" strokeWidth={2} name="Gamma (加速度)" />
                  <Line type="monotone" dataKey="vega" stroke="#ef4444" strokeWidth={2} name="Vega (波动率风险)" />
                </LineChart>

                <div className="grid grid-cols-2 gap-4 mt-6">
                  <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
                    <div className="text-blue-700 font-bold mb-2">Delta ≈ 0（方向中性）</div>
                    <div className="text-sm text-slate-700">
                      理想蝴蝶的 |Delta| &lt; 0.10，不受小幅价格波动影响。
                      如果 |Delta| &gt; 0.15，说明存在方向性赌注，需要扣分。
                    </div>
                  </div>
                  <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                    <div className="text-green-700 font-bold mb-2">Gamma &gt; 0（在中心）</div>
                    <div className="text-sm text-slate-700">
                      价格接近 K₂ 时 Gamma 为正，意味着获利加速。
                      但远离 K₂ 时 Gamma 为负，风险加大。
                    </div>
                  </div>
                  <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
                    <div className="text-red-700 font-bold mb-2">Vega &lt; 0（做空波动率）</div>
                    <div className="text-sm text-slate-700">
                      蝴蝶做空波动率，IV 下降时获利。
                      适合在 IV 高位（&gt;75 百分位）入场。
                    </div>
                  </div>
                  <div className="bg-amber-50 border-2 border-amber-300 rounded-lg p-4">
                    <div className="text-amber-700 font-bold mb-2">Theta &gt; 0（时间是朋友）</div>
                    <div className="text-sm text-slate-700">
                      每天赚取时间价值衰减，典型 +$0.05~$0.15/天。
                      适合盘整市场，避免大幅波动。
                    </div>
                  </div>
                </div>

                <div className="mt-6 bg-red-50 border-l-4 border-red-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    ⚠️ <strong>风险控制：</strong> 如果 |Delta| &gt; 0.15 或 IV 百分位 &lt; 50%，
                    即使其他因子评分高，也要降低推荐等级或拒绝交易。
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer Summary */}
        <div className="mt-8 bg-gradient-to-r from-slate-800 to-slate-900 text-white rounded-xl p-6">
          <h3 className="text-xl font-bold mb-3">📚 核心理论总结</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="font-semibold text-blue-300 mb-1">✓ 市场价格 &gt; 理论价格</div>
              <div className="text-slate-300">市场价格反映真实供需，BS 模型只是参考锚点</div>
            </div>
            <div>
              <div className="font-semibold text-purple-300 mb-1">✓ IV Skew 不可忽视</div>
              <div className="text-slate-300">不同行权价必须使用不同波动率，否则误差 20%+</div>
            </div>
            <div>
              <div className="font-semibold text-green-300 mb-1">✓ 傅立叶必须去趋势</div>
              <div className="text-slate-300">直接对价格 FFT 是错误的，要用 Price-VWAP 或收益率</div>
            </div>
            <div>
              <div className="font-semibold text-amber-300 mb-1">✓ Greeks 约束核心风险</div>
              <div className="text-slate-300">Delta 中性、Vega 负值、Theta 正值是蝴蝶策略的特征</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ButterflyTheoryDemo;