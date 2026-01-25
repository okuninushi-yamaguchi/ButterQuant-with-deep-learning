import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, BarChart, Bar } from 'recharts';
import { TrendingUp, Activity, RefreshCw } from 'lucide-react';

const ARIMAMathDemo = () => {
  const [arimaParams, setArimaParams] = useState({
    p: 1,
    d: 1,
    q: 1
  });
  
  const [showStep, setShowStep] = useState('original');

  const generateOriginalData = () => {
    const data = [];
    let price = 100;
    const trend = 0.05;
    
    for (let t = 0; t < 100; t++) {
      const seasonality = 3 * Math.sin(2 * Math.PI * t / 20);
      const noise = (Math.random() - 0.5) * 2;
      price = price * (1 + trend/100) + seasonality + noise;
      
      data.push({
        time: t,
        price: price,
        trend: 100 * Math.exp(trend * t / 100)
      });
    }
    return data;
  };

  const applyDifferencing = (data, order) => {
    const result = [];
    for (let i = order; i < data.length; i++) {
      const diff = data[i].price - data[i - order].price;
      result.push({
        time: data[i].time,
        original: data[i].price,
        differenced: diff
      });
    }
    return result;
  };

  const generateARProcess = (phi) => {
    const data = [];
    let y = 0;
    
    for (let t = 0; t < 100; t++) {
      const noise = (Math.random() - 0.5) * 2;
      y = phi * y + noise;
      data.push({ time: t, value: y });
    }
    return data;
  };

  const generateMAProcess = (theta) => {
    const data = [];
    let prevNoise = 0;
    
    for (let t = 0; t < 100; t++) {
      const noise = (Math.random() - 0.5) * 2;
      const y = noise + theta * prevNoise;
      prevNoise = noise;
      data.push({ time: t, value: y });
    }
    return data;
  };

  const computeACF = (data, maxLag = 20) => {
    const mean = data.reduce((sum, d) => sum + d.value, 0) / data.length;
    const variance = data.reduce((sum, d) => sum + Math.pow(d.value - mean, 2), 0) / data.length;
    
    const acf = [];
    for (let lag = 0; lag <= maxLag; lag++) {
      let sum = 0;
      for (let i = lag; i < data.length; i++) {
        sum += (data[i].value - mean) * (data[i - lag].value - mean);
      }
      const correlation = sum / (data.length * variance);
      acf.push({
        lag,
        acf: correlation,
        significant: Math.abs(correlation) > 1.96 / Math.sqrt(data.length)
      });
    }
    return acf;
  };

  const originalData = useMemo(() => generateOriginalData(), []);
  const differencedData = useMemo(() => applyDifferencing(originalData, 1), [originalData]);
  const arData = useMemo(() => generateARProcess(0.7), []);
  const maData = useMemo(() => generateMAProcess(0.5), []);
  const acfData = useMemo(() => computeACF(differencedData.map(d => ({ value: d.differenced }))), [differencedData]);

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50">
      <div className="bg-white rounded-2xl shadow-2xl p-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-800 mb-3">
            📊 ARIMA 数学模型详解
          </h1>
          <p className="text-slate-600 text-lg">AutoRegressive Integrated Moving Average</p>
        </div>

        <div className="bg-gradient-to-r from-blue-900 to-indigo-900 text-white rounded-xl p-8 mb-8">
          <div className="text-center mb-6">
            <h2 className="text-3xl font-bold mb-4">ARIMA(p, d, q) 模型公式</h2>
            <div className="text-xl mb-2">完整形式：</div>
            <div className="text-3xl font-mono bg-white/10 rounded-lg p-4 inline-block">
              φ(B)(1-B)ᵈ yₜ = θ(B)εₜ
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 mt-6">
            <div className="bg-white/10 rounded-lg p-4">
              <div className="text-blue-300 text-sm mb-1">p (AR阶数)</div>
              <div className="text-4xl font-bold">{arimaParams.p}</div>
              <div className="text-sm mt-2">自回归项数量</div>
            </div>
            <div className="bg-white/10 rounded-lg p-4">
              <div className="text-purple-300 text-sm mb-1">d (差分阶数)</div>
              <div className="text-4xl font-bold">{arimaParams.d}</div>
              <div className="text-sm mt-2">去趋势次数</div>
            </div>
            <div className="bg-white/10 rounded-lg p-4">
              <div className="text-pink-300 text-sm mb-1">q (MA阶数)</div>
              <div className="text-4xl font-bold">{arimaParams.q}</div>
              <div className="text-sm mt-2">移动平均项数量</div>
            </div>
          </div>
        </div>

        <div className="flex gap-3 mb-8 overflow-x-auto pb-2">
          {[
            { id: 'original', label: '原始序列', icon: Activity },
            { id: 'differencing', label: '差分 (I)', icon: TrendingUp },
            { id: 'ar', label: 'AR 过程', icon: RefreshCw },
            { id: 'ma', label: 'MA 过程', icon: RefreshCw },
            { id: 'acf', label: 'ACF定阶', icon: Activity }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setShowStep(id)}
              className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all whitespace-nowrap ${
                showStep === id
                  ? 'bg-blue-600 text-white shadow-lg scale-105'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              <Icon size={18} />
              {label}
            </button>
          ))}
        </div>

        <div className="bg-slate-50 rounded-xl p-6">
          
          {showStep === 'original' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">📈 步骤 1：原始时间序列</h2>
                
                <div className="mb-4 text-slate-700">
                  <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded text-lg font-mono">
                    yₜ = Trend(趋势) + Seasonal(季节性) + Noise(随机噪音)
                  </div>
                </div>

                <LineChart width={900} height={350} data={originalData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="time" label={{ value: '时间', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: '价格', angle: -90, position: 'insideLeft' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
                  <Legend />
                  <Line type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={2} name="原始价格" dot={false} />
                  <Line type="monotone" dataKey="trend" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" name="趋势成分" dot={false} />
                </LineChart>

                <div className="grid grid-cols-3 gap-4 mt-6">
                  <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
                    <div className="text-red-700 font-bold mb-2">🔺 趋势</div>
                    <div className="text-sm text-slate-700">长期方向性运动</div>
                    <div className="text-2xl font-bold text-red-600 mt-2">非平稳</div>
                  </div>
                  <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                    <div className="text-green-700 font-bold mb-2">🌊 季节性</div>
                    <div className="text-sm text-slate-700">周期性重复模式</div>
                    <div className="text-2xl font-bold text-green-600 mt-2">可预测</div>
                  </div>
                  <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
                    <div className="text-blue-700 font-bold mb-2">⚡ 噪音</div>
                    <div className="text-sm text-slate-700">随机波动</div>
                    <div className="text-2xl font-bold text-blue-600 mt-2">白噪音</div>
                  </div>
                </div>

                <div className="mt-6 bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    ⚠️ <strong>问题：</strong>原始序列包含趋势，是<strong>非平稳</strong>的。
                    ARIMA 要求数据平稳，需要先做<strong>差分</strong>。
                  </p>
                </div>
              </div>
            </div>
          )}

          {showStep === 'differencing' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">🔄 步骤 2：差分 (I)</h2>
                
                <div className="mb-4 text-slate-700">
                  <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                    <div className="font-mono text-xl mb-2">∇yₜ = yₜ - yₜ₋₁</div>
                    <div className="text-sm">一阶差分消除线性趋势</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div>
                    <h3 className="text-lg font-bold text-slate-700 mb-3">原始序列（非平稳）</h3>
                    <LineChart width={430} height={250} data={originalData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={2} dot={false} />
                    </LineChart>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-bold text-slate-700 mb-3">差分后（平稳）</h3>
                    <LineChart width={430} height={250} data={differencedData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="5 5" />
                      <Line type="monotone" dataKey="differenced" stroke="#10b981" strokeWidth={2} dot={false} />
                    </LineChart>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
                    <div className="text-red-700 font-bold mb-2">❌ 差分前</div>
                    <ul className="text-sm text-slate-700 space-y-1">
                      <li>• 均值不稳定（有趋势）</li>
                      <li>• 方差随时间变化</li>
                      <li>• 无法直接建模</li>
                    </ul>
                  </div>
                  <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                    <div className="text-green-700 font-bold mb-2">✅ 差分后</div>
                    <ul className="text-sm text-slate-700 space-y-1">
                      <li>• 均值恒定（围绕0）</li>
                      <li>• 方差稳定</li>
                      <li>• 可应用AR/MA模型</li>
                    </ul>
                  </div>
                </div>

                <div className="mt-6 bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    💡 <strong>I = Integrated：</strong> d=1 做一阶差分，d=2 做二阶差分。
                  </p>
                </div>
              </div>
            </div>
          )}

          {showStep === 'ar' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">🔁 步骤 3：自回归 (AR)</h2>
                
                <div className="mb-4">
                  <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                    <div className="font-mono text-xl mb-2">yₜ = φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + εₜ</div>
                    <div className="text-sm">当前值依赖过去 p 个值</div>
                  </div>
                </div>

                <div className="mb-6">
                  <h3 className="text-lg font-bold text-slate-700 mb-3">AR(1)：yₜ = 0.7·yₜ₋₁ + εₜ</h3>
                  <LineChart width={900} height={350} data={arData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" label={{ value: '时间', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: '值', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="5 5" />
                    <Line type="monotone" dataKey="value" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                  </LineChart>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
                    <div className="text-purple-700 font-bold mb-2">φ = 0.7</div>
                    <div className="text-sm text-slate-700">正相关：高值后继续高</div>
                  </div>
                  <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
                    <div className="text-red-700 font-bold mb-2">φ = -0.5</div>
                    <div className="text-sm text-slate-700">负相关：高低交替震荡</div>
                  </div>
                  <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                    <div className="text-green-700 font-bold mb-2">φ = 0</div>
                    <div className="text-sm text-slate-700">无相关：纯白噪音</div>
                  </div>
                </div>

                <div className="mt-6 bg-indigo-50 border-l-4 border-indigo-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    🎯 <strong>实战：</strong> AR 捕捉动量效应。φ₁ &gt; 0 说明价格有惯性。
                  </p>
                </div>
              </div>
            </div>
          )}

          {showStep === 'ma' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">📊 步骤 4：移动平均 (MA)</h2>
                
                <div className="mb-4">
                  <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
                    <div className="font-mono text-xl mb-2">yₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ...</div>
                    <div className="text-sm">当前值依赖过去 q 个随机冲击</div>
                  </div>
                </div>

                <div className="mb-6">
                  <h3 className="text-lg font-bold text-slate-700 mb-3">MA(1)：yₜ = εₜ + 0.5·εₜ₋₁</h3>
                  <LineChart width={900} height={350} data={maData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" label={{ value: '时间', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: '值', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="5 5" />
                    <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={false} />
                  </LineChart>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-5">
                    <div className="text-blue-700 font-bold mb-3">AR vs MA</div>
                    <div className="space-y-2 text-sm">
                      <div><strong>AR：</strong>记住过去的值</div>
                      <div><strong>MA：</strong>记住过去的冲击</div>
                    </div>
                  </div>
                  
                  <div className="bg-amber-50 border-2 border-amber-300 rounded-lg p-5">
                    <div className="text-amber-700 font-bold mb-3">MA 记忆长度</div>
                    <div className="text-sm space-y-1">
                      <div>• MA(1)：记得1次冲击</div>
                      <div>• MA(2)：记得2次冲击</div>
                      <div className="text-xs text-slate-500">q 期后冲击消失</div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 bg-green-50 border-l-4 border-green-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    🎯 <strong>实战：</strong> MA 捕捉均值回归。θ₁ &lt; 0 说明价格快速修正。
                  </p>
                </div>
              </div>
            </div>
          )}

          {showStep === 'acf' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg p-6 shadow-md">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">📉 步骤 5：ACF 定阶</h2>
                
                <div className="mb-4">
                  <div className="bg-slate-100 border-l-4 border-slate-500 p-4 rounded">
                    <div className="text-sm space-y-1">
                      <div><strong>ACF：</strong>自相关函数，测量 yₜ 与 yₜ₋ₖ 相关性</div>
                      <div><strong>PACF：</strong>偏自相关，去除中间影响</div>
                    </div>
                  </div>
                </div>

                <div className="mb-6">
                  <h3 className="text-lg font-bold text-slate-700 mb-3">ACF 图</h3>
                  <BarChart width={900} height={300} data={acfData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="lag" label={{ value: '滞后期', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'ACF', angle: -90, position: 'insideLeft' }} domain={[-1, 1]} />
                    <Tooltip />
                    <ReferenceLine y={0} stroke="#94a3b8" />
                    <ReferenceLine y={0.196} stroke="#ef4444" strokeDasharray="5 5" />
                    <ReferenceLine y={-0.196} stroke="#ef4444" strokeDasharray="5 5" />
                    <Bar dataKey="acf" radius={[4, 4, 0, 0]}>
                      {acfData.map((entry, idx) => (
                        <cell key={idx} fill={entry.significant ? '#3b82f6' : '#94a3b8'} />
                      ))}
                    </Bar>
                  </BarChart>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
                    <div className="text-blue-700 font-bold mb-2">MA(q) 特征</div>
                    <div className="text-sm space-y-1">
                      <div>• ACF 在 lag=q 截断</div>
                      <div>• PACF 拖尾衰减</div>
                    </div>
                  </div>
                  <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
                    <div className="text-purple-700 font-bold mb-2">AR(p) 特征</div>
                    <div className="text-sm space-y-1">
                      <div>• ACF 拖尾衰减</div>
                      <div>• PACF 在 lag=p 截断</div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 bg-indigo-50 border-l-4 border-indigo-500 p-4 rounded">
                  <p className="text-slate-700 font-medium">
                    💡 <strong>定阶规则：</strong>观察 ACF/PACF 图，找截断位置确定 p, q。
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="mt-8 bg-gradient-to-r from-slate-800 to-slate-900 text-white rounded-xl p-6">
          <h3 className="text-xl font-bold mb-3">🎯 ARIMA 完整流程</h3>
          <div className="text-sm space-y-2">
            <div>1️⃣ 检查平稳性 → 如非平稳，做差分（确定 d）</div>
            <div>2️⃣ 观察 ACF/PACF → 确定 p 和 q</div>
            <div>3️⃣ 估计参数 φ, θ → 最大似然估计</div>
            <div>4️⃣ 模型诊断 → 检查残差是否白噪音</div>
            <div>5️⃣ 预测未来 → 生成置信区间</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ARIMAMathDemo;