import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
    TrendingUp,
    Clock,
    History,
    CheckCircle2,
    AlertCircle,
    Shield,
    Filter,
    ArrowUpRight,
    ExternalLink
} from 'lucide-react';

interface Trade {
    id: number;
    ticker: string;
    order_id: number;
    order_ref: string;
    strategy_type: string;
    butterfly_type: string;
    strikes: string;
    expiry: string;
    quantity: number;
    status: string;
    price: number;
    timestamp: string;
}

const TradeJournal: React.FC = () => {
    const { t } = useTranslation();
    const [trades, setTrades] = useState<Trade[]>([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState<'ALL' | 'AI' | 'Baseline'>('ALL');

    useEffect(() => {
        fetchTrades();
    }, []);

    const fetchTrades = async () => {
        try {
            const response = await fetch('http://localhost:5000/api/trades?limit=50');
            const data = await response.json();
            if (data.success) {
                setTrades(data.data);
            }
        } catch (error) {
            console.error('Failed to fetch trades:', error);
        } finally {
            setLoading(false);
        }
    };

    const filteredTrades = trades.filter(trade => {
        if (filter === 'ALL') return true;
        return trade.order_ref.includes(filter);
    });

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'SUBMITTED': return 'text-blue-400 bg-blue-400/10 border-blue-400/20';
            case 'FILLED': return 'text-green-400 bg-green-400/10 border-green-400/20';
            case 'CANCELLED': return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
            case 'ERROR': return 'text-red-400 bg-red-400/10 border-red-400/20';
            default: return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
        }
    };

    return (
        <div className="max-w-7xl mx-auto px-4 py-8">
            {/* 页面头部 / Page Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                <div>
                    <h1 className="text-3xl font-black text-white flex items-center gap-3">
                        <History className="w-8 h-8 text-indigo-500" />
                        Trade Execution Journal
                    </h1>
                    <p className="text-indigo-200/60 mt-2 font-medium">Real-time tracking of AI vs Baseline strategies</p>
                </div>

                <div className="flex items-center gap-2 bg-white/5 p-1 rounded-xl border border-white/10 backdrop-blur-md">
                    <button
                        onClick={() => setFilter('ALL')}
                        className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${filter === 'ALL' ? 'bg-indigo-600 text-white shadow-lg' : 'text-gray-400 hover:text-white'}`}
                    >
                        All Trades
                    </button>
                    <button
                        onClick={() => setFilter('AI')}
                        className={`px-4 py-2 rounded-lg text-sm font-bold transition-all flex items-center gap-2 ${filter === 'AI' ? 'bg-indigo-600 text-white shadow-lg' : 'text-gray-400 hover:text-white'}`}
                    >
                        <Shield className="w-4 h-4" /> AI Models
                    </button>
                    <button
                        onClick={() => setFilter('Baseline')}
                        className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${filter === 'Baseline' ? 'bg-indigo-600 text-white shadow-lg' : 'text-gray-400 hover:text-white'}`}
                    >
                        Baseline
                    </button>
                </div>
            </div>

            {/* 布局网格 / Layout Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
                <div className="lg:col-span-3 bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 overflow-hidden shadow-2xl">
                    <div className="p-6 border-b border-white/10 flex items-center justify-between">
                        <h2 className="text-xl font-bold text-white">Recent Orders</h2>
                        <button onClick={fetchTrades} className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                            <TrendingUp className="w-5 h-5 text-indigo-400" />
                        </button>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead>
                                <tr className="bg-white/5 text-gray-400 text-xs font-black uppercase tracking-widest border-b border-white/10">
                                    <th className="px-6 py-4">Instrument</th>
                                    <th className="px-6 py-4">Strategy</th>
                                    <th className="px-6 py-4">Type</th>
                                    <th className="px-6 py-4">Status</th>
                                    <th className="px-6 py-4">Quantity</th>
                                    <th className="px-6 py-4">Price</th>
                                    <th className="px-6 py-4">Timestamp</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/5">
                                {loading ? (
                                    <tr>
                                        <td colSpan={7} className="px-6 py-12 text-center text-indigo-200/50">
                                            Loading trade history...
                                        </td>
                                    </tr>
                                ) : filteredTrades.length === 0 ? (
                                    <tr>
                                        <td colSpan={7} className="px-6 py-12 text-center text-indigo-200/50">
                                            No trades found for this filter.
                                        </td>
                                    </tr>
                                ) : (
                                    filteredTrades.map((trade) => (
                                        <tr key={trade.id} className="hover:bg-white/5 transition-colors group">
                                            <td className="px-6 py-4">
                                                <div className="flex flex-col">
                                                    <span className="text-white font-bold">{trade.ticker}</span>
                                                    <span className="text-[10px] text-gray-500 font-medium font-mono">{trade.expiry}</span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className={`text-[10px] font-black px-2 py-1 rounded border ${trade.order_ref === 'ButterAI' ? 'text-purple-400 border-purple-400/30' : 'text-blue-400 border-blue-400/30'}`}>
                                                    {trade.order_ref}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 text-sm text-gray-300">
                                                {trade.butterfly_type}
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className={`text-[10px] font-black px-2 py-1 rounded-full border flex items-center gap-1 w-fit ${getStatusColor(trade.status)}`}>
                                                    {trade.status === 'SUBMITTED' ? <Clock className="w-3 h-3" /> : <CheckCircle2 className="w-3 h-3" />}
                                                    {trade.status}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 text-white font-mono font-bold">
                                                {trade.quantity}
                                            </td>
                                            <td className="px-6 py-4 text-white font-mono font-bold">
                                                ${trade.price.toFixed(2)}
                                            </td>
                                            <td className="px-6 py-4 text-[10px] text-gray-500 font-mono">
                                                {new Date(trade.timestamp).toLocaleString()}
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* 侧边栏统计 / Sidebar Stats */}
                <div className="space-y-6">
                    <div className="bg-gradient-to-br from-indigo-600 to-purple-700 rounded-3xl p-6 text-white shadow-xl relative overflow-hidden">
                        <div className="relative z-10">
                            <h3 className="text-sm font-black uppercase tracking-widest opacity-70">AI Win Rate Estimate</h3>
                            {/* 胜率估算 (待实盘数据支持) / Win Rate Estimate (Pending paper trading data) */}
                            <div className="text-4xl font-black mt-2">-- %</div>
                            <p className="text-xs mt-4 opacity-70">Collecting data from initial 50 paper trades...</p>
                        </div>
                        <Shield className="absolute -bottom-4 -right-4 w-24 h-24 opacity-10" />
                    </div>

                    <div className="bg-white/5 backdrop-blur-xl rounded-3xl p-6 border border-white/10">
                        <h3 className="text-sm font-black uppercase tracking-widest text-gray-400 mb-4">Total Execution</h3>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center">
                                <span className="text-indigo-200 text-sm">AI Trades</span>
                                <span className="text-white font-bold">{trades.filter(t => t.order_ref === 'ButterAI').length}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-indigo-200 text-sm">Baseline Trades</span>
                                <span className="text-white font-bold">{trades.filter(t => t.order_ref === 'ButterBaseline').length}</span>
                            </div>
                            <div className="h-2 bg-white/5 rounded-full mt-2 flex overflow-hidden">
                                <div
                                    className="bg-purple-500 h-full transition-all duration-1000"
                                    style={{ width: `${(trades.filter(t => t.order_ref === 'ButterAI').length / (trades.length || 1)) * 100}%` }}
                                />
                                <div
                                    className="bg-indigo-500 h-full transition-all duration-1000"
                                    style={{ width: `${(trades.filter(t => t.order_ref === 'ButterBaseline').length / (trades.length || 1)) * 100}%` }}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TradeJournal;
