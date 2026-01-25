import React, { useState } from 'react';
import ButterflyOptionAnalyzer from './components/OptionAnalyzer';
import ButterflyDashboard from './components/Dashboard';
import StrategyRank from './components/StrategyRank';
import VixStrategy from './components/VixStrategy';
import TradeJournal from './components/TradeJournal';
import LegalTerms from './components/LegalTerms';
import { LayoutGrid, Waves, Languages, Trophy, Activity, History } from 'lucide-react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import logo from './assets/logo_remove_background.png';

export default function App() {
  const [view, setView] = useState<'dashboard' | 'analyzer' | 'rank' | 'vix' | 'trades' | 'documentation' | 'privacy' | 'terms'>('dashboard');
  const [selectedTicker, setSelectedTicker] = useState<string>('');
  const { t, i18n } = useTranslation();

  const toggleLanguage = () => {
    const newLang = i18n.language === 'zh' ? 'en' : 'zh';
    i18n.changeLanguage(newLang);
    localStorage.setItem('i18nextLng', newLang);
  };

  const handleTickerSelect = (ticker: string) => {
    setSelectedTicker(ticker);
    setView('analyzer');
  };

  // Dynamic Browser Tab Title Mapping
  const getPageTitle = () => {
    switch (view) {
      case 'dashboard': return `${t('nav.dashboard')} | ButterQuant`;
      case 'rank': return `${t('nav.rank')} | ButterQuant`;
      case 'vix': return `${t('nav.vix')} | ButterQuant`;
      case 'trades': return `Trade Journal | ButterQuant`;
      case 'documentation': return `Documentation | ButterQuant`;
      case 'privacy': return `Privacy Policy | ButterQuant`;
      case 'terms': return `Terms of Service | ButterQuant`;
      case 'analyzer': return ''; // Analyzer handles its own title with Helmet
      default: return 'ButterQuant';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {getPageTitle() && (
        <Helmet>
          <title>{getPageTitle()}</title>
        </Helmet>
      )}
      {/* Navigation Bar */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center gap-2">
                <img src={logo} alt="ButterQuant Logo" className="w-8 h-8 rounded-full object-cover" />
                <span className="font-bold text-xl text-gray-800 tracking-tight">ButterQuant</span>
              </div>
              <div className="hidden sm:ml-8 sm:flex sm:space-x-8">
                <button
                  onClick={() => setView('dashboard')}
                  className={`${view === 'dashboard'
                    ? 'border-indigo-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                    } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200`}
                >
                  <LayoutGrid className="w-4 h-4 mr-2" />
                  {t('nav.dashboard')}
                </button>
                <button
                  onClick={() => setView('rank')}
                  className={`${view === 'rank'
                    ? 'border-indigo-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                    } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200`}
                >
                  <Trophy className="w-4 h-4 mr-2" />
                  {t('nav.rank')}
                </button>
                <button
                  onClick={() => setView('analyzer')}
                  className={`${view === 'analyzer'
                    ? 'border-indigo-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                    } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200`}
                >
                  <Waves className="w-4 h-4 mr-2" />
                  {t('nav.analyzer')}
                </button>
                <button
                  onClick={() => setView('vix')}
                  className={`${view === 'vix'
                    ? 'border-indigo-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                    } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200`}
                >
                  <Activity className="w-4 h-4 mr-2" />
                  {t('nav.vix')}
                </button>
                <button
                  onClick={() => setView('trades')}
                  className={`${view === 'trades'
                    ? 'border-indigo-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                    } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200`}
                >
                  <History className="w-4 h-4 mr-2" />
                  {t('nav.trades')}
                </button>
              </div>
            </div>

            <div className="flex items-center">
              <button
                onClick={toggleLanguage}
                className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-50 border border-gray-300 rounded-md hover:bg-gray-100 transition-colors"
              >
                <Languages className="w-4 h-4" />
                {i18n.language === 'zh' ? 'English' : '中文'}
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className={view === 'dashboard' ? 'block' : 'hidden'}>
          <ButterflyDashboard onAnalyzeTicker={handleTickerSelect} />
        </div>

        <div className={view === 'rank' ? 'block' : 'hidden'}>
          <StrategyRank onAnalyze={handleTickerSelect} />
        </div>

        {view === 'analyzer' && (
          <ButterflyOptionAnalyzer initialTicker={selectedTicker} />
        )}

        <div className={view === 'vix' ? 'block' : 'hidden'}>
          <VixStrategy />
        </div>

        <div className={view === 'trades' ? 'block' : 'hidden'}>
          <TradeJournal />
        </div>

        {/* Legal Pages */}
        {(view === 'documentation' || view === 'privacy' || view === 'terms') && (
          <LegalTerms
            type={view as 'documentation' | 'privacy' | 'terms'}
            onBack={() => setView('dashboard')}
          />
        )}
      </main>

      {/* Global Footer */}
      {!(view === 'documentation' || view === 'privacy' || view === 'terms') && (
        <footer className="bg-white border-t border-gray-200 py-8 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col md:flex-row justify-between items-center gap-4">
              <div className="text-gray-400 text-sm font-medium">
                &copy; {new Date().getFullYear()} ButterQuant. All rights reserved.
              </div>
              <div className="flex items-center gap-6">
                <button
                  onClick={() => setView('documentation')}
                  className="text-sm font-medium text-gray-500 hover:text-indigo-600 transition-colors"
                >
                  Documentation
                </button>
                <button
                  onClick={() => setView('privacy')}
                  className="text-sm font-medium text-gray-500 hover:text-indigo-600 transition-colors"
                >
                  Privacy
                </button>
                <button
                  onClick={() => setView('terms')}
                  className="text-sm font-medium text-gray-500 hover:text-indigo-600 transition-colors"
                >
                  Terms
                </button>
              </div>
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}