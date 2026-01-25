import React from 'react';
import { ArrowLeft } from 'lucide-react';

interface LegalTermsProps {
    type: 'documentation' | 'privacy' | 'terms';
    onBack: () => void;
}

const LegalTerms: React.FC<LegalTermsProps> = ({ type, onBack }) => {
    const getContent = () => {
        switch (type) {
            case 'documentation':
                return (
                    <div className="prose prose-indigo max-w-none">
                        <h1 className="text-3xl font-bold mb-6">Documentation</h1>
                        <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
                            <h2 className="text-xl font-semibold mb-3">Disclaimer</h2>
                            <p className="text-gray-700 leading-relaxed text-lg italic">
                                The documentation provided on ButterQuant is for informational and educational purposes only.
                                It does not constitute investment advice, trading advice, or a recommendation to buy or sell any financial instrument.
                            </p>
                        </div>
                    </div>
                );
            case 'privacy':
                return (
                    <div className="prose prose-indigo max-w-none text-gray-700">
                        <h1 className="text-3xl font-bold text-gray-900 mb-6">Privacy Policy – ButterQuant</h1>
                        <p className="text-sm text-gray-500 mb-8">Last updated: January 2026</p>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">1. Introduction</h2>
                            <p>ButterQuant (“we”, “us”, “our”) respects your privacy and is committed to protecting your personal data. This Privacy Policy explains how we collect, use, and protect your information.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">2. Information We Collect</h2>
                            <ul className="list-disc pl-5 space-y-2">
                                <li>Email address and account information</li>
                                <li>Subscription and payment-related information (processed by third-party payment providers)</li>
                                <li>Usage data, cookies, and analytics data</li>
                                <li>Technical data such as IP address, browser type, and device information</li>
                            </ul>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">3. How We Use Your Information</h2>
                            <ul className="list-disc pl-5 space-y-2">
                                <li>Provide and maintain the ButterQuant service</li>
                                <li>Manage subscriptions and payments</li>
                                <li>Improve platform functionality and performance</li>
                                <li>Communicate service-related updates</li>
                                <li>Comply with legal and regulatory obligations</li>
                            </ul>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">4. Legal Basis for Processing (GDPR / UK GDPR)</h2>
                            <p>We process personal data based on:</p>
                            <ul className="list-disc pl-5 mt-2 space-y-2">
                                <li>Contractual necessity</li>
                                <li>Legitimate business interests</li>
                                <li>Legal compliance</li>
                                <li>User consent, where applicable</li>
                            </ul>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">5. Data Sharing</h2>
                            <p>We may share data with:</p>
                            <ul className="list-disc pl-5 mt-2 space-y-2">
                                <li>Payment processors (e.g., Stripe, Paddle)</li>
                                <li>Analytics and infrastructure providers</li>
                                <li>Legal or regulatory authorities when required by law</li>
                            </ul>
                            <p className="mt-4 font-semibold">We do not sell personal data.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">6. Cookies & Analytics</h2>
                            <p>ButterQuant uses cookies and similar technologies to improve user experience and analyze platform usage. You may control cookie preferences through your browser settings.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">7. Data Retention</h2>
                            <p>We retain personal data only for as long as necessary to fulfill the purposes described in this Policy or as required by law.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">8. Your Rights</h2>
                            <p>Depending on your jurisdiction, you may have the right to:</p>
                            <ul className="list-disc pl-5 mt-2 space-y-2">
                                <li>Access your personal data</li>
                                <li>Request correction or deletion</li>
                                <li>Restrict or object to processing</li>
                                <li>Withdraw consent</li>
                            </ul>
                            <p className="mt-4">Requests may be submitted to the contact address below.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">9. Data Security</h2>
                            <p>We implement reasonable technical and organizational safeguards to protect personal data. However, no system can be guaranteed to be 100% secure.</p>
                        </section>

                        <section className="mb-8 p-6 bg-gray-50 rounded-lg border border-gray-100">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">10. Contact</h2>
                            <p><strong>Email:</strong> <a href="mailto:privacy@butterquant.com" className="text-indigo-600 hover:underline">privacy@butterquant.com</a></p>
                        </section>
                    </div>
                );
            case 'terms':
                return (
                    <div className="prose prose-indigo max-w-none text-gray-700">
                        <h1 className="text-3xl font-bold text-gray-900 mb-6">Terms of Service & Financial Disclaimers – ButterQuant</h1>
                        <p className="text-sm text-gray-500 mb-8">Last updated: January 2026</p>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">1. Acceptance of Terms</h2>
                            <p>By accessing or subscribing to <strong>ButterQuant</strong> (“Platform”, “Service”), you agree to these Terms of Service. If you do not agree, you must not use the Platform.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">2. Description of Service</h2>
                            <p>ButterQuant provides:</p>
                            <ul className="list-disc pl-5 mt-2 space-y-2">
                                <li>Quantitative market analytics</li>
                                <li>Algorithmic signals and indicators</li>
                                <li>Statistical dashboards and visualizations</li>
                                <li>Backtesting and hypothetical performance data</li>
                            </ul>
                            <p className="mt-4">The Service is <strong>informational only</strong> and does not include trade execution or portfolio management.</p>
                        </section>

                        <section className="mb-8 bg-amber-50 p-6 rounded-lg border border-amber-100">
                            <h2 className="text-xl font-bold text-amber-900 mb-4">3. No Investment Advice (IMPORTANT)</h2>
                            <p className="text-amber-800">ButterQuant does <strong>not</strong> provide:</p>
                            <ul className="list-disc pl-5 mt-2 space-y-2 text-amber-800">
                                <li>Investment advice</li>
                                <li>Trading advice</li>
                                <li>Financial advice</li>
                                <li>Portfolio management services</li>
                            </ul>
                            <p className="mt-4 text-amber-800 font-medium italic">Nothing on the Platform constitutes a recommendation, solicitation, or offer to buy or sell any security, option, or derivative.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">4. No Advisory or Fiduciary Relationship</h2>
                            <p>Use of ButterQuant does <strong>not</strong> create:</p>
                            <ul className="list-disc pl-5 mt-2 space-y-2">
                                <li>An investment adviser–client relationship</li>
                                <li>A fiduciary duty</li>
                                <li>A broker-dealer relationship</li>
                                <li>A Commodity Trading Advisor (CTA) relationship</li>
                            </ul>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">5. Regulatory Status Disclosure</h2>
                            <div className="space-y-4">
                                <div>
                                    <h3 className="font-bold text-gray-900 mb-2 underline">United States</h3>
                                    <p>ButterQuant is <strong>not registered</strong> as an investment adviser with the U.S. Securities and Exchange Commission (SEC), is not a broker-dealer, and is not registered with FINRA, CFTC, or NFA. All signals are impersonal, automated, and informational.</p>
                                </div>
                                <div>
                                    <h3 className="font-bold text-gray-900 mb-2 underline">European Union</h3>
                                    <p>ButterQuant does not provide investment services under MiFID II and does not produce investment research or recommendations under ESMA regulations.</p>
                                </div>
                                <div>
                                    <h3 className="font-bold text-gray-900 mb-2 underline">United Kingdom</h3>
                                    <p>ButterQuant is not authorised or regulated by the Financial Conduct Authority (FCA).</p>
                                </div>
                            </div>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">6. No Personalization or Suitability Assessment</h2>
                            <p>ButterQuant does not consider your financial situation, objectives, or risk tolerance. All information is general in nature and may not be suitable for all users.</p>
                        </section>

                        <section className="mb-8 text-red-700 bg-red-50 p-6 rounded-lg border border-red-100 font-medium">
                            <h2 className="text-xl font-bold text-red-900 mb-4">7. Risk Disclosure</h2>
                            <p>Trading and investing in financial markets, particularly <strong>options and derivatives</strong>, involves substantial risk, including the possible loss of all invested capital.</p>
                            <p className="mt-2 text-red-900 font-bold">Past performance is not indicative of future results.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">8. Quantitative & Model Disclaimer</h2>
                            <p>You acknowledge that:</p>
                            <ul className="list-disc pl-5 mt-2 space-y-2">
                                <li>Signals are generated using statistical models and algorithms</li>
                                <li>Models rely on historical data and assumptions</li>
                                <li>Market conditions may change, rendering models ineffective</li>
                            </ul>
                            <p className="mt-4 italic">No model or signal can predict future market behavior with certainty.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">9. Backtesting & Hypothetical Performance</h2>
                            <p>Backtested or simulated results:</p>
                            <ul className="list-disc pl-5 mt-2 space-y-2">
                                <li>Do not reflect actual trading conditions</li>
                                <li>Exclude liquidity constraints, slippage, and execution delays</li>
                                <li>Are hypothetical in nature and subject to bias</li>
                            </ul>
                            <p className="mt-4 font-medium">Such results are not indicative of future performance.</p>
                        </section>

                        <section className="mb-8 underline decoration-double underline-offset-4 decoration-indigo-200">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">10. Subscription & Fees</h2>
                            <p>Subscription fees provide access to information and tools only. Fees are not performance-based and do not guarantee profitability.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">11. No Automated Trading or Execution</h2>
                            <p>ButterQuant does not execute trades, place orders, or connect directly to brokerage accounts. All trading decisions are solely the user’s responsibility.</p>
                        </section>

                        <section className="mb-8 bg-gray-900 text-gray-100 p-8 rounded-xl shadow-2xl">
                            <h2 className="text-xl font-bold text-white mb-4">12. Limitation of Liability</h2>
                            <p className="text-gray-300 leading-relaxed uppercase text-sm tracking-wide">To the maximum extent permitted by law, ButterQuant shall not be liable for: Trading losses or financial damages; Loss of profits or opportunities; Data inaccuracies or service interruptions.</p>
                            <p className="mt-4 text-white font-bold text-center border-t border-gray-700 pt-4">USE OF THE PLATFORM IS ENTIRELY AT YOUR OWN RISK.</p>
                        </section>

                        <section className="mb-8 mt-12">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">13. Intellectual Property</h2>
                            <p>All content, software, models, and analytics on ButterQuant are the exclusive property of ButterQuant and may not be copied or redistributed without permission.</p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">14. Governing Law</h2>
                            <p>These Terms are governed by and construed in accordance with the laws of <strong>England and Wales</strong>, without regard to conflict of law principles.</p>
                        </section>

                        <section className="mb-12 border-t pt-8 border-gray-100">
                            <h2 className="text-xl font-bold text-gray-900 mb-4">15. Contact</h2>
                            <p className="mb-1"><strong>Email:</strong> <a href="mailto:legal@butterquant.com" className="text-indigo-600 hover:underline">legal@butterquant.com</a></p>
                            <p><strong>Website:</strong> <a href="https://butterquant.com" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">https://butterquant.com</a></p>
                        </section>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <div className="min-h-screen bg-white">
            <div className="max-w-4xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
                <button
                    onClick={onBack}
                    className="flex items-center gap-2 text-indigo-600 hover:text-indigo-800 mb-10 transition-colors font-medium group"
                >
                    <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
                    Back to ButterQuant
                </button>

                <div className="bg-white">
                    {getContent()}
                </div>

                <div className="mt-16 pt-8 border-t border-gray-100 text-center text-gray-400 text-sm">
                    &copy; {new Date().getFullYear()} ButterQuant. All rights reserved.
                </div>
            </div>
        </div>
    );
};

export default LegalTerms;
