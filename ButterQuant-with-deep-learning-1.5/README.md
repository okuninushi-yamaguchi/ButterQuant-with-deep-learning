# 🦋 ButterQuant v1.5 - Multi-Account Edition

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![Deep Learning](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Alpaca](https://img.shields.io/badge/Trade-Alpaca_Ready-yellow.svg)](https://alpaca.markets/)
[![IBKR](https://img.shields.io/badge/Trade-IBKR_Ready-green.svg)](https://www.interactivebrokers.com/)

[中文文档](README_CN.md)

---

**ButterQuant** is an AI-driven quantitative trading platform specializing in **Butterfly Option Strategies**. It now supports **Multi-Broker Integration**, allowing seamless switching between Alpaca and IBKR.

### 🆕 What's New in v1.5
- **Alpaca Integration**: Full support for Alpaca Markets (Paper & Live), including multi-leg option orders.
- **Account Isolation**: Independent execution engines for Alpaca and IBKR stored in separate directories.
- **Improved Contract Matching**: Enhanced algorithm for finding optimal option contracts across different exchanges.

### 🚀 Core Features
1.  **Dual-Broker Support**: Execute trades on either IBKR TWS or Alpaca via specialized folders.
2.  **AI Success Probability**: Uses a 4-class classification model (Loss, Minor, Good, Excellent) to filter trades.
3.  **Automated Execution**: Specialized scripts for AI-predicted ROI (>15%) and Baseline indicator signals.
4.  **Real-Time Dashboard**: React/TypeScript frontend for monitoring strategy performance and trade journals.

### 🛠️ Step-by-Step Installation

#### 1. Prerequisites
Ensure you have the following installed:
- **Python 3.12+**: [Download here](https://www.python.org/downloads/)
- **Node.js**: [Download here](https://nodejs.org/) (Required for the Dashboard)
- **Git**: [Download here](https://git-scm.com/)
- **IBKR TWS/Gateway**: Required for IBKR trading execution.
- **Alpaca API Keys**: Required for Alpaca trading execution.

#### 2. Clone & Setup Environments
```bash
# Clone the repository
git clone https://github.com/okuninushi-yamaguchi/ButterQuant-with-deep-learning.git
cd ButterQuant-with-deep-learning

# Create & Activate Backend Environment
python -m venv backend/.venv
# Windows:
.\backend\.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

#### 3. Broker Configuration
Set up your `.env` file in the `backend/` directory:
```env
# Alpaca
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=True
```

### ⛵ How to Run

#### Phase A: Model Preparation (Optional)
If pre-trained models are missing:
1. **Generate Data**: `python ml/generate_simulated_data.py`
2. **Train Model**: `python ml/train_model.py`
3. **Deploy**: `python ml/export_onnx.py`

#### Phase B: Daily Operation
1. **Run Scanner**: `python backend/daily_scanner.py`
2. **Start Dashboard**:
   - Backend API: `python backend/app.py`
   - Frontend UI: `npm run dev`
3. **Trade Execution**:
   - **For Alpaca**: `python "alpaca trade/execution_engine_alpaca.py"`
   - **For IBKR**: `python "ibkr trade/execution_engine_ibkr.py"`

---
> [!NOTE]
> Use the [Alpaca Setup Guide](docs/Alpaca_Setup_Guide.md) for detailed Alpaca configuration.

---

## 📖 In-depth Guides
- [Operation Guide](docs/OPERATION_GUIDE.md): Scheduling and maintenance instructions.
- [MLOps Guide](docs/ML_OPERATIONS_GUIDE.md): Model updates and troubleshooting.
- [ML Workflow](ml/ML_GUIDE.md): Detailed explanation of the DL pipeline.


---

## ⚖️ License & Disclaimer
ButterQuant is for **educational and research purposes only**. Trading options involves significant risk. The authors are not responsible for any financial losses incurred through use of this software.

---
*Developed by okuninushi-yamaguchi*
