# Project Structure & Architecture Documentation

## 1. Overview
This document outlines the architecture and directory structure of the **ButterQuant** project. It serves as a guide for developers to understand the codebase organization, facilitating easier upgrades and maintenance as the project scales.

**Tech Stack:**
- **Backend:** Python (Flask), Pandas, NumPy, Statsmodels, Arch (GARCH models), yfinance.
- **Frontend:** React, Vite, TypeScript, Recharts, TailwindCSS (implied or standard CSS), Lucide React.
- **Data:** SQLite (primary), JSON (fallback/cache).

---

## 2. Directory Structure Map

```text
butterquantdltest/
├── backend/                        # Python Backend Application
│   ├── data/                       # Data persistence layer
│   │   ├── history.db              # Main SQLite database (historical data)
│   │   ├── rankings_combined.json  # Cached ranking data
│   │   └── ...
│   ├── logs/                       # Application runtime logs
│   ├── venv/                       # Python Virtual Environment
│   ├── app.py                      # Main Flask Application Entry Point & API Routes
│   ├── analyzer.py                 # Core Quant Analysis Logic (Butterly Option, Greeks, etc.)
│   ├── daily_scanner.py            # Background Job: Scans market data daily
│   ├── database.py                 # Database Connection & ORM-like helpers
│   ├── analyze_db.py               # DB Analysis utilities
│   ├── deep_analysis_db.py         # Deep Analysis utilities
│   ├── ticker_utils.py             # Helper functions for stock tickers
│   ├── streamlit_app.py            # Standalone Streamlit Dashboard (Alternative UI)
│   ├── config.yaml                 # Backend Configuration
│   ├── Procfile                    # Deployment configuration (Heroku/Render)
│   └── requirements.txt            # Python Dependencies
│
├── src/                            # Frontend Source Code (React)
│   ├── assets/                     # Static assets (images, icons)
│   ├── components/                 # Reusable UI Components
│   │   ├── Dashboard.tsx           # Main Dashboard View
│   │   └── ...
│   ├── locales/                    # Internationalization (i18n) Json files
│   ├── App.tsx                     # Main React Component & Routing
│   ├── main.tsx                    # React Entry Point
│   ├── config.ts                   # Frontend Configuration (API URLs)
│   ├── i18n.ts                     # i18n Setup
│   └── vite-env.d.ts               # Vite Types
│
├── data collection and processing/  # Independent Data Scripts
│   └── (Scripts for exploratory data analysis or fetching)
│
├── Methodology/                    # Documentation & Research Notes
│   └── (Markdown files explaining Quant strategies)
│
├── dist/                           # Production Build Output (Frontend)
├── public/                         # Public Static Files (root level)
├── node_modules/                   # Node.js Dependencies
├── index.html                      # HTML Entry Point
├── package.json                    # Frontend Dependencies & Scripts
├── tsconfig.json                   # TypeScript Configuration
├── vite.config.ts                  # Vite Bundler Configuration
└── README.md                       # General Project Info
```

---

## 3. Component Details

### 3.1 Backend (Flask)
The backend is a RESTful API built with **Flask**. It handles data fetching (`yfinance`), complex quantitative analysis (`analyzer.py`), and serves data to the frontend.

- **`app.py`**: The heart of the backend. Defines API endpoints like `/api/rankings`, `/api/analyze`, and manages background threads for scanning.
- **`analyzer.py`**: Contains the `ButterflyAnalyzer` class. This is where the "Brain" lives. It performs:
    - De-trended Fourier Analysis.
    - GARCH Volatility Modeling.
    - Black-Scholes Pricing & Greeks Calculation.
    - Signal Generation.
- **`daily_scanner.py`**: Designed to run periodically. It iterates through a list of tickers, performs analysis, and updates the database/JSON files.
- **`database.py`**: A wrapper around SQLite connections to ensure thread safety and consistent path handling.

### 3.2 Frontend (React)
The frontend is a modern **SPA (Single Page Application)** built with **Vite**.

- **`App.tsx`**: Orchestrates the layout. Connects the Dashboard and Analyzer components.
- **`components/`**: Clean separation of concerns.
    - **Dashboard**: Displays market overview and top rankings.
    - **OptionAnalyzer**: Detailed view for a specific ticker.
- **`i18n`**: Built-in support for Multi-language (English/Chinese etc.) via `react-i18next`.

---

## 4. Workflows

### Data Pipeline
1.  **Ingestion**: `daily_scanner.py` fetches data via `yfinance`.
2.  **Processing**: Data is passed to `analyzer.py` for math modeling.
3.  **Storage**: Results are saved to `backend/data/history.db` and often cached in `.json` for fast read access.
4.  **Serving**: `app.py` reads from DB/JSON and serves JSON responses to the React frontend.

### Deployment (Implied)
- **Frontend**: Built via `npm run build` to `dist/`.
- **Backend**: Runs via `gunicorn` (production) or `python app.py` (dev).
- **Integration**: The frontend is likely served separately or as static files by the Flask app in some configurations, but currently set up for separate dev servers (Vite: 5173, Flask: 5000).

---

## 5. Future Standardization Guidelines
As the project grows, adhere to these rules:

1.  **API Versioning**: Keep endpoints under `/api/`. If breaking changes occur, use `/api/v2/`.
2.  **Type Safety**: Continue using TypeScript in `src/`. Add type definitions for API responses to ensure Frontend/Backend sync.
3.  **Modular Analysis**: If `analyzer.py` exceeds 1000 lines, break it into `analysis/volatility.py`, `analysis/pricing.py`, etc.
4.  **Testing**:
    - Add `tests/` folder in `backend/` for unit tests (pytest).
    - Add component tests in `src/` (Vitest/Jest).
