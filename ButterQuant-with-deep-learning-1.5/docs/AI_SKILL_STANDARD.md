# ButterQuant AI Interaction Standard / ButterQuant AI 交互标准

> **Purpose**: This document serves as the "System 2" instructions for AI agents working on the ButterQuant project. Read this first to understand the context, constraints, and standard operating procedures (SOPs).
> **Objective**: Maximize coding efficiency, minimize token usage, and ensure architectural consistency.

---

## 🗺️ Project Map (Token Saver)

Instead of searching the entire workspace, lookup the relevant files here:

### 1. Core Backend (Python/Flask)
- **Entry Point**: `backend/app.py` (Flask API routes)
- **Batch Processing**: `backend/daily_scanner.py` (Daily batch scan & analysis)
- **Quant Logic**: `backend/analyzer.py` (Technical analysis, Fourier, GARCH)
- **Trading (Alpaca)**: `alpaca trade/alpaca_trader.py`, `alpaca trade/execution_engine_alpaca.py`
- **Trading (IBKR)**: `ibkr trade/ibkr_trader.py`, `ibkr trade/execution_engine_ibkr.py`
- **Database**: `backend/database.py` (Unified `DatabaseManager` for SQLite/PG)

### 2. Frontend (React/TypeScript)
- **Dashboards**: `src/components/Dashboard.tsx` (Main view), `src/components/TradeJournal.tsx` (Order management)
- **Analysis UI**: `src/components/OptionAnalyzer.tsx` (Strategy visualizer & AI score card)
- **Routing**: `src/App.tsx`

### 3. Machine Learning (PyTorch/ONNX)
- **Inference**: `backend/ml_inference.py` (Loads `.onnx` model & `.joblib` scaler)
- **Paths**: Models are stored in `backend/ml/models/`

---

## ⚡ Skills & SOPs (Standard Operating Procedures)

### Skill 1: Bilingual Coding Protocol 🇨🇳/🇺🇸
**Rule**: All comments in code MUST be bilingual in the format `{Chinese}/{English}`.
- **Good**: `# 计算移动平均线 / Calculate moving average`
- **Bad**: `# Calculate MA` or `# 计算均线`
- **Why**: Ensures maintainability for mixed language teams and future AI context.

### Skill 2: Database Interaction
**Rule**: NEVER use raw SQL in snippets unless modifying `database.py`. ALWAYS use `DatabaseManager` methods.
- **Pattern**:
  ```python
  from backend.database import DatabaseManager
  db = DatabaseManager()
  # Use high-level methods:
  # db.save_analysis(data)
  # db.get_latest_ranking()
  ```
- **Context**:
  - `history.db`: UI-facing JSON blobs (Source of Truth for Frontend).
  - `market_research.db`: Flattened rows for AI training.

### Skill 3: IBKR Trade Execution
**Rule**: Virtual account separation is enforced via `orderRef`.
- **`ButterAI`**: Trades triggered by the ML model (`execution_engine.py`).
- **`ButterBaseline`**: Trades from traditional ranking logic.
- **Safety**:
  - Always check `ml_inference.predict_success_probability()` > 0.7 before execution in AI mode.
  - Max allocation: $1,000 per strategy unit.

### Skill 4: Frontend Development
**Rule**: Use `lucide-react` for icons and `recharts` for charts.
- **Theme**: Dark mode default (`#1a1a1a` background).
- **Type Safety**: No `any`. Define interfaces in `src/types.ts` or component files.

### Skill 5: Strategy Coverage Integrity
**Rule**: ALWAYS ensure multi-directional strategy support (**CALL**, **PUT**, and **IRON**) in both logic and execution.
- **Check**: When modifying `analyzer.py`, `trader.py`, or `execution_engine.py`, verify that all three butterfly types are correctly handled.
- **Data Integrity**: Ensure the `butterfly_type` key is consistently passed between analysis and execution layers.
- **Why**: Prevents algorithmic bias and ensures the system can profit from both directional trends and sideways markets.

### Skill 6: Diagnostic Scripts (System Health)
**Rule**: Use standardized scripts in the `check/` directory for system verification. Do NOT write ad-hoc temp scripts in the root.
- **Location**: `check/`
- **Standard Scripts**:
  - `check/check_db_strategy.py`: Verifies if a specific ticker's strategy type (PUT/CALL/IRON) is correctly stored in `history.db`.
  - usage: `python check/check_db_strategy.py`
- **Protocol**:
  - Before confirming a "missing data" bug, run the relevant check script.
  - If a new type of check is needed, create a reusable script in `check/` instead of a one-off snippet.

### Skill 7: Dependency Management
**Rule**: Environmental consistency is mandatory. When introducing new libraries:
- **Python**: First, add the library to `backend/requirements.txt` before writing the code import.
- **Node/JS**: Update `package.json` immediately.
- **Other**: Clearly document installation steps in `task.md` or a setup script.
- **Protocol**:
  - Always verify package compatibility (e.g., `onnxruntime` vs `onnxruntime-gpu`).
  - Keep requirements versions pinned or minimum-bounded (e.g., `pkg>=1.0.0`).

### Skill 8: ML & Data Pipeline Protocol
**Rule**: Robustness is critical for long-running data jobs.
- **Validation First**: Every ML training pipeline MUST have a standalone `validate_data.py` script. Never train on unchecked data.
- **Simulation Reality**: Simulated data MUST include random noise (jitter) to prevent "Zero Variance" errors during model training (e.g., `value + np.random.normal(0, 0.01)`).
- **Save Fallbacks**: Long-running scripts (>10 mins) MUST implement save fallbacks. If `to_parquet` fails, automatically try `to_csv` or `to_json` to preserve work.

### Skill 9: External Data Robustness
**Rule**: Third-party data fetchers (e.g., `yfinance`, IBKR) are noisy and unreliable.
- **Noise Suppression**: Explicitly suppress low-value warnings from libraries.
  ```python
  logging.getLogger('yfinance').setLevel(logging.CRITICAL)
  ```
- **Timezone Safety**: Always be explicit about timezones. Use `ignore_tz=True` or convert to UTC immediately to avoid Pandas warnings.
- **Resilience**: Wrap individual ticker/item processing in `try/except` blocks. One failure (e.g., delisted stock) should NEVER crash the entire batch.

### Skill 10: Release Protocol
**Rule**: Before any `git push` or GitHub release, you MUST:
1.  **Update README**: Ensure `README.md` is up-to-date and supports **Bilingual (English/Chinese)**.
2.  **Write Diary**: Create a log entry in the `docs/diary/` folder (e.g., `docs/diary/YYYY-MM-DD_version_changes.md`) documenting key features and changes.
3.  **Check Standards**: Verify all changes comply with `AI_SKILL_STANDARD.md`.

---

## 🛑 Token Saving Protocol (Context Optimization)

1.  **Do NOT Read**:
    - `node_modules/`, `venv/`, `__pycache__/`
    - Large `.json` or `.db` files (unless checking schema header)
    - `package-lock.json` or `yarn.lock` (read `package.json` only)

2.  **Context Loading Strategy**:
    - **Debugging**: Read `backend/logs/app.log` or browser console first.
    - **New Function**: Read only the specific target file + `backend/database.py` (if DB needed).
    - **Refactoring**: Read `docs/AI_SKILL_STANDARD.md` (this file) + `task.md`.

---

## 📝 Recent Context (v1.2 Status)
- **Current Phase**: Phase 7 (Cleanup & Optimization).
- **Active Feature**: Deep Learning Integration & Live Paper Trading.
- **Known Issues**: `analyze_db.py` and `streamlit_app.py` have been removed. Do not reference them.
