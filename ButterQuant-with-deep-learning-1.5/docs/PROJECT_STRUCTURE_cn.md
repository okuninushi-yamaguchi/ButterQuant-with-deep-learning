# 项目结构与架构文档

## 1. 项目概览
本文档概述了 **ButterQuant** 项目的架构和目录结构。它旨在帮助开发人员理解代码库的组织方式，以便随着项目的扩展进行升级和维护。

**技术栈：**
- **后端 (Backend):** Python (Flask), Pandas, NumPy, Statsmodels, Arch (GARCH 模型), yfinance.
- **前端 (Frontend):** React, Vite, TypeScript, Recharts, TailwindCSS (隐含或标准 CSS), Lucide React.
- **数据 (Data):** SQLite (主要), JSON (后备/缓存).

---

## 2. 目录结构图

```text
butterquantdltest/
├── backend/                        # Python 后端应用程序
│   ├── data/                       # 数据持久层
│   │   ├── history.db              # 主 SQLite 数据库 (历史数据)
│   │   ├── rankings_combined.json  # 缓存的排名数据
│   │   └── ...
│   ├── logs/                       # 应用程序运行日志
│   ├── venv/                       # Python 虚拟环境
│   ├── app.py                      # Flask 应用程序入口点和 API 路由
│   ├── analyzer.py                 # 核心量化分析逻辑 (蝴蝶期权, Greeks 等)
│   ├── daily_scanner.py            # 后台任务：每日扫描市场数据
│   ├── database.py                 # 数据库连接和类 ORM 辅助工具
│   ├── analyze_db.py               # 数据库分析工具
│   ├── deep_analysis_db.py         # 深度分析工具
│   ├── ticker_utils.py             # 股票代码辅助函数
│   ├── streamlit_app.py            # 独立的 Streamlit 仪表盘 (替代 UI)
│   ├── config.yaml                 # 后端配置
│   ├── Procfile                    # 部署配置 (Heroku/Render)
│   └── requirements.txt            # Python 依赖项
│
├── src/                            # 前端源代码 (React)
│   ├── assets/                     # 静态资源 (图片, 图标)
│   ├── components/                 # 可复用的 UI 组件
│   │   ├── Dashboard.tsx           # 主仪表盘视图
│   │   └── ...
│   ├── locales/                    # 国际化 (i18n) Json 文件
│   ├── App.tsx                     # React 主组件与路由
│   ├── main.tsx                    # React 入口点
│   ├── config.ts                   # 前端配置 (API URL)
│   ├── i18n.ts                     # i18n 设置
│   └── vite-env.d.ts               # Vite 类型定义
│
├── data collection and processing/  # 独立数据脚本
│   └── (用于探索性数据分析或抓取的脚本)
│
├── Methodology/                    # 文档与研究笔记
│   └── (解释量化策略的 Markdown 文件)
│
├── dist/                           # 生产构建输出 (前端)
├── public/                         # 公共静态文件 (根级别)
├── node_modules/                   # Node.js 依赖项
├── index.html                      # HTML 入口点
├── package.json                    # 前端依赖项与脚本
├── tsconfig.json                   # TypeScript 配置
├── vite.config.ts                  # Vite 打包配置
└── README.md                       # 通用项目信息
```

---

## 3. 组件详情

### 3.1 后端 (Flask)
后端是基于 **Flask** 构建的 RESTful API。它处理数据获取 (`yfinance`)、复杂的量化分析 (`analyzer.py`)，并将数据提供给前端。

- **`app.py`**: 后端的核心。定义了 API 端点，如 `/api/rankings`, `/api/analyze`，并管理扫描的后台线程。
- **`analyzer.py`**: 包含 `ButterflyAnalyzer` 类。这是“大脑”所在的地方。它执行：
    - 去趋势傅立叶分析 (De-trended Fourier Analysis)。
    - GARCH 波动率建模。
    - Black-Scholes 定价与 Greeks 计算。
    - 信号生成。
- **`daily_scanner.py`**: 设计为定期运行。它遍历股票代码列表，执行分析，并更新数据库/JSON 文件。
- **`database.py`**: SQLite 连接的封装，确保线程安全和一致的路径处理。

### 3.2 前端 (React)
前端是基于 **Vite** 构建的现代 **SPA (单页应用程序)**。

- **`App.tsx`**: 编排布局。连接仪表盘 (Dashboard) 和分析器 (Analyzer) 组件。
- **`components/`**: 关注点分离。
    - **Dashboard**: 显示市场概览和热门排名。
    - **OptionAnalyzer**: 特定股票代码的详细视图。
- **`i18n`**: 通过 `react-i18next` 内置支持多语言（英语/中文等）。

---

## 4. 工作流

### 数据管道 (Data Pipeline)
1.  **摄取 (Ingestion)**: `daily_scanner.py` 通过 `yfinance` 获取数据。
2.  **处理 (Processing)**: 数据传递给 `analyzer.py` 进行数学建模。
3.  **存储 (Storage)**: 结果保存到 `backend/data/history.db`，并经常缓存在 `.json` 中以便快速读取。
4.  **服务 (Serving)**: `app.py` 从 DB/JSON 读取并向 React 前端提供 JSON 响应。

### 部署 (隐含)
- **前端**: 通过 `npm run build` 构建到 `dist/`。
- **后端**: 通过 `gunicorn` (生产环境) 或 `python app.py` (开发环境) 运行。
- **集成**: 在某些配置中，前端可能由 Flask 应用程序作为静态文件提供服务，但目前设置为分离的开发服务器 (Vite: 5173, Flask: 5000)。

---

## 5. 未来规范化指南
随着项目的增长，请遵守以下规则：

1.  **API 版本控制**: 将端点保持在 `/api/` 下。如果发生破坏性更改，请使用 `/api/v2/`。
2.  **类型安全**: 继续在 `src/` 中使用 TypeScript。为 API 响应添加类型定义，以确保前端/后端同步。
3.  **模块化分析**: 如果 `analyzer.py` 超过 1000 行，请将其拆分为 `analysis/volatility.py`, `analysis/pricing.py` 等。
4.  **测试**:
    - 在 `backend/` 中添加 `tests/` 文件夹用于单元测试 (pytest)。
    - 在 `src/` 中添加组件测试 (Vitest/Jest)。
