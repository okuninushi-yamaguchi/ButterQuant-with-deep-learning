# 🦋 ButterQuant v1.5 - 多账户集成版

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![Deep Learning](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Alpaca](https://img.shields.io/badge/Trade-Alpaca_Ready-yellow.svg)](https://alpaca.markets/)
[![IBKR](https://img.shields.io/badge/Trade-IBKR_Ready-green.svg)](https://www.interactivebrokers.com/)

[English (英文文档)](README.md)

---

**ButterQuant** 是一个 AI 驱动的量化交易平台，专注于 **蝴蝶期权策略**。现已支持 **多券商集成**，可无缝切换 Alpaca 和 IBKR 交易。

### 🆕 v1.5 新版特性
- **Alpaca 集成**: 全面支持 Alpaca Markets (模拟与实盘)，包括多腿期权订单执行。
- **账户隔离**: 为 Alpaca 和 IBKR 设立了独立的执行引擎文件夹，确保环境互不干扰。
- **合约搜索优化**: 增强型期权合约匹配算法，在多平台间精准识别最优成交合约。

### 🚀 核心功能
1.  **多账户支持**: 通过特定文件夹 (`alpaca trade` / `ibkr trade`) 在不同券商间执行自动化交易。
2.  **AI 胜率预测**: 使用 4 分类模型（亏损、微利、良好、优秀）过滤高风险交易。
3.  **自动交易**: 根据 AI 预测收益 (>15%) 或传统技术指標信号自动下单。
4.  **实时仪表盘**: 基于 React/TypeScript 的前端界面，实时监控策略表现和交易日志。

### 🛠️ 分步安装指南

#### 1. 必要前提
请确保你的电脑已安装以下软件：
- **Python 3.12+**: [下载地址](https://www.python.org/downloads/)
- **Node.js**: [下载地址](https://nodejs.org/) (运行 Web 仪表盘必需)
- **Git**: [下载地址](https://git-scm.com/)
- **IBKR TWS/Gateway**: 自动化执行 IBKR 订单必需。
- **Alpaca API Keys**: 执行 Alpaca 交易必需。

#### 2. 克隆项目与环境配置
```bash
# 克隆仓库
git clone https://github.com/okuninushi-yamaguchi/ButterQuant-with-deep-learning.git
cd ButterQuant-with-deep-learning

# 创建并激活后端虚拟环境
python -m venv backend/.venv
# Windows 系统:
.\backend\.venv\Scripts\activate

# 安装依赖
pip install -r backend/requirements.txt
```

#### 3. 前端环境配置
**打开一个新的终端界面**（保持在项目根目录）：
```bash
# 安装前端依赖 (仅需执行一次)
npm install --legacy-peer-deps
```

#### 3. 券商配置
在 `backend/` 目录下设置您的 `.env` 文件：
```env
# Alpaca
ALPACA_API_KEY=您的_API_KEY
ALPACA_SECRET_KEY=您的_SECRET_KEY
ALPACA_PAPER=True
```

### ⛵ 如何运行

#### 阶段 A：模型准备 (可选)
如果目录下没有预训练模型：
1. **生成数据**: `python ml/generate_simulated_data.py`
2. **训练模型**: `python ml/train_model.py`
3. **导出模型**: `python ml/export_onnx.py`

#### 阶段 B：日常运行
1. **执行市场扫描**: `python backend/daily_scanner.py`

2. **启动仪表盘**:
   - 后端 API: `python backend/app.py`
   - 前端 UI: `npm run dev`
   - 在浏览器中打开 [http://localhost:5173](http://localhost:5173)。

3. **执行自动化交易**:
   - **Alpaca 交易**: `python "alpaca trade/execution_engine_alpaca.py"`
   - **IBKR 交易**: `python "ibkr trade/execution_engine_ibkr.py"`

---
> [!NOTE]
> 请参考 [Alpaca 设置指南](docs/Alpaca_Setup_Guide.md) 获取详细的 Alpaca 配置帮助。

---

## 📖 详细指南
- [日常运维手册](docs/OPERATION_GUIDE.md): 包含自动化排程和监控建议。
- [MLOps 维护指南](docs/ML_OPERATIONS_GUIDE.md): 模型更新、数据验证及故障排除。
- [ML 工作流详解](ml/ML_GUIDE.md): 深度学习管道的详细步骤。


---

## ⚖️ 免责声明
本项目仅供**教育和研究使用**。期权交易存在重大风险，作者不对任何资金损失负责。

---
*Developed by okuninushi-yamaguchi*