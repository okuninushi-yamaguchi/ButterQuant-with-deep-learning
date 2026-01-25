# Alpaca 自动化交易设置指南

为了让 ButterQuant 通过 Alpaca 执行交易，请按照以下步骤配置您的环境。

## 1. 获取 API 密钥
1. 登录 [Alpaca Dashboard](https://app.alpaca.markets/)。
2. 在右侧栏找到 **API Keys** 部分。
3. 如果是第一次使用，点击 **Generate Keys**。
4. 您将获得一个 **API Key ID** 和 **Secret Key**。

## 2. 配置 Python 环境
1. 确保已安装 `alpaca-py` 和 `python-dotenv`：
   ```bash
   pip install alpaca-py python-dotenv
   ```
2. 在项目根目录或 `backend/` 目录下创建 `.env` 文件（或修改现有文件）：
   ```env
   ALPACA_API_KEY=您的_API_KEY_ID
   ALPACA_SECRET_KEY=您的_SECRET_KEY
   ALPACA_PAPER=True
   ```

## 3. 开启期权交易权限
1. 在 Alpaca Dashboard 中，确保您的账户已开启 **Options Trading**。
2. 即使是 Paper 账户，也需要完成简单的问卷申请以激活期权权限。

## 4. 运行与验证
1. 使用我们提供的验证脚本检查连接：
   ```bash
   python check/test_alpaca_connection.py
   ```
2. 如果连接成功，您可以运行执行引擎：
   ```bash
   python backend/execution_engine.py
   ```

---
**注意**: Alpaca Paper 账户提供实时的 IEX 数据，但对于期权，由于市场波动，延时数据可能会影响成交。建议在交易时段运行。
