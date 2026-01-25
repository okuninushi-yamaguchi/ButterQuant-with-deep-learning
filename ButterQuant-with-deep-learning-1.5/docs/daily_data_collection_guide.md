# 每日数据采集指南

## 📅 每日数据写入流程

### 前置条件检查

1. **确认 Docker 容器运行中**
   ```powershell
   docker ps
   ```
   应该看到 `butterquant_timescale` 容器在运行

2. **如果容器未运行，启动它**
   ```powershell
   cd c:\Users\okuninushi\Downloads\butterquantdltest
   docker-compose up -d
   ```

---

## 🚀 数据采集方式

### 方式一：完整扫描（推荐用于每日收盘后）

**运行完整的市场扫描**：
```powershell
cd c:\Users\okuninushi\Downloads\butterquantdltest\backend
python daily_scanner.py
```

**这个脚本会做什么**：
- 扫描 S&P 500 和 NASDAQ 100 的所有股票
- 对每只股票进行完整的蝴蝶策略分析（傅立叶、ARIMA、GARCH）
- 调用 AI 模型计算 `ml_success_prob`（成功概率）
- 将所有结果写入 TimescaleDB 的 `analysis_history` 表
- 生成排名数据到 `rankings_combined.json`

**预计时间**：30-60 分钟（取决于网络和股票数量）

---

### 方式二：增量更新（快速测试）

如果您只想测试几只股票或更新特定股票：

```powershell
cd c:\Users\okuninushi\Downloads\butterquantdltest\backend
python -c "from daily_scanner import scan_and_rank; scan_and_rank(limit=10)"
```

这会只扫描前 10 只股票，适合快速验证系统是否正常。

---

## 📊 数据验证

### 检查今天的数据是否成功写入

**方法 1：通过前端查看**
1. 确保后端和前端都在运行
2. 打开浏览器访问 `http://localhost:5173`
3. 查看热力图或排名页面，应该能看到最新数据

**方法 2：直接查询数据库**
```powershell
docker exec -it butterquant_timescale psql -U postgres -d butterquant -c "SELECT COUNT(*), MAX(timestamp) FROM analysis_history WHERE timestamp::date = CURRENT_DATE;"
```

这会显示今天写入了多少条记录和最新的时间戳。

---

## 🔄 自动化建议（未来）

### 设置 Windows 任务计划程序

您可以设置每天美股收盘后（美东时间 16:30，北京时间凌晨 4:30/5:30）自动运行：

1. 打开"任务计划程序"
2.- [x] Phase 3: IBKR Paper Trading Integration
    - [x] Setup IBKR TWS/Gateway and connection test (Fixed Scanner/VIX NaN issues)
    - [/] Implement `trader.py` with `orderRef` virtual separation
    - [x] Integrate ML scores into logic (Verified & Debugged)
rquantdltest\backend`

---

## ⚠️ 常见问题

### 问题 1：数据库连接失败
**错误信息**：`could not connect to server`

**解决方案**：
```powershell
docker-compose restart
```

### 问题 2：扫描中断
**原因**：网络问题或 yfinance API 限流

**解决方案**：
- 脚本会自动跳过失败的股票
- 可以重新运行，已扫描的股票会被跳过（基于时间戳）

### 问题 3：AI 模型加载失败
**错误信息**：`ONNX model not found`

**解决方案**：
确认模型文件存在：
```powershell
dir ml\models\success_model.onnx
```

如果不存在，重新训练模型：
```powershell
cd ml
python train_model.py
python export_onnx.py
```

---

## 📈 数据积累策略

### 建议的采集频率

- **工作日**：每天收盘后运行一次
- **周末**：不需要运行（市场休市）
- **节假日**：跳过

### 数据保留策略

TimescaleDB 会自动管理数据压缩和归档。当前配置：
- 保留所有原始数据
- 自动创建时间分区
- 定期压缩旧数据以节省空间

---

## 🎯 今天的行动建议

由于您的 IBKR 模拟账户还在审批中，今天可以：

1. **运行一次完整扫描**，为 AI 模型积累更多训练数据
2. **检查数据质量**，确保没有异常值
3. **观察 AI 评分分布**，看看哪些股票被 AI 认为最有潜力

运行命令：
```powershell
cd c:\Users\okuninushi\Downloads\butterquantdltest\backend
python daily_scanner.py
```

等 IBKR 账号通过后，这些历史数据会成为回测和策略验证的宝贵资源！
