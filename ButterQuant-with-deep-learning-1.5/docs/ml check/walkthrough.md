# ButterQuant ML Integration Walkthrough

## 🎯 项目成果 / Project Achievements

本项目成功完成了 ButterQuant ML v2.0 的深度集成与优化。我们解决了早期诊断出的关键问题，并建立了一套健壮的 ML 运维体系。

### ✅ 核心修复
- **特征维度修正**: 确认并修复了代码与文档的不一致（实际 **22维**，而非文档所述23维）。
- **数据泄露根除**: V2模型已完全移除 `total_score` 特征，杜绝了"用答案预测答案"的问题。
- **重训完成**: 使用最新的 10000+ 条样本阈值完成了模型重训，生成了新的 `success_model_v2.onnx`。

### 🛠️ 新增工具集

我们在 `check/` 和 `ml/` 目录下创建了一系列实用脚本：

| 脚本 | 路径 | 用途 |
|------|------|------|
| **集成验证** | `check/verify_ml_integration.py` | 一键检查模型文件、特征维度和推理引擎状态 |
| **性能监控** | `check/monitor_ml_performance.py` | 追踪 ButterAI vs ButterBaseline 的实盘表现和预测分布 |
| **重训调度** | `ml/retrain_scheduler.py` | **反馈闭环核心**。检查新数据量，自动备份旧模型并触发重训 |
| **A/B测试** | `check/ab_test_manager.py` | 管理实验组/对照组流量比例，生成 A/B 对比报告 |

## 🚀 使用指南 / User Guide

### 1. 日常巡检
每天运行以下命令确保 ML 组件正常工作：

```bash
python check/verify_ml_integration.py
```

### 2. 监控表现
每周查看一次 AI 与传统策略的对比：

```bash
python check/monitor_ml_performance.py --days 30
```

### 3. 反馈闭环 (重训)
系统会自动积累数据。当你想检查是否需要重训时：

```bash
# 检查状态 (我们已设置每日新增样本阈值为 10000)
python ml/retrain_scheduler.py --check

# 如果需要强制重训
python ml/retrain_scheduler.py --retrain --force
```

### 4. 调整 A/B 测试
目前的默认配置是 70% AI / 30% Baseline。如果你想调整：

```bash
# 查看当前状态
python check/ab_test_manager.py --status

# 调整为 50/50
python check/ab_test_manager.py --set-ratio 50 50
```

## 📊 验证结果快照

- **特征维度**: 22维 (已验证)
- **模型状态**: `ml/models/success_model_v2.onnx` (刚刚更新)
- **A/B 比例**: ButterAI (70%) / ButterBaseline (30%)

---
*Created by Antigravity on 2026-01-20*
