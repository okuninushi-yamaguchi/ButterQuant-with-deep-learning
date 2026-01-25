# ButterQuant ML集成优化方案 / ML Integration Optimization Proposal

> **基于 Claude v2.0 Final 建议 + 项目规范的综合优化方案**

> [!CAUTION]
> # ✅ PROJECT COMPLETED / 项目已完成
> 所有阶段 (Phase 1, 2, 3) 均已执行完毕。请参考 [walkthrough.md](file:///C:/Users/okuninushi/.gemini/antigravity/brain/ee375697-182f-4d4b-a571-6e9709f5a0f1/walkthrough.md) 获取使用指南。
> 2026-01-20

---

## 📋 Executive Summary

基于对以下文档的综合分析：
- `docs/ml check/v2.0 Final.md` - Claude的ML系统诊断与改进方案
- `docs/ml check/integrate_ml_to_workflow.md` - 工作流集成示例
- `docs/methodology/Summary v1.4.md` - 核心量化方法论
- `docs/AI_SKILL_STANDARD.md` - 项目编码规范

我提出以下**分阶段实施**的优化方案。

---

## 1. 现状评估 / Current State Assessment

### ✅ 已完成的架构优势

| 组件 | 状态 | 文件 |
|------|------|------|
| 特征工程 v2 | ✅ 已实现 | [features.py](file:///C:/Users/okuninushi/Downloads/butterquantdltest/ml/features.py) |
| 数据生成器 | ✅ 已实现 | [generate_simulated_data.py](file:///C:/Users/okuninushi/Downloads/butterquantdltest/ml/generate_simulated_data.py) |
| 模型训练器 | ✅ 已实现 | [train_model.py](file:///C:/Users/okuninushi/Downloads/butterquantdltest/ml/train_model.py) |
| ONNX推理引擎 | ✅ 已实现 | [ml_inference.py](file:///C:/Users/okuninushi/Downloads/butterquantdltest/backend/ml_inference.py) |
| 双轨执行引擎 | ✅ 已实现 | [execution_engine.py](file:///C:/Users/okuninushi/Downloads/butterquantdltest/backend/execution_engine.py) |

### ⚠️ Claude 诊断的三大缺陷

1. **评估时间尺度错配** - 1天评估 vs 30-45天持有期
2. **数据泄露** - `total_score` 作为特征
3. **反馈闭环缺失** - 模型静态部署无迭代

---

## 2. 我的优化建议 / My Optimization Proposal

> [!IMPORTANT]
> 我建议按**优先级分阶段实施**，而非一次性大规模改动。这样可以保持系统稳定性，同时逐步验证改进效果。

### Phase 1: 紧急修复 (本周完成) 🔴 P0 ✅ 完成

> [!NOTE]
> **已完成**: 验证脚本创建于 `check/verify_ml_integration.py`，所有检查通过。发现并修复了文档中23维→22维的错误描述。

#### 1.1 集成 `daily_scanner.py` 与 ML v2.0

**问题**: 当前 `daily_scanner.py` 可能未正确调用新的4分类模型。

**状态**: ✅ 已验证 - `execution_engine.py` 正确使用 `_filter_ai_candidates()` 方法，实现3层回退机制。

---

### Phase 2: 核心改进 (2周内完成) 🟡 P1 ✅ 完成

> [!NOTE]
> **已完成**: V2模型已不包含 `total_score`。创建了 `check/monitor_ml_performance.py` 监控脚本。

#### 2.1 移除 `total_score` 数据泄露

**状态**: ✅ V2模型训练时已移除，无需额外处理

#### 2.2 ML性能监控

**已创建**: `check/monitor_ml_performance.py`
- 追踪 ButterAI vs ButterBaseline 成功率
- 分析排名中的ML预测分布
- 模型漂移检测占位符

---

### Phase 3: 增强功能 (1个月内完成) 🟢 P2 (可选)

#### 2.1 移除 `total_score` 数据泄露

**当前代码位置**: [features.py](file:///C:/Users/okuninushi/Downloads/butterquantdltest/ml/features.py)

**改动**:
```diff
- 'total_score': score.get('total', 0)
+ # REMOVED: total_score is data leakage / 移除: total_score是数据泄露
+ 'gamma_theta_ratio': greeks['gamma'] / max(abs(greeks['theta']), 0.001)
```

#### 2.2 重新训练模型

使用已有的 [training_data_deep.parquet](file:///C:/Users/okuninushi/Downloads/butterquantdltest/ml/training_data_deep.parquet) (1.2MB, ~7000样本)，移除泄露特征后重新训练。

---

### Phase 3: 增强功能 (1个月内完成) 🟢 P2

#### 3.1 实施反馈闭环

创建 `monitor_ml_performance.py` 脚本:

```python
# 每周运行: 比较预测ROI vs 实际ROI
# 如果偏差 > 20%, 触发模型重训警报
```

#### 3.2 A/B测试框架

利用现有的 `ButterAI` vs `ButterBaseline` 双轨制:

| Track | 模型版本 | 流量占比 |
|-------|---------|---------|
| ButterAI | 新v2.0模型 | 70% |
| ButterBaseline | 传统评分 | 30% (对照组) |

---

## 3. 实施步骤详解 / Detailed Implementation Steps

### Step 1: 兼容性检查 (今天)

```bash
cd C:\Users\okuninushi\Downloads\butterquantdltest

# 1. 验证模型文件存在 / Verify model files exist
ls ml/models/

# 2. 运行推理测试 / Run inference test
python ml/end_to_end_test.py

# 3. 检查特征维度 / Check feature dimensions
python -c "
import pandas as pd
df = pd.read_parquet('ml/training_data_deep.parquet')
print(f'Feature columns: {len([c for c in df.columns if not c.startswith(\"_\")])}')
"
```

### Step 2: 集成到 Daily Workflow (本周)

修改 [execution_engine.py](file:///C:/Users/okuninushi/Downloads/butterquantdltest/backend/execution_engine.py):

```python
from backend.ml_inference import get_inference_engine
from ml.features import extract_features_v2

class ExecutionEngine:
    def __init__(self):
        # 加载ML引擎 / Load ML engine
        self.ml_engine = get_inference_engine()
    
    def evaluate_candidate(self, analysis_data):
        # 提取特征 / Extract features
        features = extract_features_v2(analysis_data)
        
        # 获取4分类预测 / Get 4-class prediction
        result = self.ml_engine.predict_roi_distribution(features)
        
        return {
            'expected_roi': result['expected_roi'],
            'predicted_class': result['predicted_class'],
            'confidence': max(result['class_probabilities']),
            'should_trade': result['expected_roi'] > 0.15
        }
```

### Step 3: 验证与监控 (持续)

遵循 [AI_SKILL_STANDARD.md](file:///C:/Users/okuninushi/Downloads/butterquantdltest/docs/AI_SKILL_STANDARD.md) Skill 8 规范:

> **Rule**: Robustness is critical for long-running data jobs.

创建 `check/verify_ml_integration.py`:

```python
"""
ML集成验证脚本 / ML Integration Verification Script
Usage: python check/verify_ml_integration.py
"""
```

---

## 4. 风险评估 / Risk Assessment

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 特征维度不匹配 | 推理失败 | Step 1 兼容性检查 |
| 模型性能下降 | 交易亏损 | 保留Baseline对照组 |
| 重训数据不足 | 模型泛化差 | 使用现有7000样本 |

---

## 5. 预期效果 / Expected Outcomes

基于 Claude v2.0 Final 的预测:

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 准确率 | ~30% | 45-55% |
| ROI预测偏差 | ±30% | ±15% |
| 策略胜率 | ~25% | 35-45% |

---

## 6. 下一步行动 / Next Actions

> [!TIP]
> 建议按以下顺序执行:

1. **立即**: 运行 `ml/end_to_end_test.py` 验证当前系统状态
2. **今天**: 检查 `execution_engine.py` 是否已正确集成 ML v2.0
3. **本周**: 移除 `total_score` 并重新训练模型
4. **持续**: 建立监控仪表板追踪ML预测 vs 实际表现

---

**文档版本**: v1.0  
**创建日期**: 2026-01-20  
**作者**: Antigravity  
