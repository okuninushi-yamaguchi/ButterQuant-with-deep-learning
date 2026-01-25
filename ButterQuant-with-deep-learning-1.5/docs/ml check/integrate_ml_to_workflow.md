"""
ButterQuant ML集成示例 - 在现有工作流中使用训练好的模型
"""

from ml.ml_inference import get_inference_engine
from ml.features import extract_features_v2

class StrategyEvaluator:
    """策略评估器 - 集成ML模型"""
    
    def __init__(self):
        # 加载训练好的模型(只需要加载一次)
        self.ml_engine = get_inference_engine()
        print("✅ ML模型已加载")
    
    def evaluate_butterfly_strategy(self, analysis_data):
        """
        评估单个蝴蝶策略
        
        参数:
            analysis_data: 您现有的策略分析数据字典
                需要包含: butterfly, greeks, risk, market, score等字段
        
        返回:
            dict: {
                'should_trade': bool,  # 是否应该交易
                'ml_score': float,     # ML评分 (0-1)
                'predicted_class': str, # 预测类别
                'expected_roi': float,  # 期望ROI
                'confidence': float     # 置信度
            }
        """
        
        # 1. 提取特征
        features = extract_features_v2(analysis_data)
        
        # 2. ML预测
        result = self.ml_engine.predict_roi_distribution(features)
        
        # 3. 决策逻辑 (根据您的风险偏好调整)
        should_trade = self._make_decision(result)
        
        return {
            'should_trade': should_trade,
            'ml_score': result['class_probabilities'][3],  # Excellent概率
            'predicted_class': self._get_class_name(result['predicted_class']),
            'expected_roi': result['expected_roi'],
            'confidence': max(result['class_probabilities']),
            'raw_probabilities': result['class_probabilities']
        }
    
    def _make_decision(self, ml_result):
        """
        决策规则 - 可根据实际情况调整
        
        三种策略可选:
        """
        
        # 策略1: 保守 - 只做模型非常确定是Excellent的
        # return ml_result['class_probabilities'][3] > 0.70
        
        # 策略2: 平衡 - 期望ROI为正即可
        # return ml_result['expected_roi'] > 0.05
        
        # 策略3: 激进 - Excellent或Good都可以
        return (ml_result['class_probabilities'][3] > 0.50 or 
                ml_result['class_probabilities'][2] > 0.60)
    
    def _get_class_name(self, class_idx):
        """类别索引转名称"""
        class_names = {0: 'Loss', 1: 'Minor', 2: 'Good', 3: 'Excellent'}
        return class_names.get(class_idx, 'Unknown')
    
    def batch_evaluate(self, candidates_list):
        """
        批量评估多个候选策略
        
        参数:
            candidates_list: 候选策略列表 [analysis_data1, analysis_data2, ...]
        
        返回:
            排序后的推荐列表(按ML评分从高到低)
        """
        results = []
        
        for candidate in candidates_list:
            try:
                eval_result = self.evaluate_butterfly_strategy(candidate)
                
                if eval_result['should_trade']:
                    results.append({
                        'strategy': candidate,
                        'ml_evaluation': eval_result
                    })
            except Exception as e:
                print(f"⚠️  评估失败: {e}")
                continue
        
        # 按ML评分排序
        results.sort(key=lambda x: x['ml_evaluation']['ml_score'], reverse=True)
        
        return results


# ============================================
# 使用示例
# ============================================

def example_usage():
    """完整使用示例"""
    
    # 初始化评估器(只需要初始化一次)
    evaluator = StrategyEvaluator()
    
    # 假设这是您现有系统生成的策略分析数据
    strategy_analysis = {
        'symbol': 'AAPL',
        'date': '2025-01-20',
        'butterfly': {
            'dte': 45,
            'lower_strike': 180,
            'center_strike': 185,
            'upper_strike': 190,
            'width': 5,
            'net_premium': 1.5,
            'max_profit': 3.5,
            'max_loss': 1.5,
            # ... 其他字段
        },
        'greeks': {
            'delta': 0.05,
            'gamma': 0.12,
            'theta': 0.02,
            'vega': -0.15,
            # ... 其他字段
        },
        'risk': {
            'max_loss_pct': 0.15,
            'profit_loss_ratio': 2.33,
            'breakeven_upper': 188.5,
            'breakeven_lower': 181.5,
        },
        'market': {
            'current_price': 185.0,
            'iv_rank': 45,
            'iv_percentile': 52,
        },
        'score': {
            'total_score': 75,
            # ... 其他字段
        }
    }
    
    # 单个策略评估
    print("\n=== 单个策略评估 ===")
    result = evaluator.evaluate_butterfly_strategy(strategy_analysis)
    
    print(f"是否交易: {result['should_trade']}")
    print(f"预测类别: {result['predicted_class']}")
    print(f"ML评分: {result['ml_score']:.2%}")
    print(f"期望ROI: {result['expected_roi']:.2%}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"各类别概率: Loss={result['raw_probabilities'][0]:.2%}, "
          f"Minor={result['raw_probabilities'][1]:.2%}, "
          f"Good={result['raw_probabilities'][2]:.2%}, "
          f"Excellent={result['raw_probabilities'][3]:.2%}")
    
    # 批量评估示例
    print("\n=== 批量评估 ===")
    candidates = [strategy_analysis] * 5  # 假设有5个候选
    recommended = evaluator.batch_evaluate(candidates)
    
    print(f"推荐策略数量: {len(recommended)}")
    for i, rec in enumerate(recommended[:3], 1):
        print(f"\n推荐 #{i}:")
        print(f"  ML评分: {rec['ml_evaluation']['ml_score']:.2%}")
        print(f"  期望ROI: {rec['ml_evaluation']['expected_roi']:.2%}")


if __name__ == "__main__":
    example_usage()

TXT workflow

明白了！模型训练完成后，让我帮您规划清楚**实际使用**和**日常维护**的流程。

## 🎯 下一步要做的事

### **Step 1: 验证模型能用 (10分钟)**

```bash
# 测试推理引擎是否正常工作
python ml/ml_inference.py

# 应该看到类似输出:
# ✅ 模型加载成功
# ✅ 预测: Class 3 (Excellent), Prob: 0.85
# ✅ Expected ROI: 18.5%
```

**如果报错:**
- 缺少ONNX: `pip install onnx onnxruntime`
- 找不到模型文件: 检查 `ml/models/` 目录下是否有 `.pth` 或 `.onnx` 文件

---

### **Step 2: 把模型集成到您的实际交易流程**

您现在应该有一个**选股/选策略**的地方,在那里加入ML评分:---

## 📅 日常维护流程

### **每天要做的事: 无**

**模型不需要每天重新训练**。已训练好的模型可以直接用,就像一个固定的评分函数。

---

### **每周/每月要做的事:**---

## 📊 完整维护时间表

| 频率 | 要做的事 | 耗时 | 自动化 |
|------|---------|------|--------|
| **每天** | 无需任何操作 | 0分钟 | - |
| **每周** | 运行健康检查,查看周报 | 2分钟 | 可设置cron定时任务 |
| **每月** | 检查30天表现,决定是否重训练 | 5分钟 | 可自动 |
| **每季度** | 如果模型表现下降,重新训练 | 10分钟 | 手动 |

---

## 🎯 重新训练模型的时机

**只在以下情况需要重新训练:**

1. ✅ **准确率下降**: 最近30天实际准确率 < 55%
2. ✅ **盈利率下降**: 模型推荐的策略盈利率 < 40%
3. ✅ **市场环境变化**: 波动率制度明显改变 (如VIX从20→60)
4. ✅ **积累新数据**: 新增500+真实交易样本

**不需要重新训练的情况:**
- ❌ 每天都训练 (完全没必要)
- ❌ 模型表现正常 (准确率>60%, 盈利率>50%)
- ❌ 只有少量新数据 (<100个样本)

---

## 🚀 立即可执行的步骤

**今天完成:**
1. 把 `integrate_ml_to_workflow.py` 复制到您的项目
2. 在您选股/评估的地方调用 `StrategyEvaluator`
3. 运行一次测试,看看预测结果

**本周设置:**
1. 在您的交易系统中集成 `log_prediction()` (开仓时)
2. 在您的交易系统中集成 `update_actual_roi()` (平仓时)
3. 设置每周日晚上自动运行健康检查

**下周开始:**
- 模型自动运行,您只需要每周看一眼周报
- 如果周报提示需要重新训练,再运行 `python ml/train_model_fixed.py`

---

需要我帮您写具体集成到您现有代码的部分吗?比如您现在选股的代码在哪个文件?