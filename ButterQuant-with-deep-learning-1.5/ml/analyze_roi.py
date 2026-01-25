"""
修正标注逻辑 - 基于实际ROI分布调整阈值

步骤:
1. 分析现有数据的ROI分布
2. 重新定义更合理的阈值
3. 重新标注数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_roi_distribution(parquet_path):
    """分析ROI分布，找到合理的阈值"""
    
    df = pd.read_parquet(parquet_path)
    
    # 假设有 _debug_roi 字段 (如果没有需要重新计算)
    if '_debug_roi' not in df.columns:
        print("⚠️ 数据中没有ROI字段，需要重新生成")
        return None
    
    roi = df['_debug_roi'].values
    
    print("=" * 60)
    print("ROI分布分析")
    print("=" * 60)
    
    # 基础统计
    print(f"\n基础统计:")
    print(f"  样本数: {len(roi)}")
    print(f"  均值: {np.mean(roi):.2%}")
    print(f"  中位数: {np.median(roi):.2%}")
    print(f"  标准差: {np.std(roi):.2%}")
    print(f"  最小值: {np.min(roi):.2%}")
    print(f"  最大值: {np.max(roi):.2%}")
    
    # 分位数
    print(f"\n分位数:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(roi, p)
        print(f"  P{p}: {val:.2%}")
    
    # 当前标注统计
    current_labels = df['label'].value_counts().sort_index()
    print(f"\n当前标签分布:")
    for label, count in current_labels.items():
        pct = count / len(df) * 100
        print(f"  Class {label}: {count} ({pct:.1f}%)")
    
    # 绘制分布图
    plt.figure(figsize=(12, 4))
    
    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(roi, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', label='ROI=0')
    plt.axvline(0.10, color='orange', linestyle='--', label='ROI=10%')
    plt.axvline(0.30, color='green', linestyle='--', label='ROI=30%')
    plt.xlabel('ROI')
    plt.ylabel('Frequency')
    plt.title('ROI Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 累积分布
    plt.subplot(1, 2, 2)
    sorted_roi = np.sort(roi)
    cumulative = np.arange(1, len(sorted_roi) + 1) / len(sorted_roi)
    plt.plot(sorted_roi, cumulative, linewidth=2)
    plt.axvline(0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(0.10, color='orange', linestyle='--', alpha=0.5)
    plt.axvline(0.30, color='green', linestyle='--', alpha=0.5)
    plt.xlabel('ROI')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative ROI Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml/roi_distribution_analysis.png', dpi=150)
    print("\n📊 分布图已保存: ml/roi_distribution_analysis.png")
    
    # 推荐阈值
    print("\n" + "=" * 60)
    print("推荐的新阈值 (基于分位数):")
    print("=" * 60)
    
    p25 = np.percentile(roi, 25)
    p50 = np.percentile(roi, 50)
    p75 = np.percentile(roi, 75)
    
    print(f"\n方案A (四分位数法):")
    print(f"  亏损:  ROI < {p25:.2%}")
    print(f"  微利:  {p25:.2%} ≤ ROI < {p50:.2%}")
    print(f"  良好:  {p50:.2%} ≤ ROI < {p75:.2%}")
    print(f"  优秀:  ROI ≥ {p75:.2%}")
    print(f"  预期分布: 25% / 25% / 25% / 25%")
    
    # 更实用的方案
    t1 = -0.05  # 亏损超过5%
    t2 = 0.10   # 盈利10%
    t3 = 0.25   # 盈利25%
    
    print(f"\n方案B (实用阈值法):")
    print(f"  亏损:  ROI < {t1:.0%}")
    print(f"  微利:  {t1:.0%} ≤ ROI < {t2:.0%}")
    print(f"  良好:  {t2:.0%} ≤ ROI < {t3:.0%}")
    print(f"  优秀:  ROI ≥ {t3:.0%}")
    
    c0 = (roi < t1).sum()
    c1 = ((roi >= t1) & (roi < t2)).sum()
    c2 = ((roi >= t2) & (roi < t3)).sum()
    c3 = (roi >= t3).sum()
    
    print(f"  预期分布: {c0} / {c1} / {c2} / {c3}")
    print(f"  百分比: {c0/len(roi)*100:.1f}% / {c1/len(roi)*100:.1f}% / {c2/len(roi)*100:.1f}% / {c3/len(roi)*100:.1f}%")
    
    return {
        'method_a': (p25, p50, p75),
        'method_b': (t1, t2, t3)
    }


def relabel_dataset(parquet_path, thresholds, output_path=None):
    """
    使用新阈值重新标注数据
    
    参数:
        thresholds: (t1, t2, t3) - 三个阈值
    """
    df = pd.read_parquet(parquet_path)
    
    if '_debug_roi' not in df.columns:
        print("❌ 缺少ROI字段，无法重新标注")
        return
    
    roi = df['_debug_roi'].values
    t1, t2, t3 = thresholds
    
    # 重新标注
    new_labels = np.zeros(len(roi), dtype=int)
    new_labels[roi < t1] = 0
    new_labels[(roi >= t1) & (roi < t2)] = 1
    new_labels[(roi >= t2) & (roi < t3)] = 2
    new_labels[roi >= t3] = 3
    
    df['label'] = new_labels
    
    # 统计
    print("\n" + "=" * 60)
    print("重新标注结果")
    print("=" * 60)
    
    old_dist = df['label'].value_counts().sort_index()
    print("\n新的标签分布:")
    for label, count in old_dist.items():
        pct = count / len(df) * 100
        print(f"  Class {label}: {count} ({pct:.1f}%)")
    
    # 保存
    if output_path is None:
        pass # output_path = parquet_path (overwrite)
    
    # 为了安全，这里覆盖原文件，但建议先备份
    # user wants to train on this, so overwriting is the standard way or pointing train_model to new file.
    # The prompt says "Please use new data file to retrain".
    # I'll overwrite 'training_data_deep.parquet' directly if output_path is None, 
    # OR I'll save to 'training_data_deep_relabeled.parquet' and update train script?
    # Claude's script said "relabel_dataset(parquet_path, thresholds)" which defaults to _relabeled.parquet.
    # But then said "Please use new data file to retrain". 
    # I will modify main to default overwrite or output to _relabeled and print instructions.
    
    if output_path:
        save_path = output_path
    else:
        save_path = str(parquet_path).replace('.parquet', '_relabeled.parquet')

    df.to_parquet(save_path, index=False)
    print(f"\n✅ 已保存到: {save_path}")
    
    return save_path


if __name__ == "__main__":
    parquet_path = "ml/training_data_deep.parquet"
    
    if not Path(parquet_path).exists():
        print(f"❌ 文件不存在: {parquet_path}")
        exit()

    # 1. 分析分布
    thresholds_dict = analyze_roi_distribution(parquet_path)
    
    if thresholds_dict is None:
        exit()

    # 2. 自动选择方案B (实用阈值法) - 因为这是全自动脚本
    print("\n" + "=" * 60)
    print("自动选择方案B (实用阈值法) 进行重标注 ...")
    print("=" * 60)
    
    thresholds = thresholds_dict['method_b']
    
    # 重新标注并覆盖/保存
    # 既然是全自动流程，为了方便可以直接覆盖，或者保存为新文件然后我改训练脚本路径。
    # 比较稳妥的是保存为新文件，然后改训练脚本读取这个新文件。
    
    new_path = relabel_dataset(parquet_path, thresholds)
    
    print(f"\n✅ 数据已准备好: {new_path}")
    print("建议修改 ml/train_model.py 读取此新文件进行训练。")
