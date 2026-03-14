#!/usr/bin/env python3
"""
解释概率重构分类器ROC曲线绘制逻辑
"""

import numpy as np

def explain_roc_logic():
    """解释概率重构分类器的ROC曲线是如何绘制的"""

    print("="*70)
    print("概率重构分类器ROC曲线绘制逻辑详解")
    print("="*70)

    # 模拟数据
    n_samples = 10
    true_labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])  # 4个类别的样本
    predicted_probs = np.array([
        [0.8, 0.1, 0.05, 0.05],  # 预测为类别0，概率最高
        [0.1, 0.7, 0.1, 0.1],    # 预测为类别1，概率最高
        [0.1, 0.1, 0.6, 0.2],    # 预测为类别2，概率最高
        [0.05, 0.05, 0.1, 0.8],  # 预测为类别3，概率最高
        [0.6, 0.2, 0.1, 0.1],    # 预测为类别0，概率最高
        [0.2, 0.6, 0.1, 0.1],    # 预测为类别1，概率最高
        [0.1, 0.1, 0.7, 0.1],    # 预测为类别2，概率最高
        [0.1, 0.1, 0.1, 0.7],    # 预测为类别3，概率最高
        [0.5, 0.3, 0.1, 0.1],    # 预测为类别0，概率最高
        [0.1, 0.8, 0.05, 0.05]   # 预测为类别1，概率最高
    ])

    class_names = ['a', 'b', 'c', 'd']

    print("\n原始数据:")
    print(f"样本数量: {n_samples}")
    print(f"真实标签: {true_labels}")
    print("预测概率矩阵 (每行是一个样本的[P(a), P(b), P(c), P(d)]):")
    for i, (label, probs) in enumerate(zip(true_labels, predicted_probs)):
        print(".1f")

    print("
" + "="*70)
    print("ROC曲线绘制逻辑 (One-vs-Rest策略)")
    print("="*70)

    for class_idx in range(4):
        print(f"\n🎯 为类别 '{class_names[class_idx]}' 绘制ROC曲线:")
        print("-" * 40)

        # One-vs-Rest: 当前类别 vs 其他所有类别
        binary_true = (true_labels == class_idx).astype(int)
        binary_scores = predicted_probs[:, class_idx]

        print("  样本重新标记 (One-vs-Rest):")
        for i, (true_label, score, is_positive) in enumerate(zip(true_labels, binary_scores, binary_true)):
            status = "✓ 正类" if is_positive else "✗ 负类"
            print(".1f")

        print("\n  ROC曲线使用的数据:")
        print(f"    - 正类样本数量: {np.sum(binary_true)}")
        print(f"    - 负类样本数量: {len(binary_true) - np.sum(binary_true)}")
        print(f"    - 分数范围: [{binary_scores.min():.3f}, {binary_scores.max():.3f}]")

        # 模拟ROC计算
        sorted_indices = np.argsort(binary_scores)[::-1]  # 从高到低排序
        print(f"    - 按分数排序后的样本: {binary_scores[sorted_indices]}")
        print(f"    - 对应的真实标签: {binary_true[sorted_indices]}")

    print("
" + "="*70)
    print("关键洞察")
    print("="*70)

    print("\n🤔 为什么概率重构分类器还能画四条ROC曲线？")
    print("\n虽然概率重构分类器最终输出的是一个多分类概率分布 [P(a), P(b), P(c), P(d)]，")
    print("但是ROC曲线的绘制使用的是 'One-vs-Rest' (OvR) 策略：")
    print("\n1️⃣ 为每个类别单独构建二分类问题：")
    print("   - 类别a的ROC： '是否为a类型？' vs '不是a类型'")
    print("   - 类别b的ROC： '是否为b类型？' vs '不是b类型'")
    print("   - 以此类推...")
    print("\n2️⃣ 使用对应的概率作为决策分数：")
    print("   - 类别a的ROC使用 P(a) 作为分数")
    print("   - 类别b的ROC使用 P(b) 作为分数")
    print("   - 类别c的ROC使用 P(c) 作为分数")
    print("   - 类别d的ROC使用 P(d) 作为分数")
    print("\n3️⃣ 这样就能为每个类别绘制独立的ROC曲线！")
    print("\n🎯 核心思想：")
    print("多分类器的概率输出可以通过One-vs-Rest策略转换为多个ROC曲线，")
    print("每个类别都有自己的'信心分数'和对应的二分类性能评估。")

if __name__ == "__main__":
    explain_roc_logic()
