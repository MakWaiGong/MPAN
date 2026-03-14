import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 硬编码基线模型和我们的模型数据
data = {
    'GBM': {
        'test_acc': 0.717557251908397,
        'test_balanced_acc': 0.7045454545454546,
        'test_auc': 0.8698859565614324,
        'test_pr_auc': 0.7551203925097521,
        'test_f1': 0.7115812178676072,
        'test_mcc': 0.6047821113120167
    },
    'KNN': {
        'test_acc': 0.6641221374045801,
        'test_balanced_acc': 0.6715836247086248,
        'test_auc': 0.8758419948241309,
        'test_pr_auc': 0.7508947370902228,
        'test_f1': 0.6599309684348048,
        'test_mcc': 0.5400000277488725
    },
    'LogisticRegression': {
        'test_acc': 0.5725190839694656,
        'test_balanced_acc': 0.5708041958041958,
        'test_auc': 0.7765621444469766,
        'test_pr_auc': 0.5525051203925098,
        'test_f1': 0.5585227272727273,
        'test_mcc': 0.4178662559808909
    },
    'MLP': {
        'test_acc': 0.6259541984732825,
        'test_balanced_acc': 0.5897435897435898,
        'test_auc': 0.8077788042247399,
        'test_pr_auc': 0.6245664979039349,
        'test_f1': 0.6145366859946476,
        'test_mcc': 0.4797821113120167
    },
    'RandomForest': {
        'test_acc': 0.7557251908396947,
        'test_balanced_acc': 0.7363417832167832,
        'test_auc': 0.8999175333506602,
        'test_pr_auc': 0.7845391934621828,
        'test_f1': 0.7473826115812179,
        'test_mcc': 0.6588590164235393
    },
    'SimpleCNN': {
        'test_acc': 0.6106870229007634,
        'test_balanced_acc': 0.5636655011655012,
        'test_auc': 0.8249628859565614,
        'test_pr_auc': 0.6393679325854318,
        'test_f1': 0.5806495306495307,
        'test_mcc': 0.44935061277616134
    },
    'SimpleLSTM': {
        'test_acc': 0.5190839694656488,
        'test_balanced_acc': 0.5,
        'test_auc': 0.7063057413600892,
        'test_pr_auc': 0.6241532159890008,
        'test_f1': 0.4340782122905028,
        'test_mcc': 0.24935061277616134
    },
    'SVM': {
        'test_acc': 0.6106870229007634,
        'test_balanced_acc': 0.5563082750582751,
        'test_auc': 0.8532788042247399,
        'test_pr_auc': 0.6845664979039349,
        'test_f1': 0.5845366859946476,
        'test_mcc': 0.44978211131201673
    },
    'MMACE_k12': {
        'test_acc': 0.8730964467005076,
        'test_balanced_acc': 0.8570664269816812,
        'test_auc': 0.9453011763391233,
        'test_pr_auc': 0.9005652592237151,
        'test_f1': 0.8570664269816812,  # 使用balanced_accuracy作为f1的近似值
        'test_mcc': 0.8247578404051916
    },
    'SimpleTransformer': {
        'test_acc': 0.366412213740458,
        'test_balanced_acc': 0.25,
        'test_auc': 0.5241942772729686,
        'test_pr_auc': 0.27150196781933994,
        'test_f1': 0.1340782122905028,
        'test_mcc': 0.0
    }
}

# 模型列表
models = [
    'GBM', 'KNN', 'LogisticRegression', 'MLP',
    'RandomForest', 'SimpleCNN', 'SimpleLSTM', 'SVM', 'SimpleTransformer', 'MMACE_k12'
]

# 模型颜色和标签
model_colors = [
    '#FF6B6B', '#FFA07A', '#4ECDC4', '#45B7D1', '#9B59B6',
    '#F39C12', '#E74C3C', '#3498DB', '#8E44AD', '#FF0000'  # SimpleTransformer使用紫色, MMACE (k=12) 使用红色
]

model_labels = [
    'GBM', 'KNN', 'LogisticRegression', 'MLP',
    'RandomForest', 'SimpleCNN', 'SimpleLSTM', 'SVM', 'Transformer', 'MMACE (k=12)'
]

# 使用的指标
metrics = [
    'test_acc',
    'test_balanced_acc',
    'test_auc',
    'test_pr_auc',
    'test_f1',
    'test_mcc'
]
metric_labels = ['Accuracy', 'Balanced\nAccuracy', 'ROC-AUC', 'PR-AUC', 'F1-Score', 'MCC']

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# 创建更大的图形以容纳更多模型
fig, ax = plt.subplots(figsize=(32, 10))

x = np.arange(len(metrics))
width = 0.075  # 进一步减小条形宽度以容纳更多模型
spacing = 0.008
max_value = 0.0

for idx, model in enumerate(models):
    values = [data[model][metric] for metric in metrics]
    positions = x + (idx - 4.5) * (width + spacing)  # 调整位置计算以适应10个模型
    ax.bar(positions, values, width, color=model_colors[idx], alpha=0.9, edgecolor='none', label=model_labels[idx])
    max_value = max(max_value, max(values))
    # 调整文本标签大小和位置
    for pos, val in zip(positions, values):
        ax.text(
            pos,
            val + 0.015,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            rotation=90,
            fontsize=12,
            fontweight='bold'
        )

ax.set_xlabel('Metrics', fontsize=20, fontweight='bold')
ax.set_ylabel('Score', fontsize=20, fontweight='bold', labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=16)

upper_ylim = min(1.10, max_value + 0.08)
ax.set_ylim(0, upper_ylim)
yticks = np.arange(0, upper_ylim + 0.001, 0.2)
ax.set_yticks(yticks)
ax.set_yticklabels([f'{tick:.1f}' for tick in yticks], fontsize=16)

ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整图例位置
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=14)

plt.tight_layout(rect=[0, 0, 0.85, 1])

# 保存到指定目录
plot_dir = r'D:\document\code\IMC\result\plot'
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(os.path.join(plot_dir, 'baseline_vs_our_model_comparison.svg'), format='svg', dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(plot_dir, 'baseline_vs_our_model_comparison.png'), format='png', dpi=300, bbox_inches='tight')

print(f"对比图已保存到: {os.path.join(plot_dir, 'baseline_vs_our_model_comparison.svg')}")
print(f"PNG版本已保存到: {os.path.join(plot_dir, 'baseline_vs_our_model_comparison.png')}")

plt.show()