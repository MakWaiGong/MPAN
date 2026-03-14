import numpy as np
import matplotlib.pyplot as plt
import os

# 硬编码数据 - 所有16个模态的ACC和ROC-AUC结果
modal_performance_data = {
    'M1': {'accuracy': 0.4213197969543147, 'roc_auc': 0.6717976089779656},
    'M2': {'accuracy': 0.40609137055837563, 'roc_auc': 0.6594849355390452},
    'M3': {'accuracy': 0.38071065989847713, 'roc_auc': 0.6227407820589561},
    'M4': {'accuracy': 0.41116751269035534, 'roc_auc': 0.6069477256796082},
    'M5': {'accuracy': 0.39593908629441626, 'roc_auc': 0.6936835621065806},
    'M6': {'accuracy': 0.47715736040609136, 'roc_auc': 0.6746797153024911},
    'M7': {'accuracy': 0.41624365482233505, 'roc_auc': 0.694657839834076},
    'M8': {'accuracy': 0.36548223350253806, 'roc_auc': 0.5598915628325238},
    'M9': {'accuracy': 0.38578680203045684, 'roc_auc': 0.6486615690488652},
    'M10': {'accuracy': 0.36548223350253806, 'roc_auc': 0.5877736981221892},
    'M11': {'accuracy': 0.3756345177664975, 'roc_auc': 0.6223803100336724},
    'M12': {'accuracy': 0.38071065989847713, 'roc_auc': 0.6312199852289272},
    'M13': {'accuracy': 0.47766497461928935, 'roc_auc': 0.7222492682926829},
    'M14': {'accuracy': 0.41116751269035534, 'roc_auc': 0.6604585169959022},
    'M15': {'accuracy': 0.41116751269035534, 'roc_auc': 0.6922774617410875},
    'M16': {'accuracy': 0.36548223350253806, 'roc_auc': 0.5621196757825755}
}

# 定义要绘制的指标
metrics = ['accuracy', 'roc_auc']
metric_labels = ['ACC', 'ROC-AUC']

# 准备数据
modal_indices = [f'M{i}' for i in range(1, 17)]
modal_data = {}
for modal in modal_indices:
    modal_data[modal] = [modal_performance_data[modal][metric] for metric in metrics]

print(f"显示所有16个模态的性能对比")

# 设置颜色 - 使用不同的颜色区分模态，使用更多的颜色映射
n_modals = len(modal_indices)
# 使用viridis颜色映射，提供更多不同的颜色
colors = plt.cm.viridis(np.linspace(0, 1, n_modals))

# 设置字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# 创建子图 - 为每个指标创建单独的子图
fig, axes = plt.subplots(1, 2, figsize=(36, 12))
fig.suptitle('Modal Importance Analysis (k=1)', fontsize=32, fontweight='bold', y=0.95)

max_value = 0.0

# 为每个指标绘制柱状图
for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[metric_idx]
    x = np.arange(n_modals)  # 16个模态的位置
    width = 0.8  # 柱子宽度

    values = [modal_data[modal][metric_idx] for modal in modal_indices]
    bars = ax.bar(x, values, width, color=colors, alpha=0.9, edgecolor='none')

    max_value = max(max_value, max(values))

    # 在柱子上方添加数值标签
    for i, (pos, val) in enumerate(zip(x, values)):
        ax.text(
            pos,
            val + 0.005,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )

    # 设置子图标签和标题
    ax.set_xlabel('Modal Index', fontsize=24, fontweight='bold')
    ax.set_ylabel('Score', fontsize=24, fontweight='bold', labelpad=10)
    ax.set_title(f'{label} Performance', fontsize=28, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'M{i}' for i in range(1, 17)], fontsize=18)

    # 设置Y轴范围
    upper_ylim = min(1.0, max_value + 0.08)
    ax.set_ylim(0, upper_ylim)
    yticks = np.arange(0, upper_ylim + 0.001, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks], fontsize=20)

    # 移除网格和边框
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 创建统一的图例
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], edgecolor='none', alpha=0.9)
                  for i in range(n_modals)]
fig.legend(legend_elements, modal_indices, loc='center right',
           bbox_to_anchor=(1.02, 0.5), fontsize=16, frameon=False, ncol=2)

# 创建统一的图例
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], edgecolor='none', alpha=0.9)
                  for i in range(n_modals)]
fig.legend(legend_elements, modal_indices, loc='center right',
           bbox_to_anchor=(1.02, 0.5), fontsize=16, frameon=False, ncol=2)

# 调整布局
plt.tight_layout(rect=[0, 0, 0.92, 0.95])

# 创建输出目录
output_dir = r'D:\document\code\IMC\result\plot'
os.makedirs(output_dir, exist_ok=True)

# 保存图表
plt.savefig(os.path.join(output_dir, 'modal_importance_bar_plot.svg'), format='svg', dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'modal_importance_bar_plot.png'), format='png', dpi=300, bbox_inches='tight')

# 打印统计信息
print(f"模态重要性柱状图已保存到: {output_dir}")
print("文件: modal_importance_bar_plot.svg 和 modal_importance_bar_plot.png")

# 打印每个指标的最佳模态
print("\n各指标表现最佳的模态:")
for i, metric in enumerate(metrics):
    best_modal = max(modal_data.keys(), key=lambda m: modal_data[m][i])
    best_value = modal_data[best_modal][i]
    print(f"{metric_labels[i]}: {best_modal} ({best_value:.3f})")

plt.show()
