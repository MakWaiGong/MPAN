import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

# 设置中文字体和英文字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# 直接使用实际的实验对比数据（从命令行输出中硬编码）
traditional_vs_prob_data = {}

# 从实际实验输出的数据
# 传统多分类器: ACC=0.6041, BA=0.5535, ROC-AUC=0.7632, PR-AUC=0.5555, F1=0.5602, MCC=0.4442
# 概率重构分类器: ACC=0.8731, BA=0.8571, ROC-AUC=0.9453, PR-AUC=0.9006, F1=0.8677, MCC=0.8248

k_values = ['Experiment_Result']  # 单次实验结果

traditional_vs_prob_data['Experiment_Result'] = {
    'Traditional': [0.6041, 0.5535, 0.7632, 0.5555, 0.5602, 0.4442],
    'Probability_Reconstruction': [0.8731, 0.8571, 0.9453, 0.9006, 0.8677, 0.8248]
}

print("使用实际实验对比数据:")
print("传统多分类器: ACC=0.6041, BA=0.5535, ROC-AUC=0.7632, PR-AUC=0.5555, F1=0.5602, MCC=0.4442")
print("概率重构分类器: ACC=0.8731, BA=0.8571, ROC-AUC=0.9453, PR-AUC=0.9006, F1=0.8677, MCC=0.8248")

print("创建的传统vs概率重构对比数据:")
for k, methods in traditional_vs_prob_data.items():
    print(f"k={k}:")
    for method, values in methods.items():
        print(f"  {method}: {values}")

# 六个指标
metrics = ['Accuracy', 'Balanced_Accuracy', 'ROC_AUC', 'PR_AUC', 'F1', 'MCC']
metric_labels = ['Accuracy', 'Balanced\nAccuracy', 'ROC-AUC', 'PR-AUC', 'F1-Score', 'MCC']
num_metrics = len(metrics)

# 颜色设置
method_colors = {
    'Traditional': '#FF6B6B',
    'Probability_Reconstruction': '#4ECDC4'
}

# 创建雷达图的函数
def create_classifier_comparison_radar(data, filename):
    """创建分类器对比雷达图"""
    # 计算角度
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]  # 闭合图形

    # 创建子图
    fig, axes = plt.subplots(1, len(k_values), figsize=(20, 6), subplot_kw=dict(projection='polar'))
    if len(k_values) == 1:
        axes = [axes]

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        methods_data = data[k]

        # 绘制每个方法的雷达图
        for method_name, values in methods_data.items():
            # 对MCC进行归一化处理
            display_values = []
            for i, val in enumerate(values):
                if metrics[i] == 'MCC':
                    # 将MCC从[-1,1]映射到[0,1]: (x + 1) / 2
                    display_val = (val + 1) / 2
                else:
                    display_val = val
                display_values.append(display_val)

            display_values += display_values[:1]  # 闭合图形

            ax.plot(angles, display_values, 'o-', linewidth=3,
                   label=method_name.replace('_', ' '),
                   color=method_colors[method_name], markersize=8, alpha=0.8)
            ax.fill(angles, display_values, alpha=0.1, color=method_colors[method_name])

        # 添加指标标签
        labels_with_ranges = []
        for i, metric in enumerate(metrics):
            if metric == 'MCC':
                labels_with_ranges.append(f'{metric_labels[i]}\n(Mapped)')
            else:
                labels_with_ranges.append(f'{metric_labels[i]}\n(0-1)')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_with_ranges, fontsize=8, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=6)
        ax.set_title(f'Classifier Performance Comparison\n(Actual Experiment Results)', fontsize=12, fontweight='bold', pad=20)

    # 添加图例
    axes[-1].legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=10, frameon=False)

    # 设置总标题
    fig.suptitle('Traditional Multi-Classifier vs Probability Reconstruction\nPerformance Comparison (Actual Experiment Results)',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()

# 主程序 - 生成雷达图
if __name__ == "__main__":
    # 创建plot目录
    plot_dir = r'D:\document\code\IMC\result\plot'  # 绝对路径
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot目录: {os.path.abspath(plot_dir)}")

    # 生成分类器对比雷达图
    filename = os.path.join(plot_dir, 'classifier_comparison_radar_chart.svg')
    create_classifier_comparison_radar(traditional_vs_prob_data, filename)
    print(f'分类器对比雷达图已保存: {filename}')

    print('分类器对比可视化创建完成！')

    # 生成性能差异分析
    print('\n' + '='*80)
    print('性能差异分析')
    print('='*80)

    methods_data = traditional_vs_prob_data['Experiment_Result']
    trad_values = methods_data['Traditional']
    prob_values = methods_data['Probability_Reconstruction']

    print('\n实际实验结果对比:')
    print('方法                    ACC      BA       ROC      PR       F1       MCC')
    print('-' * 73)
    print('<15'
          '<8.4f'
          '<8.4f'
          '<8.4f'
          '<8.4f'
          '<8.4f')

    # 计算差异
    differences = np.array(prob_values) - np.array(trad_values)
    print('性能提升:          ' +
          '<8.4f' * len(differences))

    # 计算相对提升
    relative_improvements = (np.array(differences) / np.array(trad_values)) * 100
    print('相对提升(%):       ' +
          '<8.1f' * len(relative_improvements))

    print('\n结论: 概率重构分类器在所有指标上都显著优于传统多分类器')
    print('特别是在PR-AUC和MCC上的提升最为明显')
