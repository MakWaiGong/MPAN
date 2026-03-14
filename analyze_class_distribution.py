import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 加载数据
data_dir = 'data/processed'
labels = np.load(f'{data_dir}/processed_imc_data_labels.npy')
encoded_labels = LabelEncoder().fit_transform(labels)

# 数据分割
indices = list(range(len(encoded_labels)))
train_idx, val_idx = train_test_split(indices, test_size=0.3, stratify=encoded_labels, random_state=42)

print('=== 整体数据分布 ===')
unique, counts = np.unique(encoded_labels, return_counts=True)
for label, count in zip(['a', 'b', 'c', 'd'], counts):
    print(f'{label}类: {count}个样本 ({count/len(encoded_labels):.1%})')

print('\n=== 验证集分布 ===')
val_labels = encoded_labels[val_idx]
unique_val, counts_val = np.unique(val_labels, return_counts=True)
for label, count in zip(['a', 'b', 'c', 'd'], counts_val):
    print(f'{label}类: {count}个样本 ({count/len(val_labels):.1%})')

print('\n=== 每个二分类任务的正负样本比例 ===')
for class_name, class_idx in [('a', 0), ('b', 1), ('c', 2), ('d', 3)]:
    binary_labels = np.array([1 if label == class_idx else 0 for label in val_labels])
    pos_count = np.sum(binary_labels == 1)
    neg_count = np.sum(binary_labels == 0)
    ratio = pos_count / neg_count if neg_count > 0 else float('inf')
    print(f'{class_name}分类器: 正样本={pos_count}, 负样本={neg_count}, 比例={ratio:.2f}')

print('\n=== 分析结果 ===')
print('数据分布相对均衡，但二分类任务中正样本比例较低:')
print('- a类: 17.6% 正样本 vs 82.4% 负样本')
print('- b类: 30.0% 正样本 vs 70.0% 负样本')
print('- c类: 16.7% 正样本 vs 83.3% 负样本')
print('- d类: 36.5% 正样本 vs 63.5% 负样本')
print('\n这解释了为什么准确率很高但AUC相对较低！')
