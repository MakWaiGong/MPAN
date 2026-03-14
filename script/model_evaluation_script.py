#!/usr/bin/env python3
#!/usr/bin/env python3
"""
模型评估脚本 - 用于测试训练好的IMC分类模型

功能：
1. 加载训练好的传统多分类器和二分类器
2. 在验证集上评估模型性能
3. 计算各种分类指标 (准确率、ROC-AUC、F1等)
4. 生成详细的ROC曲线可视化图表
5. 输出性能对比结果

输出文件位置：
- 结果文件：result/inference/model_evaluation_results.json
- ROC曲线：result/inference/roc_curves_comparison.svg/png

使用方法：
python script/model_evaluation_script.py

作者：AI Assistant
日期：2025年1月
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_curve, auc, f1_score, roc_curve
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
from tqdm import tqdm
import json
import os
from datetime import datetime

class SingleModalNet(nn.Module):
    """单模态网络"""
    def __init__(self, sequence_length=81):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )

    def forward(self, x):
        return self.feature_extractor(x).flatten(1)

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x
        )
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class TransformerAttentionFusion(nn.Module):
    """基于Transformer的注意力融合"""
    def __init__(self, feature_dim, num_modals, num_layers=8, num_heads=2, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modals = num_modals
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.fusion_weights = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, modal_features):
        batch_size, num_modals, feature_dim = modal_features.shape

        for layer in self.transformer_layers:
            modal_features = layer(modal_features)

        attention_weights = self.fusion_weights(modal_features).squeeze(-1)
        fused_features = torch.sum(attention_weights.unsqueeze(-1) * modal_features, dim=1)

        return fused_features, attention_weights

class AttentionFusion(nn.Module):
    """注意力融合"""
    def __init__(self, feature_dim, num_modals):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4), nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim), nn.ReLU()
        )

    def forward(self, modal_features):
        attention_weights = F.softmax(self.attention(modal_features), dim=1)
        transformed_features = self.feature_transform(modal_features)
        fused_features = torch.sum(attention_weights * transformed_features, dim=1)
        return fused_features, attention_weights.squeeze(-1)

class ClassifierNet(nn.Module):
    """多模态分类器网络"""
    def __init__(self, num_classes, num_modals, sequence_length=81, use_transformer_attention=False):
        super().__init__()
        self.num_modals = num_modals
        self.modal_extractors = nn.ModuleList([
            SingleModalNet(sequence_length) for _ in range(num_modals)
        ])
        self.feature_dim = 256 * 8

        if use_transformer_attention:
            self.attention_fusion = TransformerAttentionFusion(self.feature_dim, num_modals)
        else:
            self.attention_fusion = AttentionFusion(self.feature_dim, num_modals)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, modal_data, return_features=False):
        modal_features = []
        for i in range(self.num_modals):
            single_modal = modal_data.flatten(1).unsqueeze(1)
            features = self.modal_extractors[i](single_modal)
            modal_features.append(features)

        modal_features = torch.stack(modal_features, dim=1)
        fused_features, attention_weights = self.attention_fusion(modal_features)
        output = self.classifier(fused_features)

        return (output, attention_weights, fused_features) if return_features else (output, attention_weights)

class ProbabilityReconstructor:
    """概率重构器"""
    def __init__(self, method='simple_normalize'):
        self.method = method

    def reconstruct_probabilities(self, binary_outputs):
        """从二分类器输出重构多分类概率"""
        batch_size = binary_outputs[0].size(0)
        num_classes = len(binary_outputs)
        device = binary_outputs[0].device

        positive_probs = []
        for binary_output in binary_outputs:
            prob = torch.softmax(binary_output, dim=1)[:, 1]
            positive_probs.append(prob)

        positive_probs = torch.stack(positive_probs, dim=1)

        if self.method == 'simple_normalize':
            reconstructed = positive_probs / (torch.sum(positive_probs, dim=1, keepdims=True) + 1e-8)

        return reconstructed

class MultiModalDataset(Dataset):
    """多模态数据集"""
    def __init__(self, modal_features, labels):
        self.modal_features = torch.FloatTensor(modal_features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'modal_data': self.modal_features[idx].transpose(0, 1),
            'label': self.labels[idx]
        }

def select_best_modals(modal_features, labels, k=12):
    """模态选择 - 基于Fisher比率"""
    print(f"选择{k}个最佳模态...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    unique_labels = np.unique(encoded_labels)

    modal_scores = []
    for modal_idx in range(modal_features.shape[3]):
        modal_data = modal_features[:, :, 1, modal_idx]
        sample_features = np.array([[np.mean(modal_data[i]), np.std(modal_data[i]), np.max(modal_data[i])]
                                  for i in range(len(modal_data))])

        feature_scores = []
        for feat_idx in range(sample_features.shape[1]):
            feature_values = sample_features[:, feat_idx]
            overall_mean = np.mean(feature_values)

            between_var = sum(len(feature_values[encoded_labels == label]) *
                            (np.mean(feature_values[encoded_labels == label]) - overall_mean) ** 2
                            for label in unique_labels) / len(feature_values)

            within_var = sum(len(feature_values[encoded_labels == label]) * np.var(feature_values[encoded_labels == label])
                           for label in unique_labels) / len(feature_values)

            fisher_ratio = between_var / (within_var + 1e-8)
            feature_scores.append(fisher_ratio)

        modal_scores.append(np.mean(feature_scores))

    selected_indices = sorted(np.argsort(modal_scores)[-k:])
    print(f"选择的模态: {[i+1 for i in selected_indices]}")

    selected_features = modal_features[:, :, :, selected_indices]
    return selected_indices, selected_features

def load_processed_data():
    """加载处理好的数据"""
    data_dir = 'data/processed'
    filename_prefix = 'processed_imc_data'

    try:
        data_pairs = np.load(os.path.join(data_dir, f'{filename_prefix}_data_pairs.npy'))
        labels = np.load(os.path.join(data_dir, f'{filename_prefix}_labels.npy'))
        multi_modal_features = np.load(os.path.join(data_dir, f'{filename_prefix}_modal_features.npy'))

        with open(os.path.join(data_dir, f'{filename_prefix}_sequence_info.json'), 'r', encoding='utf-8') as f:
            sequence_info = json.load(f)

        with open(os.path.join(data_dir, f'{filename_prefix}_summary.json'), 'r', encoding='utf-8') as f:
            summary = json.load(f)

        print("数据加载成功!")
        print(f"数据形状: {data_pairs.shape}")
        print(f"标签数量: {len(labels)}")

        return {
            'data_pairs': data_pairs,
            'labels': labels,
            'multi_modal_features': multi_modal_features,
            'sequence_info': sequence_info,
            'summary': summary
        }

    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def plot_roc_curves(true_labels, probs_traditional, probs_probability, output_dir):
    """
    绘制ROC曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ROC Curves for Multi-Class Classification', fontsize=16, fontweight='bold')

    class_names = ['ACL', 'MS', 'ACL+MS', 'PFJ']
    colors = ['blue', 'red', 'green', 'orange']

    # 传统多分类器的ROC曲线
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    for i in range(4):
        fpr, tpr, _ = roc_curve((np.array(true_labels) == i).astype(int), probs_traditional[:, i])
        auc_score = roc_auc_score((np.array(true_labels) == i).astype(int), probs_traditional[:, i])
        axes[0, 0].plot(fpr, tpr, color=colors[i], linewidth=2,
                       label=f'{class_names[i]} (AUC = {auc_score:.4f})')

    axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 0].set_title('Traditional Multi-Classifier ROC Curves', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='lower right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])

    # 概率重构分类器的ROC曲线
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    for i in range(4):
        fpr, tpr, _ = roc_curve((np.array(true_labels) == i).astype(int), probs_probability[:, i])
        auc_score = roc_auc_score((np.array(true_labels) == i).astype(int), probs_probability[:, i])
        axes[0, 1].plot(fpr, tpr, color=colors[i], linewidth=2,
                       label=f'{class_names[i]} (AUC = {auc_score:.4f})')

    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title('Probability Reconstruction Classifier ROC Curves', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='lower right', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])

    # Macro-average ROC曲线对比
    axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

    # 传统多分类器的macro-average
    all_fpr_traditional = np.linspace(0, 1, 100)
    mean_tpr_traditional = np.zeros_like(all_fpr_traditional)
    for i in range(4):
        fpr, tpr, _ = roc_curve((np.array(true_labels) == i).astype(int), probs_traditional[:, i])
        mean_tpr_traditional += np.interp(all_fpr_traditional, fpr, tpr)
    mean_tpr_traditional /= 4
    macro_auc_traditional = roc_auc_score(true_labels, probs_traditional, multi_class='ovr', average='macro')
    axes[1, 0].plot(all_fpr_traditional, mean_tpr_traditional, color='blue', linewidth=3,
                   label=f'Traditional (Macro-AUC = {macro_auc_traditional:.4f})')

    # 概率重构分类器的macro-average
    all_fpr_probability = np.linspace(0, 1, 100)
    mean_tpr_probability = np.zeros_like(all_fpr_probability)
    for i in range(4):
        fpr, tpr, _ = roc_curve((np.array(true_labels) == i).astype(int), probs_probability[:, i])
        mean_tpr_probability += np.interp(all_fpr_probability, fpr, tpr)
    mean_tpr_probability /= 4
    macro_auc_probability = roc_auc_score(true_labels, probs_probability, multi_class='ovr', average='macro')
    axes[1, 0].plot(all_fpr_probability, mean_tpr_probability, color='red', linewidth=3,
                   label=f'Probability Reconstruction (Macro-AUC = {macro_auc_probability:.4f})')

    axes[1, 0].set_xlabel('False Positive Rate', fontsize=12)
    axes[1, 0].set_ylabel('True Positive Rate', fontsize=12)
    axes[1, 0].set_title('Macro-Average ROC Curves Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='lower right', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])

    # 性能对比柱状图
    labels = ['Accuracy', 'Balanced Acc', 'ROC-AUC', 'F1-Score', 'MCC']
    traditional_scores = [
        accuracy_score(true_labels, np.argmax(probs_traditional, axis=1)),
        balanced_accuracy_score(true_labels, np.argmax(probs_traditional, axis=1)),
        macro_auc_traditional,
        f1_score(true_labels, np.argmax(probs_traditional, axis=1), average='macro'),
        matthews_corrcoef(true_labels, np.argmax(probs_traditional, axis=1))
    ]
    probability_scores = [
        accuracy_score(true_labels, np.argmax(probs_probability, axis=1)),
        balanced_accuracy_score(true_labels, np.argmax(probs_probability, axis=1)),
        macro_auc_probability,
        f1_score(true_labels, np.argmax(probs_probability, axis=1), average='macro'),
        matthews_corrcoef(true_labels, np.argmax(probs_probability, axis=1))
    ]

    x = np.arange(len(labels))
    width = 0.35

    axes[1, 1].bar(x - width/2, traditional_scores, width, label='Traditional', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, probability_scores, width, label='Probability Reconstruction', color='red', alpha=0.7)

    axes[1, 1].set_xlabel('Metrics', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Performance Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 1].legend(loc='lower right', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0.0, 1.0])

    # 为每个柱子添加数值标签
    for i, (trad, prob) in enumerate(zip(traditional_scores, probability_scores)):
        axes[1, 1].text(i - width/2, trad + 0.01, '.3f', ha='center', va='bottom', fontsize=9)
        axes[1, 1].text(i + width/2, prob + 0.01, '.3f', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存图片
    svg_path = os.path.join(output_dir, 'roc_curves_comparison.svg')
    png_path = os.path.join(output_dir, 'roc_curves_comparison.png')

    plt.savefig(svg_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"ROC曲线已保存到: {svg_path}")
    print(f"ROC曲线已保存到: {png_path}")

    plt.show()

def evaluate_models():
    """
    评估所有训练好的模型 - 完全复制训练脚本的逻辑
    """
    print("="*60)
    print("IMC模型评估脚本")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载数据
    data = load_processed_data()
    if data is None:
        return

    # 2. 模态选择
    k_modals = 12
    selected_indices, selected_features = select_best_modals(
        data['multi_modal_features'], data['labels'], k_modals
    )

    # 3. 数据分割 (与训练完全一致)
    modal_data = selected_features[:, :, 1, :]
    labels = data['labels']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_

    indices = list(range(len(modal_data)))
    train_idx, val_idx = train_test_split(indices, test_size=0.3,
                                         stratify=y_encoded, random_state=42)

    print(f"训练集大小: {len(train_idx)} 个样本")
    print(f"验证集大小: {len(val_idx)} 个样本")
    print("使用与训练脚本完全相同的数据分割和预处理！")

    # 4. 创建验证数据集
    val_modal_data = modal_data[val_idx]
    val_labels = y_encoded[val_idx]
    val_dataset = MultiModalDataset(val_modal_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)

    # 5. 加载训练好的模型
    base_path = r'result/experiment/imc_classifier_recon_simp_k12_lr1e-04_bs64_230800/models'

    print("\n加载传统多分类器...")
    traditional_model = ClassifierNet(num_classes=4, num_modals=k_modals).to(device)
    traditional_model.load_state_dict(torch.load(os.path.join(base_path, 'best_traditional_multi_classifier_model.pth'), map_location=device))
    traditional_model.eval()

    print("加载二分类器...")
    binary_classifiers = {}
    for class_name in class_names:
        binary_model = ClassifierNet(num_classes=2, num_modals=k_modals).to(device)
        binary_model.load_state_dict(torch.load(os.path.join(base_path, f'best_binary_{class_name}_model.pth'), map_location=device))
        binary_classifiers[class_name] = binary_model
        binary_model.eval()

    # 6. 初始化概率重构器
    reconstructor = ProbabilityReconstructor(method='simple_normalize')

    print("\n开始评估所有模型...")

    # 7. 在验证集上评估
    all_preds_traditional = []
    all_preds_probability = []
    all_probabilities_traditional = []
    all_probabilities_probability = []
    all_true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            modal_data_batch = batch['modal_data'].to(device)
            labels_batch = batch['label']

            batch_size = modal_data_batch.size(0)

            # 传统多分类器预测
            traditional_outputs, _ = traditional_model(modal_data_batch)
            traditional_probs = torch.softmax(traditional_outputs, dim=1)
            _, traditional_preds = torch.max(traditional_outputs, 1)

            # 概率重构分类器预测
            binary_outputs = []
            for class_name in class_names:
                output, _ = binary_classifiers[class_name](modal_data_batch)
                binary_outputs.append(output)

            reconstructed_probs = reconstructor.reconstruct_probabilities(binary_outputs)
            _, probability_preds = torch.max(reconstructed_probs, 1)

            # 收集结果
            all_preds_traditional.extend(traditional_preds.cpu().numpy())
            all_preds_probability.extend(probability_preds.cpu().numpy())
            all_probabilities_traditional.extend(traditional_probs.cpu().numpy())
            all_probabilities_probability.extend(reconstructed_probs.cpu().numpy())
            all_true_labels.extend(labels_batch.cpu().numpy())

    # 8. 计算指标
    traditional_acc = accuracy_score(all_true_labels, all_preds_traditional)
    probability_acc = accuracy_score(all_true_labels, all_preds_probability)
    traditional_balanced_acc = balanced_accuracy_score(all_true_labels, all_preds_traditional)
    probability_balanced_acc = balanced_accuracy_score(all_true_labels, all_preds_probability)

    y_true_encoded = np.array(all_true_labels)
    probs_array_traditional = np.array(all_probabilities_traditional)
    probs_array_probability = np.array(all_probabilities_probability)

    # 归一化概率
    if len(probs_array_traditional.shape) > 1:
        row_sums_traditional = np.sum(probs_array_traditional, axis=1)
        if not np.allclose(row_sums_traditional, 1.0, atol=1e-6):
            probs_array_traditional = probs_array_traditional / row_sums_traditional[:, np.newaxis]

    if len(probs_array_probability.shape) > 1:
        row_sums_probability = np.sum(probs_array_probability, axis=1)
        if not np.allclose(row_sums_probability, 1.0, atol=1e-6):
            probs_array_probability = probs_array_probability / row_sums_probability[:, np.newaxis]

    traditional_roc_auc = roc_auc_score(y_true_encoded, probs_array_traditional, multi_class='ovr', average='macro')
    probability_roc_auc = roc_auc_score(y_true_encoded, probs_array_probability, multi_class='ovr', average='macro')
    traditional_f1 = f1_score(y_true_encoded, all_preds_traditional, average='macro')
    probability_f1 = f1_score(y_true_encoded, all_preds_probability, average='macro')
    traditional_mcc = matthews_corrcoef(y_true_encoded, all_preds_traditional)
    probability_mcc = matthews_corrcoef(y_true_encoded, all_preds_probability)

    # 9. 输出结果
    print("\n" + "="*70)
    print("                    分类器性能对比")
    print("="*70)
    print(f"{'指标':<15} {'传统多分类器':<15} {'概率重构分类器':<15}")
    print("-"*70)
    print(f"{'准确率(ACC)':<15} {traditional_acc:.4f} ({traditional_acc*100:.1f}%)    {probability_acc:.4f} ({probability_acc*100:.1f}%)")
    print(f"{'平衡准确率(BA)':<15} {traditional_balanced_acc:.4f}         {probability_balanced_acc:.4f}")
    print(f"{'ROC-AUC':<15} {traditional_roc_auc:.4f}         {probability_roc_auc:.4f}")
    print(f"{'PR-AUC':<15} {0.0:.4f}         {0.0:.4f}")
    print(f"{'F1-Score':<15} {traditional_f1:.4f}         {probability_f1:.4f}")
    print(f"{'MCC':<15} {traditional_mcc:.4f}         {probability_mcc:.4f}")

    # 10. 创建输出目录并保存结果
    output_dir = 'result/inference'
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'evaluation_time': datetime.now().isoformat(),
        'data_info': {
            'total_samples': int(len(data['labels'])),
            'train_samples': int(len(train_idx)),
            'validation_samples': int(len(val_idx)),
            'selected_modals': [int(i+1) for i in selected_indices]
        },
        'traditional_multi_classifier': {
            'accuracy': float(traditional_acc),
            'balanced_accuracy': float(traditional_balanced_acc),
            'roc_auc': float(traditional_roc_auc),
            'f1': float(traditional_f1),
            'mcc': float(traditional_mcc)
        },
        'probability_reconstruction': {
            'accuracy': float(probability_acc),
            'balanced_accuracy': float(probability_balanced_acc),
            'roc_auc': float(probability_roc_auc),
            'f1': float(probability_f1),
            'mcc': float(probability_mcc)
        }
    }

    # 保存评估结果
    output_file = os.path.join(output_dir, 'model_evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n评估结果已保存到: {output_file}")

    # 11. 绘制并保存ROC曲线
    plot_roc_curves(all_true_labels, probs_array_traditional, probs_array_probability, output_dir)

    # 12. 返回最佳方法
    if probability_acc > traditional_acc:
        print("概率重构分类器表现更好！")
        return 'probability_reconstruction'
    else:
        print("传统多分类器表现更好！")
        return 'traditional_multi_classifier'

def main():
    """主函数"""
    try:
        best_method = evaluate_models()
        print(f"\n评估完成！最佳方法: {best_method}")

        # 与训练结果对比
        print("\n" + "="*50)
        print("与训练结果对比:")
        print("="*50)

        try:
            with open('result/experiment/imc_classifier_recon_simp_k12_lr1e-04_bs64_230800/experiment_summary.json', 'r') as f:
                train_results = json.load(f)

            print("训练时的最终结果:")
            print(".4f")
            print(".4f")
            print(".4f")
        except FileNotFoundError:
            print("未找到训练结果文件")

    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
