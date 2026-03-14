#!/usr/bin/env python3
"""
模态可解释性分析脚本 - 测试每个模态的重要性

功能：
1. 对16个模态逐一进行测试 (k=1)
2. 评估每个模态单独时的模型性能
3. 生成模态重要性分析图表
4. 保存结果到指定目录

使用方法：
python script/modal_interpretability_analysis.py

输出位置：result/interpretability/

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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_curve, auc, f1_score
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

def evaluate_single_modal(modal_idx, modal_features, labels):
    """
    评估单个模态的性能
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用第modal_idx个模态 (k=1)
    single_modal_features = modal_features[:, :, :, modal_idx:modal_idx+1]  # 选择单个模态
    modal_data = single_modal_features[:, :, 1, :]  # 提取角度数据

    # 数据分割
    indices = list(range(len(modal_data)))
    train_idx, val_idx = train_test_split(indices, test_size=0.3,
                                         stratify=labels, random_state=42)

    val_modal_data = modal_data[val_idx]
    val_labels = labels[val_idx]
    val_dataset = MultiModalDataset(val_modal_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)

    # 加载训练好的模型
    base_path = r'result/experiment/imc_classifier_recon_simp_k12_lr1e-04_bs64_230800/models'

    # 加载传统多分类器 (为单个模态重新训练的版本)
    traditional_model_path = os.path.join(base_path, f'best_traditional_multi_classifier_modal_{modal_idx+1}.pth')
    if not os.path.exists(traditional_model_path):
        # 如果没有特定模态的模型，使用原始模型（可能不准确，但用于演示）
        traditional_model_path = os.path.join(base_path, 'best_traditional_multi_classifier_model.pth')
        print(f"警告: 未找到模态{modal_idx+1}的专用模型，使用默认模型")

    try:
        traditional_model = ClassifierNet(num_classes=4, num_modals=1).to(device)
        traditional_model.load_state_dict(torch.load(traditional_model_path, map_location=device))
        traditional_model.eval()
    except:
        print(f"无法加载模态{modal_idx+1}的传统模型，跳过")
        return None

    # 加载二分类器
    binary_classifiers = {}
    class_names = ['a', 'b', 'c', 'd']
    loaded_count = 0

    for class_name in class_names:
        binary_model_path = os.path.join(base_path, f'best_binary_{class_name}_model_modal_{modal_idx+1}.pth')
        if not os.path.exists(binary_model_path):
            binary_model_path = os.path.join(base_path, f'best_binary_{class_name}_model.pth')
            print(f"警告: 未找到模态{modal_idx+1}的{class_name}分类器，使用默认模型")

        try:
            binary_model = ClassifierNet(num_classes=2, num_modals=1).to(device)
            binary_model.load_state_dict(torch.load(binary_model_path, map_location=device))
            binary_classifiers[class_name] = binary_model
            binary_model.eval()
            loaded_count += 1
        except:
            print(f"无法加载模态{modal_idx+1}的{class_name}分类器，跳过")

    if loaded_count < 4:
        print(f"模态{modal_idx+1}的二分类器加载不完整，跳过")
        return None

    # 概率重构器
    reconstructor = ProbabilityReconstructor(method='simple_normalize')

    # 评估
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

    # 计算指标
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

    return {
        'modal_idx': modal_idx + 1,  # 1-based indexing
        'traditional': {
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

def plot_modal_importance(results, output_dir):
    """
    绘制模态重要性分析图表
    """
    # 准备数据
    modal_indices = [r['modal_idx'] for r in results if r is not None]
    traditional_acc = [r['traditional']['accuracy'] for r in results if r is not None]
    prob_acc = [r['probability_reconstruction']['accuracy'] for r in results if r is not None]

    traditional_auc = [r['traditional']['roc_auc'] for r in results if r is not None]
    prob_auc = [r['probability_reconstruction']['roc_auc'] for r in results if r is not None]

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Modal Importance Analysis (k=1)', fontsize=16, fontweight='bold')

    # 准确率对比
    x = np.arange(len(modal_indices))
    width = 0.35

    axes[0, 0].bar(x - width/2, traditional_acc, width, label='Traditional Multi-Classifier',
                   color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, prob_acc, width, label='Probability Reconstruction',
                   color='red', alpha=0.7)

    axes[0, 0].set_xlabel('Modal Index', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Accuracy by Modal', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'M{i}' for i in modal_indices])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.0])

    # ROC-AUC对比
    axes[0, 1].bar(x - width/2, traditional_auc, width, label='Traditional Multi-Classifier',
                   color='blue', alpha=0.7)
    axes[0, 1].bar(x + width/2, prob_auc, width, label='Probability Reconstruction',
                   color='red', alpha=0.7)

    axes[0, 1].set_xlabel('Modal Index', fontsize=12)
    axes[0, 1].set_ylabel('ROC-AUC', fontsize=12)
    axes[0, 1].set_title('ROC-AUC by Modal', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'M{i}' for i in modal_indices])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1.0])

    # 性能趋势线
    axes[1, 0].plot(modal_indices, traditional_acc, 'b-o', linewidth=2, label='Traditional Acc')
    axes[1, 0].plot(modal_indices, prob_acc, 'r-s', linewidth=2, label='Probability Reconstruction Acc')
    axes[1, 0].plot(modal_indices, traditional_auc, 'b--o', linewidth=1, alpha=0.7, label='Traditional AUC')
    axes[1, 0].plot(modal_indices, prob_auc, 'r--s', linewidth=1, alpha=0.7, label='Probability Reconstruction AUC')

    axes[1, 0].set_xlabel('Modal Index', fontsize=12)
    axes[1, 0].set_ylabel('Performance Score', fontsize=12)
    axes[1, 0].set_title('Modal Performance Trends', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.0])

    # 统计汇总
    avg_traditional_acc = np.mean(traditional_acc)
    avg_prob_acc = np.mean(prob_acc)
    avg_traditional_auc = np.mean(traditional_auc)
    avg_prob_auc = np.mean(prob_auc)

    metrics = ['Avg Accuracy', 'Avg ROC-AUC']
    traditional_avg = [avg_traditional_acc, avg_traditional_auc]
    prob_avg = [avg_prob_acc, avg_prob_auc]

    x_avg = np.arange(len(metrics))
    axes[1, 1].bar(x_avg - width/2, traditional_avg, width, label='Traditional',
                   color='blue', alpha=0.7)
    axes[1, 1].bar(x_avg + width/2, prob_avg, width, label='Probability Reconstruction',
                   color='red', alpha=0.7)

    axes[1, 1].set_xlabel('Metrics', fontsize=12)
    axes[1, 1].set_ylabel('Average Score', fontsize=12)
    axes[1, 1].set_title('Overall Performance Summary', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x_avg)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1.0])

    # 添加数值标签
    for i, (trad, prob) in enumerate(zip(traditional_avg, prob_avg)):
        axes[1, 1].text(i - width/2, trad + 0.01, '.3f', ha='center', va='bottom', fontsize=10)
        axes[1, 1].text(i + width/2, prob + 0.01, '.3f', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # 保存图表
    svg_path = os.path.join(output_dir, 'modal_importance_analysis.svg')
    png_path = os.path.join(output_dir, 'modal_importance_analysis.png')

    plt.savefig(svg_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"模态重要性分析图已保存到: {svg_path}")
    print(f"模态重要性分析图已保存到: {png_path}")

    plt.show()

def main():
    """
    主函数：循环测试16个模态的重要性
    """
    print("="*60)
    print("模态可解释性分析脚本")
    print("测试每个模态单独的性能 (k=1)")
    print("="*60)

    # 创建输出目录
    output_dir = r'D:\document\code\IMC\result\interpretability'
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    data = load_processed_data()
    if data is None:
        return

    # 循环测试16个模态
    results = []
    print("\n开始逐个测试16个模态...")
    print("="*50)

    for modal_idx in range(16):  # 0-15 对应 16个模态
        print(f"\n测试模态 {modal_idx + 1}/16...")
        try:
            result = evaluate_single_modal(modal_idx, data['multi_modal_features'], data['labels'])
            if result is not None:
                results.append(result)
                print(f"模态{modal_idx + 1}测试完成:")
                print(".4f")
                print(".4f")
            else:
                print(f"模态{modal_idx + 1}测试失败，跳过")
        except Exception as e:
            print(f"模态{modal_idx + 1}测试出错: {e}")
            continue

    if not results:
        print("没有成功的测试结果，无法生成图表")
        return

    print(f"\n成功测试了{len(results)}个模态")

    # 保存详细结果
    output_file = os.path.join(output_dir, 'modal_importance_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluation_time': datetime.now().isoformat(),
            'total_modals_tested': len(results),
            'modal_results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存到: {output_file}")

    # 生成可视化图表
    print("\n生成模态重要性分析图表...")
    plot_modal_importance(results, output_dir)

    # 输出总结
    print("\n" + "="*60)
    print("模态可解释性分析完成！")
    print("="*60)
    print(f"测试的模态数量: {len(results)}/16")
    print(f"输出目录: {output_dir}")
    print("生成的文件:")
    print("  - modal_importance_results.json (详细结果)")
    print("  - modal_importance_analysis.svg (矢量图)")
    print("  - modal_importance_analysis.png (位图)")

if __name__ == "__main__":
    main()



