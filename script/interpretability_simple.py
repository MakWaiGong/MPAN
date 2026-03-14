import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, precision_recall_curve, auc
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
from datetime import datetime

class SimpleModalNet(nn.Module):
    """简化的单模态网络"""
    def __init__(self, sequence_length=81):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 20, 64),  # 81//4 = 20.25, 所以大约20
            nn.ReLU(),
            nn.Linear(64, 4)  # 4个类别
        )

    def forward(self, x):
        # x: (batch_size, 81) -> (batch_size, 1, 81)
        x = x.unsqueeze(1)
        return self.net(x)

class SimpleDataset(Dataset):
    def __init__(self, modal_data, labels):
        self.modal_data = torch.FloatTensor(modal_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'modal_data': self.modal_data[idx],  # (81,) 直接返回
            'label': self.labels[idx]
        }

def evaluate_single_modal_simple(modal_idx, modal_features, labels):
    """使用简化模型评估单个模态"""
    try:
        device = torch.device('cpu')

        # 提取单个模态数据
        single_modal_features = modal_features[:, :, :, modal_idx:modal_idx+1]
        modal_data = single_modal_features[:, :, 1, 0]  # (n_samples, 81) - 去掉最后一维

        # 标签编码
        if isinstance(labels[0], str):
            label_encoder = LabelEncoder()
            numeric_labels = label_encoder.fit_transform(labels)
        else:
            numeric_labels = labels

        # 数据分割 (70%训练, 30%验证)
        indices = list(range(len(modal_data)))
        train_idx, val_idx = train_test_split(indices, test_size=0.3,
                                             stratify=numeric_labels, random_state=42)

        train_data = modal_data[train_idx]
        train_labels = numeric_labels[train_idx]
        val_data = modal_data[val_idx]
        val_labels = numeric_labels[val_idx]

        # 创建数据集
        train_dataset = SimpleDataset(train_data, train_labels)
        val_dataset = SimpleDataset(val_data, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 创建模型
        model = SimpleModalNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        model.train()
        for epoch in range(30):  # 30个epoch
            for batch in train_loader:
                modal_batch = batch['modal_data'].to(device)
                labels_batch = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(modal_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

        # 评估模型
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                modal_batch = batch['modal_data'].to(device)
                labels_batch = batch['label'].to(device)

                outputs = model(modal_batch)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        try:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except:
            roc_auc = 0.5

        try:
            pr_auc = 0
            for i in range(4):
                y_binary = (all_labels == i).astype(int)
                if np.sum(y_binary) > 0 and np.sum(y_binary == 0) > 0:
                    precision, recall, _ = precision_recall_curve(y_binary, all_probs[:, i])
                    pr_auc += auc(recall, precision)
            pr_auc /= 4
        except:
            pr_auc = 0.0

        return {
            'modal_idx': modal_idx + 1,
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            'mcc': float(mcc),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'sample_count': len(val_labels)
        }

    except Exception as e:
        print(f"模态{modal_idx+1}评估出错: {e}")
        return {
            'modal_idx': modal_idx + 1,
            'accuracy': 0.25,
            'balanced_accuracy': 0.25,
            'mcc': 0.0,
            'f1_score': 0.25,
            'roc_auc': 0.5,
            'pr_auc': 0.25,
            'sample_count': 0,
            'error': str(e)
        }

def plot_modal_analysis_simple(results, output_dir):
    """简化版的可视化"""
    if not results:
        print("没有结果可绘制")
        return

    modal_indices = [r['modal_idx'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    balanced_accuracies = [r['balanced_accuracy'] for r in results]
    mccs = [r['mcc'] for r in results]
    f1_scores = [r['f1_score'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Modal Importance Analysis (k=1) - Simplified Model', fontsize=16, fontweight='bold')

    x = np.arange(len(modal_indices))
    width = 0.35

    # Accuracy
    axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, balanced_accuracies, width, label='Balanced Acc', color='cyan', alpha=0.7)
    axes[0, 0].set_title('Accuracy Metrics', fontsize=12)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'M{i}' for i in modal_indices])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.0])

    # MCC and F1
    axes[0, 1].bar(x - width/2, mccs, width, label='MCC', color='red', alpha=0.7)
    axes[0, 1].bar(x + width/2, f1_scores, width, label='F1-Score', color='orange', alpha=0.7)
    axes[0, 1].set_title('MCC & F1 Metrics', fontsize=12)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'M{i}' for i in modal_indices])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([-0.2, 1.0])

    # Statistics
    metrics_names = ['Avg Acc', 'Avg BA', 'Avg MCC', 'Avg F1']
    avg_metrics = [
        np.mean(accuracies), np.mean(balanced_accuracies),
        np.mean(mccs), np.mean(f1_scores)
    ]
    max_metrics = [
        np.max(accuracies), np.max(balanced_accuracies),
        np.max(mccs), np.max(f1_scores)
    ]

    x_stats = np.arange(len(metrics_names))
    axes[1, 0].bar(x_stats, avg_metrics, color='blue', alpha=0.7, label='Average')
    axes[1, 0].bar(x_stats, max_metrics, color='red', alpha=0.5, label='Maximum', width=0.4)
    axes[1, 0].set_title('Overall Statistics', fontsize=12)
    axes[1, 0].set_xticks(x_stats)
    axes[1, 0].set_xticklabels(metrics_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Top 5 modals
    comprehensive_scores = np.mean([
        accuracies, balanced_accuracies, (np.array(mccs) + 1) / 2, f1_scores
    ], axis=0)  # 归一化MCC到0-1

    sorted_indices = np.argsort(comprehensive_scores)[::-1]
    top_modals = [modal_indices[i] for i in sorted_indices[:5]]
    top_scores = [comprehensive_scores[i] for i in sorted_indices[:5]]

    axes[1, 1].bar(range(len(top_modals)), top_scores, color='darkgreen', alpha=0.8)
    axes[1, 1].set_title('Top 5 Modals', fontsize=12)
    axes[1, 1].set_xticks(range(len(top_modals)))
    axes[1, 1].set_xticklabels([f'M{m}' for m in top_modals])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1.0])

    plt.tight_layout()

    # 保存
    svg_path = os.path.join(output_dir, 'modal_importance_simple.svg')
    png_path = os.path.join(output_dir, 'modal_importance_simple.png')

    plt.savefig(svg_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=100, bbox_inches='tight')

    print(f"简化版模态重要性分析图已保存到: {svg_path}")
    print(f"简化版模态重要性分析图已保存到: {png_path}")

    plt.show()

def load_processed_data(output_dir, filename_prefix='processed_imc_data'):
    """加载处理后的数据"""
    try:
        data_pairs = np.load(os.path.join(output_dir, f'{filename_prefix}_data_pairs.npy'))
        labels = np.load(os.path.join(output_dir, f'{filename_prefix}_labels.npy'))
        multi_modal_features = np.load(os.path.join(output_dir, f'{filename_prefix}_modal_features.npy'))

        return {
            'data_pairs': data_pairs,
            'labels': labels,
            'multi_modal_features': multi_modal_features
        }
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

if __name__ == "__main__":
    print("=== 简化版模态可解释性分析 ===")

    # 加载数据
    data = load_processed_data(
        output_dir=os.path.join(os.path.dirname(__file__), '../data/processed'),
        filename_prefix='processed_imc_data'
    )

    if data:
        print("数据加载成功，开始简化版模态重要性分析...")

        results = []
        print("\n开始逐个测试16个模态 (简化模型)...")

        for modal_idx in range(16):
            print(f"测试模态 {modal_idx + 1}/16...")
            result = evaluate_single_modal_simple(modal_idx, data['multi_modal_features'], data['labels'])
            if result:
                results.append(result)
                print(".4f")

        # 保存结果
        output_file = os.path.join('result/interpretability', 'modal_importance_simple.json')
        os.makedirs('result/interpretability', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'evaluation_time': datetime.now().isoformat(),
                'total_modals_tested': len(results),
                'model_type': 'simplified_cnn',
                'modal_results': results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n简化版结果已保存到: {output_file}")

        # 生成可视化
        plot_modal_analysis_simple(results, 'result/interpretability')

    else:
        print("数据加载失败")
