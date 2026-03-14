import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_curve, auc, f1_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
from tqdm import tqdm
import json
import os
from datetime import datetime

class ExperimentManager:
    """实验管理器 - 自动组织实验结果"""

    def __init__(self, base_dir='result/experiment'):
        self.base_dir = base_dir
        self.experiment_id = None
        self.experiment_dir = None
        self.start_time = None

        # 不再创建基础目录，只在使用时创建实验专用目录

    def start_experiment(self, experiment_name=None, config=None):
        """开始新实验"""
        self.start_time = datetime.now()

        # 从配置中提取关键超参数生成目录名
        if config:
            # 提取关键超参数
            model_config = config.get('model_config', {})
            training_config = config.get('training_config', {})

            # 构建超参数字符串
            params = []
            if 'reconstruction_method' in model_config:
                params.append(f"recon_{model_config['reconstruction_method'][:4]}")
            if 'num_modals' in model_config:
                params.append(f"k{model_config['num_modals']}")
            if 'learning_rate' in training_config:
                lr = training_config['learning_rate']
                params.append(f"lr{lr:.0e}" if lr < 0.01 else f"lr{lr:.4f}")
            if 'batch_size' in training_config:
                params.append(f"bs{training_config['batch_size']}")

            # 添加时间戳后缀
            timestamp = self.start_time.strftime("%H%M%S")
            param_str = "_".join(params) if params else "default"

            if experiment_name:
                self.experiment_id = f"{experiment_name}_{param_str}_{timestamp}"
            else:
                self.experiment_id = f"exp_{param_str}_{timestamp}"
        else:
            # 没有配置时的默认行为
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            if experiment_name:
                self.experiment_id = f"{experiment_name}_{timestamp}"
            else:
                self.experiment_id = f"exp_{timestamp}"

        self.experiment_dir = f"{self.base_dir}/{self.experiment_id}"

        # 创建实验专用目录
        os.makedirs(f'{self.experiment_dir}/models', exist_ok=True)
        os.makedirs(f'{self.experiment_dir}/logs', exist_ok=True)
        os.makedirs(f'{self.experiment_dir}/metrics', exist_ok=True)
        os.makedirs(f'{self.experiment_dir}/plots', exist_ok=True)

        # 注意：配置文件现在在main函数中手动保存，以便包含所有最终参数

        print(f"开始实验: {self.experiment_id}")
        print(f"实验目录: {self.experiment_dir}")

        return self.experiment_id

    def get_model_path(self, model_name):
        """获取模型保存路径"""
        if self.experiment_dir:
            return f"{self.experiment_dir}/models/{model_name}"
        else:
            raise ValueError("必须先开始实验（调用start_experiment）才能保存模型")

    def log_metrics(self, metrics_dict, filename='training_metrics.json'):
        """记录训练指标"""
        if self.experiment_dir:
            metrics_path = f"{self.experiment_dir}/metrics/{filename}"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError("必须先开始实验（调用start_experiment）才能记录指标")

    def save_config(self, config):
        """手动保存实验配置"""
        if self.experiment_dir:
            config_path = f'{self.experiment_dir}/config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"配置文件已保存: {config_path}")
        else:
            raise ValueError("必须先开始实验（调用start_experiment）才能保存配置")

    def save_plot(self, plot_filename):
        """保存图表（假设plt已定义）"""
        if self.experiment_dir:
            plot_path = f"{self.experiment_dir}/plots/{plot_filename}"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {plot_path}")
        else:
            raise ValueError("必须先开始实验（调用start_experiment）才能保存图表")

    def finish_experiment(self, final_metrics=None):
        """结束实验并记录摘要"""
        if not self.experiment_dir:
            return

        end_time = datetime.now()
        duration = end_time - self.start_time

        summary = {
            'experiment_id': self.experiment_id,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'final_metrics': final_metrics or {}
        }

        summary_path = f"{self.experiment_dir}/experiment_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"实验完成: {self.experiment_id}")
        print(f"总耗时: {duration}")

class MultiModalDataset(Dataset):
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

class BinaryDataset(Dataset):
    def __init__(self, modal_features, labels, target_class):
        self.modal_features = torch.FloatTensor(modal_features)
        self.labels = torch.LongTensor([1 if label == target_class else 0 for label in labels])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'modal_data': self.modal_features[idx].transpose(0, 1),
            'label': self.labels[idx]
        }

class SingleModalNet(nn.Module):
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
    """单个Transformer块"""
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super().__init__()
        # 多头自注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )

        # 层归一化和dropout
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多头自注意力 + 残差连接
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x
        )
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class TransformerAttentionFusion(nn.Module):
    """基于多层Transformer的模态融合注意力机制"""
    def __init__(self, feature_dim, num_modals, num_layers=8, num_heads=2, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modals = num_modals
        self.num_layers = num_layers
        self.num_heads = num_heads

        # 堆叠多个Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 最终的融合权重计算（用于输出注意力权重）
        self.fusion_weights = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, modal_features):
        """
        Args:
            modal_features: (batch_size, num_modals, feature_dim)
        Returns:
            fused_features: (batch_size, feature_dim)
            attention_weights: (batch_size, num_modals)
        """
        batch_size, num_modals, feature_dim = modal_features.shape

        # 通过多层Transformer处理
        for layer in self.transformer_layers:
            modal_features = layer(modal_features)

        # 计算最终融合权重（用于输出注意力权重）
        attention_weights = self.fusion_weights(modal_features).squeeze(-1)

        # 加权融合
        fused_features = torch.sum(attention_weights.unsqueeze(-1) * modal_features, dim=1)

        return fused_features, attention_weights


# 原始注意力融合层（保持兼容性）
class AttentionFusion(nn.Module):
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
    def __init__(self, num_classes, num_modals, sequence_length=81, use_transformer_attention=False):
        super().__init__()
        self.num_modals = num_modals
        self.modal_extractors = nn.ModuleList([
            SingleModalNet(sequence_length) for _ in range(num_modals)
        ])
        self.feature_dim = 256 * 8

        # 选择注意力机制
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
            # modal_data: (batch_size, n_time_points, n_angles)
            single_modal = modal_data.flatten(1).unsqueeze(1)  # (batch_size, 1, n_time_points * n_angles)
            features = self.modal_extractors[i](single_modal)
            modal_features.append(features)
        
        modal_features = torch.stack(modal_features, dim=1)
        fused_features, attention_weights = self.attention_fusion(modal_features)
        output = self.classifier(fused_features)
        
        return (output, attention_weights, fused_features) if return_features else (output, attention_weights)

class ProbabilityReconstructor:
    """概率重构器 - 从二分类器重构多分类概率"""

    def __init__(self, method='simple_normalize'):
        self.method = method
        self.temperature_params = None
        self.calibration_params = None

    def fit_calibration(self, binary_probs, true_labels):
        """为校准归一化方法拟合参数"""
        if self.method == 'calibrated_normalize':
            # 简单的温度缩放校准
            self.temperature_params = []
            for i in range(len(binary_probs)):
                # 为每个二分类器学习温度参数
                binary_true = (true_labels == i).astype(int)
                pos_probs = binary_probs[i][:, 1]

                # 使用简单优化找最佳温度
                best_temp = 1.0
                best_loss = float('inf')
                for temp in np.arange(0.1, 3.0, 0.1):
                    calibrated_probs = torch.sigmoid(torch.logit(torch.FloatTensor(pos_probs)) / temp)
                    loss = F.binary_cross_entropy(calibrated_probs, torch.FloatTensor(binary_true))
                    if loss < best_loss:
                        best_loss = loss
                        best_temp = temp

                self.temperature_params.append(best_temp)

    def reconstruct_probabilities(self, binary_outputs):
        """从二分类器输出重构多分类概率"""
        batch_size = binary_outputs[0].size(0)
        num_classes = len(binary_outputs)
        device = binary_outputs[0].device

        # 获取正类概率
        positive_probs = []
        for binary_output in binary_outputs:
            prob = torch.softmax(binary_output, dim=1)[:, 1]  # 正类概率
            positive_probs.append(prob)

        positive_probs = torch.stack(positive_probs, dim=1)  # (batch, num_classes)

        if self.method == 'simple_normalize':
            # 简单归一化
            reconstructed = positive_probs / (torch.sum(positive_probs, dim=1, keepdims=True) + 1e-8)

        elif self.method == 'calibrated_normalize':
            # 校准后归一化
            if self.temperature_params is not None:
                calibrated_probs = []
                for i, temp in enumerate(self.temperature_params):
                    logits = torch.logit(positive_probs[:, i].clamp(1e-8, 1-1e-8))
                    calibrated = torch.sigmoid(logits / temp)
                    calibrated_probs.append(calibrated)
                calibrated_probs = torch.stack(calibrated_probs, dim=1)
                reconstructed = calibrated_probs / (torch.sum(calibrated_probs, dim=1, keepdims=True) + 1e-8)
            else:
                reconstructed = positive_probs / (torch.sum(positive_probs, dim=1, keepdims=True) + 1e-8)

        elif self.method == 'geometric_normalize':
            # 几何平均归一化 (α=0.5)
            alpha = 0.5
            powered_probs = torch.pow(positive_probs + 1e-8, alpha)
            reconstructed = powered_probs / torch.sum(powered_probs, dim=1, keepdims=True)

        elif self.method == 'softmax_temperature':
            # 温度softmax
            temperature = 2.0
            logits = torch.log(positive_probs + 1e-8)
            reconstructed = torch.softmax(logits / temperature, dim=1)

        elif self.method == 'max_confidence':
            # 最大置信度方法
            max_conf_idx = torch.argmax(positive_probs, dim=1)
            reconstructed = F.one_hot(max_conf_idx, num_classes).float()

        elif self.method == 'threshold_based':
            # 阈值决策
            threshold = 0.5
            above_threshold = (positive_probs > threshold).float()
            # 如果没有超过阈值的，选择最大的
            no_positive = torch.sum(above_threshold, dim=1) == 0
            max_idx = torch.argmax(positive_probs, dim=1)
            above_threshold[no_positive] = F.one_hot(max_idx[no_positive], num_classes).float()[no_positive]
            reconstructed = above_threshold / (torch.sum(above_threshold, dim=1, keepdims=True) + 1e-8)

        else:
            # 默认简单归一化
            reconstructed = positive_probs / (torch.sum(positive_probs, dim=1, keepdims=True) + 1e-8)

        return reconstructed

class ProbabilityReconstructionClassifier:
    """基于概率重构的多分类器 + 传统多分类器对比"""
    
    def __init__(self, k_modals, device=None, reconstruction_method='simple_normalize', experiment_manager=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.traditional_multi_classifier = None  # 传统多分类器
        self.probability_reconstruction_classifier = None  # 概率重构多分类器
        self.binary_classifiers = {}
        self.label_encoder = LabelEncoder()
        self.selected_modals = k_modals
        self.num_classes = None
        self.class_names = None
        self.reconstructor = ProbabilityReconstructor(reconstruction_method)
        self.experiment_manager = experiment_manager
        self.best_classifier_type = None  # 最佳分类器类型
        print(f"使用设备: {self.device}, 重构方法: {reconstruction_method}")
    
    def select_best_modals(self, modal_features, labels, k=8):
        """模态选择"""
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
        
        # 基于Fisher比率选择最佳模态
        selected_indices = sorted(np.argsort(modal_scores)[-k:])
        print(f"选择的模态: {[i+1 for i in selected_indices]}")
        return selected_indices, modal_features[:, :, :, selected_indices]
    
    def train_classifier(self, modal_data, labels, num_classes, class_name, num_epochs=100, batch_size=32, learning_rate=0.001, use_transformer_attention=False):
        """训练单个分类器"""
        if num_classes == 2:
            # 对于二分类器，labels已经是正确的0/1标签，直接使用
            dataset = MultiModalDataset(modal_data, labels)
        else:
            # 多分类器
            dataset = MultiModalDataset(modal_data, labels)
        
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=0.3, 
                                            stratify=dataset.labels.numpy(), random_state=42)
        
        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size)
        
        model = ClassifierNet(num_classes, len(self.selected_modals)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        best_acc = 0.0
        print(f"Training {class_name} for {num_epochs} epochs...")
        
        # 使用tqdm显示训练进度
        epoch_pbar = tqdm(range(num_epochs), desc=f"Training {class_name}", unit="epoch")

        for epoch in epoch_pbar:
            # 训练
            model.train()
            for batch in train_loader:
                modal_data_batch = batch['modal_data'].to(self.device)
                labels_batch = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = model(modal_data_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    modal_data_batch = batch['modal_data'].to(self.device)
                    labels_batch = batch['label'].to(self.device)
                    outputs, _ = model(modal_data_batch)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()
            
            val_acc = 100 * val_correct / val_total

            # 保存最佳模型（每个epoch都保存最新的）
            if val_acc >= best_acc:
                best_acc = val_acc
                model_filename = self.experiment_manager.get_model_path(f'best_{class_name.replace("-", "_").replace(" ", "_")}_model.pth')
                torch.save(model.state_dict(), model_filename)
        
            # 更新进度条显示当前准确率
            epoch_pbar.set_postfix({
                'epoch': f'{epoch+1}/{num_epochs}',
                'val_acc': f'{val_acc:.2f}%',
                'best_acc': f'{best_acc:.2f}%'
            })

        # 关闭进度条
        epoch_pbar.close()

        # 加载最佳模型
        model_filename = self.experiment_manager.get_model_path(f'best_{class_name.replace("-", "_").replace(" ", "_")}_model.pth')
        model.load_state_dict(torch.load(model_filename))
        print(f"{class_name} completed, best accuracy: {best_acc:.2f}%")
        return model, torch.utils.data.Subset(dataset, val_idx)
    
    def train(self, modal_features, labels, k_modals, num_epochs=100, batch_size=32, learning_rate=0.001, use_transformer_attention=False):
        """训练并对比传统多分类器和二分类器概率重构方法"""
        # 准备数据
        self.selected_modals, selected_features = self.select_best_modals(modal_features, labels, k_modals)
        y_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        self.class_names = self.label_encoder.classes_

        modal_data = selected_features[:, :, 1, :]

        print(f"Training and comparing classifiers for {self.num_classes} classes...")
        print("="*60)

        # 1. 训练传统多分类器
        print("\n1. 训练传统多分类器...")
        self.traditional_multi_classifier, traditional_val_dataset = self.train_classifier(
            modal_data, y_encoded, self.num_classes, "traditional_multi_classifier",
            num_epochs, batch_size, learning_rate, use_transformer_attention
        )

        # 2. 训练二分类器（用于概率重构）
        print("\n2. 训练二分类器（用于概率重构）...")
        binary_histories = {}
        for class_idx, class_name in enumerate(self.class_names):
            binary_labels = [1 if label == class_idx else 0 for label in y_encoded]
            binary_classifier, _ = self.train_classifier(
                modal_data, binary_labels, 2, f"binary_{class_name}", 
                num_epochs, batch_size, learning_rate, use_transformer_attention
            )
            self.binary_classifiers[class_name] = binary_classifier
            binary_histories[class_name] = {'completed': True}

        # 3. 校准概率重构器
        if self.reconstructor.method == 'calibrated_normalize':
            print("\n3. 校准概率重构器...")
            self._calibrate_reconstructor(modal_data, y_encoded, traditional_val_dataset.indices)

        # 4. 对比两种方法
        print("\n4. 对比分类器性能...")
        best_classifier = self.compare_classifiers(traditional_val_dataset, y_encoded[traditional_val_dataset.indices])
        
        # 返回兼容格式的历史记录
        return {
            'traditional_multi_history': {'completed': True},
            'binary_histories': binary_histories,
            'comparison_results': best_classifier,
            'val_dataset': traditional_val_dataset
        }
    
    def compare_classifiers(self, val_dataset, val_labels):
        """对比传统多分类器和概率重构分类器的性能"""
        from sklearn.metrics import accuracy_score, classification_report

        # 准备验证数据
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)
        all_preds_traditional = []
        all_preds_probability = []
        all_probabilities_traditional = []
        all_probabilities_probability = []
        all_true_labels = []

        self.traditional_multi_classifier.eval()
        for classifier in self.binary_classifiers.values():
            classifier.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                modal_data_batch = batch['modal_data'].to(self.device)
                labels_batch = batch['label']

                batch_size = modal_data_batch.size(0)

                # 传统多分类器预测
                traditional_outputs, _ = self.traditional_multi_classifier(modal_data_batch)
                traditional_probs = torch.softmax(traditional_outputs, dim=1)
                _, traditional_preds = torch.max(traditional_outputs, 1)

                # 概率重构分类器预测
                binary_outputs = []
                for class_name in self.class_names:
                    output, _ = self.binary_classifiers[class_name](modal_data_batch)
                    binary_outputs.append(output)

                # 在收集完所有二分类器输出后进行概率重构
                reconstructed_probs = self.reconstructor.reconstruct_probabilities(binary_outputs)
                _, probability_preds = torch.max(reconstructed_probs, 1)

                # 调试：检查重构概率的分布
                if batch_idx == 0:  # 只打印第一个batch
                    print(f"  调试 - 重构概率分布 (batch {batch_idx}):")
                    print(f"    平均概率: {reconstructed_probs.mean(dim=0).cpu().numpy()}")
                    print(f"    预测分布: {torch.bincount(probability_preds, minlength=self.num_classes).cpu().numpy()}")
                    print(f"    标签分布: {torch.bincount(labels_batch, minlength=self.num_classes).cpu().numpy()}")
                    # 检查每个二分类器的输出
                    print(f"    二分类器输出详情:")
                    print(f"    类别先验: {getattr(self.reconstructor, 'class_priors', 'None')}")
                    print(f"    温度校准: {getattr(self.reconstructor, 'binary_temperatures', 'None')}")
                    for i, output in enumerate(binary_outputs):
                        probs = torch.softmax(output, dim=1)
                        temp = 1.0
                        if hasattr(self.reconstructor, 'binary_temperatures') and self.reconstructor.binary_temperatures:
                            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
                            if class_name in self.reconstructor.binary_temperatures:
                                temp = self.reconstructor.binary_temperatures[class_name]
                        # 使用更低的阈值来检查预测分布
                        threshold = 0.3  # 降低阈值
                        print(f"      分类器{i} ({self.class_names[i] if i < len(self.class_names) else f'class_{i}'}): temp={temp:.3f}, pos_prob_mean={probs[:, 1].mean().item():.4f}, pred_pos_ratio_05={(probs[:, 1] > 0.5).float().mean().item():.4f}, pred_pos_ratio_03={(probs[:, 1] > threshold).float().mean().item():.4f}")

                # 确保所有输出都有相同的batch_size
                assert traditional_preds.size(0) == batch_size, f"传统预测batch大小不匹配: {traditional_preds.size(0)} vs {batch_size}"
                assert probability_preds.size(0) == batch_size, f"概率预测batch大小不匹配: {probability_preds.size(0)} vs {batch_size}"
                assert labels_batch.size(0) == batch_size, f"标签batch大小不匹配: {labels_batch.size(0)} vs {batch_size}"

                all_preds_traditional.extend(traditional_preds.cpu().numpy())
                all_preds_probability.extend(probability_preds.cpu().numpy())
                all_probabilities_traditional.extend(traditional_probs.cpu().numpy())
                all_probabilities_probability.extend(reconstructed_probs.cpu().numpy())
                all_true_labels.extend(labels_batch.cpu().numpy())  # 确保在CPU上

        # 计算各种指标
        traditional_acc = accuracy_score(all_true_labels, all_preds_traditional)
        probability_acc = accuracy_score(all_true_labels, all_preds_probability)
        traditional_balanced_acc = balanced_accuracy_score(all_true_labels, all_preds_traditional)
        probability_balanced_acc = balanced_accuracy_score(all_true_labels, all_preds_probability)

        # 计算PR-AUC和MCC
        # all_true_labels 已经是编码后的数值标签，无需再次编码
        y_true_encoded = np.array(all_true_labels)
        probs_array_traditional = np.array(all_probabilities_traditional)
        probs_array_probability = np.array(all_probabilities_probability)

        # 确保所有数组的样本数量一致
        n_true = len(y_true_encoded)
        n_traditional = probs_array_traditional.shape[0]
        n_probability = probs_array_probability.shape[0]

        if n_true != n_traditional or n_true != n_probability:
            print(f"警告: 数组长度不一致 - 标签:{n_true}, 传统概率:{n_traditional}, 概率重构:{n_probability}")
            min_samples = min(n_true, n_traditional, n_probability)
            y_true_encoded = y_true_encoded[:min_samples]
            probs_array_traditional = probs_array_traditional[:min_samples]
            probs_array_probability = probs_array_probability[:min_samples]
            print(f"已截断到最小样本数: {min_samples}")

        # 确保概率数组正确归一化
        if len(probs_array_traditional.shape) > 1:
            row_sums_traditional = np.sum(probs_array_traditional, axis=1)
            if not np.allclose(row_sums_traditional, 1.0, atol=1e-6):
                probs_array_traditional = probs_array_traditional / row_sums_traditional[:, np.newaxis]
                probs_array_traditional = np.clip(probs_array_traditional, 0.0, 1.0)

        if len(probs_array_probability.shape) > 1:
            row_sums_probability = np.sum(probs_array_probability, axis=1)
            if not np.allclose(row_sums_probability, 1.0, atol=1e-6):
                probs_array_probability = probs_array_probability / row_sums_probability[:, np.newaxis]
                probs_array_probability = np.clip(probs_array_probability, 0.0, 1.0)

        if self.num_classes == 2:
            # 二分类指标
            traditional_pr_auc = auc(*precision_recall_curve(y_true_encoded, probs_array_traditional[:, 1])[::-1][:2])
            probability_pr_auc = auc(*precision_recall_curve(y_true_encoded, probs_array_probability[:, 1])[::-1][:2])
            traditional_roc_auc = roc_auc_score(y_true_encoded, probs_array_traditional[:, 1])
            probability_roc_auc = roc_auc_score(y_true_encoded, probs_array_probability[:, 1])
            traditional_f1 = f1_score(y_true_encoded, all_preds_traditional, average='binary')
            probability_f1 = f1_score(y_true_encoded, all_preds_probability, average='binary')
            traditional_mcc = matthews_corrcoef(y_true_encoded, all_preds_traditional)
            probability_mcc = matthews_corrcoef(y_true_encoded, all_preds_probability)
        else:
            # 多分类指标 (宏平均)
            traditional_pr_auc = 0
            probability_pr_auc = 0
            for i in range(self.num_classes):
                y_binary = (y_true_encoded == i).astype(int)
                prec, rec, _ = precision_recall_curve(y_binary, probs_array_traditional[:, i])
                traditional_pr_auc += auc(rec, prec)
                prec, rec, _ = precision_recall_curve(y_binary, probs_array_probability[:, i])
                probability_pr_auc += auc(rec, prec)
            traditional_pr_auc /= self.num_classes
            probability_pr_auc /= self.num_classes
            traditional_roc_auc = roc_auc_score(y_true_encoded, probs_array_traditional, multi_class='ovr', average='macro')
            probability_roc_auc = roc_auc_score(y_true_encoded, probs_array_probability, multi_class='ovr', average='macro')
            traditional_f1 = f1_score(y_true_encoded, all_preds_traditional, average='macro')
            probability_f1 = f1_score(y_true_encoded, all_preds_probability, average='macro')
            traditional_mcc = matthews_corrcoef(y_true_encoded, all_preds_traditional)
            probability_mcc = matthews_corrcoef(y_true_encoded, all_preds_probability)

        print("\n" + "="*70)
        print("                    分类器性能对比")
        print("="*70)
        print(f"{'指标':<15} {'传统多分类器':<15} {'概率重构分类器':<15}")
        print("-"*70)
        print(f"{'准确率(ACC)':<15} {traditional_acc:.4f} ({traditional_acc*100:.1f}%)    {probability_acc:.4f} ({probability_acc*100:.1f}%)")
        print(f"{'平衡准确率(BA)':<15} {traditional_balanced_acc:.4f}         {probability_balanced_acc:.4f}")
        print(f"{'ROC-AUC':<15} {traditional_roc_auc:.4f}         {probability_roc_auc:.4f}")
        print(f"{'PR-AUC':<15} {traditional_pr_auc:.4f}         {probability_pr_auc:.4f}")
        print(f"{'F1-Score':<15} {traditional_f1:.4f}         {probability_f1:.4f}")
        print(f"{'MCC':<15} {traditional_mcc:.4f}         {probability_mcc:.4f}")

        # 选择最佳分类器 (基于平衡准确率)
        if probability_balanced_acc >= traditional_balanced_acc:
            self.best_classifier_type = 'probability_reconstruction'
            print(f"[SELECTED] 选择: 概率重构分类器 (平衡准确率更优)")
        else:
            self.best_classifier_type = 'traditional_multi'
            print(f"[SELECTED] 选择: 传统多分类器 (平衡准确率更优)")

        print("="*50)

        # 打印详细报告（将数值标签转换回字符串标签用于显示）
        true_labels_str = self.label_encoder.inverse_transform(all_true_labels)
        traditional_preds_str = self.label_encoder.inverse_transform(all_preds_traditional)
        probability_preds_str = self.label_encoder.inverse_transform(all_preds_probability)

        if self.best_classifier_type == 'probability_reconstruction':
            print("\n概率重构分类器详细报告:")
            print(classification_report(true_labels_str, probability_preds_str, target_names=self.class_names))
        else:
            print("\n传统多分类器详细报告:")
            print(classification_report(true_labels_str, traditional_preds_str, target_names=self.class_names))

        return {
            'traditional_accuracy': traditional_acc,
            'traditional_balanced_accuracy': traditional_balanced_acc,
            'traditional_roc_auc': traditional_roc_auc,
            'traditional_pr_auc': traditional_pr_auc,
            'traditional_f1': traditional_f1,
            'traditional_mcc': traditional_mcc,
            'probability_accuracy': probability_acc,
            'probability_balanced_accuracy': probability_balanced_acc,
            'probability_roc_auc': probability_roc_auc,
            'probability_pr_auc': probability_pr_auc,
            'probability_f1': probability_f1,
            'probability_mcc': probability_mcc,
            'best_classifier': self.best_classifier_type,
            'traditional_predictions': all_preds_traditional,
            'probability_predictions': all_preds_probability,
            'true_labels': all_true_labels
        }

    def train_all(self, modal_features, labels, k_modals=8, num_epochs=100, batch_size=32, learning_rate=0.001, use_transformer_attention=False):
        """原train_all方法保持向后兼容"""
        return self.train(modal_features, labels, k_modals, num_epochs, batch_size, learning_rate, use_transformer_attention)
    
    def _calibrate_reconstructor(self, modal_data, labels, val_indices):
        """校准重构器"""
        val_modal_data = modal_data[val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        dataset = MultiModalDataset(val_modal_data, val_labels)
        loader = DataLoader(dataset, batch_size=32)
        
        all_binary_probs = [[] for _ in range(self.num_classes)]
        all_labels = []
        
        for classifier in self.binary_classifiers.values():
            classifier.eval()
        
        with torch.no_grad():
            for batch in loader:
                modal_data_batch = batch['modal_data'].to(self.device)
                labels_batch = batch['label']
                
                for i, class_name in enumerate(self.class_names):
                    outputs, _ = self.binary_classifiers[class_name](modal_data_batch)
                    probs = torch.softmax(outputs, dim=1)
                    all_binary_probs[i].append(probs.cpu().numpy())
                
                all_labels.extend(labels_batch.numpy())
        
        # 合并概率
        binary_probs = [np.vstack(probs) for probs in all_binary_probs]
        self.reconstructor.fit_calibration(binary_probs, np.array(all_labels))
    
    def predict(self, modal_features):
        """使用最佳分类器进行预测"""
        if self.best_classifier_type is None:
            raise ValueError("必须先训练分类器才能进行预测")

        selected_features = modal_features[:, :, :, self.selected_modals]

        # 将数据转换为与训练时相同的格式
        # 从 (n_samples, n_angles, n_time_points, n_modals) 转换为 (n_samples, n_time_points, n_angles, n_modals)
        selected_features = selected_features.transpose(0, 2, 1, 3)

        # 创建数据集，labels可以是dummy数据，因为我们只需要特征
        dummy_labels = np.zeros(len(selected_features))
        dataset = MultiModalDataset(selected_features, dummy_labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

        return self._predict_from_loader(loader)

    def predict_from_dataset(self, dataset):
        """直接从预处理的数据集进行预测（避免重复预处理）"""
        if self.best_classifier_type is None:
            raise ValueError("必须先训练分类器才能进行预测")

        loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
        return self._predict_from_loader(loader)

    def _predict_from_loader(self, loader):
        """内部预测方法，从DataLoader进行预测"""
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            if self.best_classifier_type == 'probability_reconstruction':
                # 使用概率重构方法
                for classifier in self.binary_classifiers.values():
                    classifier.eval()

                for batch_idx, batch in enumerate(loader):
                    modal_data_batch = batch['modal_data'].to(self.device)

                    # 获取二分类器输出
                    binary_outputs = []
                    for class_name in self.class_names:
                        output, _ = self.binary_classifiers[class_name](modal_data_batch)
                        binary_outputs.append(output)

                    # 重构多分类概率
                    reconstructed_probs = self.reconstructor.reconstruct_probabilities(binary_outputs)
                    _, preds = torch.max(reconstructed_probs, 1)

                    all_predictions.extend(preds.cpu().numpy())
                    all_probabilities.extend(reconstructed_probs.cpu().numpy())

            else:  # traditional_multi
                # 使用传统多分类器
                self.traditional_multi_classifier.eval()

                for batch in loader:
                    modal_data_batch = batch['modal_data'].to(self.device)

                    outputs, _ = self.traditional_multi_classifier(modal_data_batch)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    all_predictions.extend(preds.cpu().numpy())
                    all_probabilities.extend(probs.cpu().numpy())

        predicted_labels = self.label_encoder.inverse_transform(all_predictions)
        return predicted_labels, all_probabilities

    
    def predict_with_reconstruction(self, modal_features):
        """向后兼容的方法 - 使用概率重构进行预测"""
        return self.predict(modal_features)
    
    def evaluate_best_classifier(self, modal_features, true_labels):
        """评估最佳分类器的性能"""
        print(f"评估最佳分类器: {self.best_classifier_type}")

        predictions, probabilities = self.predict(modal_features)
        accuracy = accuracy_score(true_labels, predictions)
        balanced_acc = balanced_accuracy_score(true_labels, predictions)

        y_true_encoded = self.label_encoder.transform(true_labels)
        probs_array = np.array(probabilities)

        # 确保概率数组正确归一化（sklearn对多分类ROC-AUC的要求）
        if len(probs_array.shape) > 1:
            # 检查每行概率和是否为1
            row_sums = np.sum(probs_array, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                # 如果不是精确的1，进行归一化
                probs_array = probs_array / row_sums[:, np.newaxis]
                probs_array = np.clip(probs_array, 0.0, 1.0)  # 确保在[0,1]范围内

        # 计算ROC-AUC
        if self.num_classes == 2:
            roc_auc = roc_auc_score(y_true_encoded, probs_array[:, 1])
            # 计算PR-AUC (二分类)
            precision, recall, _ = precision_recall_curve(y_true_encoded, probs_array[:, 1])
            pr_auc = auc(recall, precision)
            # 计算MCC
            mcc = matthews_corrcoef(y_true_encoded, predictions)
        else:
            roc_auc = roc_auc_score(y_true_encoded, probs_array, multi_class='ovr', average='macro')
            # 多分类PR-AUC (宏平均)
            pr_auc = 0
            for i in range(self.num_classes):
                precision, recall, _ = precision_recall_curve(
                    (y_true_encoded == i).astype(int),
                    probs_array[:, i]
                )
                pr_auc += auc(recall, precision)
            pr_auc /= self.num_classes
            # 多分类MCC (使用sklearn的实现)
            mcc = matthews_corrcoef(y_true_encoded, predictions)

        print(f"准确率: {accuracy:.4f}, 平衡准确率: {balanced_acc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, MCC: {mcc:.4f}")

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'mcc': mcc,
            'predictions': predictions,
            'probabilities': probabilities
        }

    def evaluate_all_methods(self, modal_features, true_labels):
        """评估所有重构方法（仅当使用概率重构时）"""
        if self.best_classifier_type != 'probability_reconstruction':
            print("当前使用传统多分类器，跳过重构方法评估")
            return self.evaluate_best_classifier(modal_features, true_labels)

        methods = ['simple_normalize', 'calibrated_normalize', 'geometric_normalize', 
                  'softmax_temperature', 'max_confidence', 'threshold_based']
        results = {}
        
        for method in methods:
            print(f"评估 {method}...")
            original_method = self.reconstructor.method
            self.reconstructor.method = method
            
            try:
                predictions, probabilities = self.predict(modal_features)
                accuracy = accuracy_score(true_labels, predictions)
                balanced_acc = balanced_accuracy_score(true_labels, predictions)
                
                y_true_encoded = self.label_encoder.transform(true_labels)
                probs_array = np.array(probabilities)

                # 确保概率数组正确归一化
                if len(probs_array.shape) > 1:
                    row_sums = np.sum(probs_array, axis=1)
                    if not np.allclose(row_sums, 1.0, atol=1e-6):
                        probs_array = probs_array / row_sums[:, np.newaxis]
                        probs_array = np.clip(probs_array, 0.0, 1.0)

                if self.num_classes == 2:
                    roc_auc_val = roc_auc_score(y_true_encoded, probs_array[:, 1])
                    precision, recall, _ = precision_recall_curve(y_true_encoded, probs_array[:, 1])
                    pr_auc_val = auc(recall, precision)
                    mcc_val = matthews_corrcoef(y_true_encoded, predictions)
                else:
                    roc_auc_val = roc_auc_score(y_true_encoded, probs_array, multi_class='ovr', average='macro')
                    pr_auc_val = 0
                    for i in range(self.num_classes):
                        precision, recall, _ = precision_recall_curve(
                            (y_true_encoded == i).astype(int), probs_array[:, i])
                        pr_auc_val += auc(recall, precision)
                    pr_auc_val /= self.num_classes
                    mcc_val = matthews_corrcoef(y_true_encoded, predictions)

                results[method] = {
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'roc_auc': roc_auc_val,
                    'pr_auc': pr_auc_val,
                    'mcc': mcc_val,
                    'predictions': predictions
                }
                print(f"  准确率: {accuracy:.4f}, 平衡准确率: {balanced_acc:.4f}, PR-AUC: {pr_auc_val:.4f}, MCC: {mcc_val:.4f}")
                
            except Exception as e:
                print(f"  评估失败: {e}")
                results[method] = None
            
            self.reconstructor.method = original_method
        
        return results
    
    def plot_comparison(self, results):
        """绘制方法比较"""
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        methods = list(valid_results.keys())
        accuracies = [valid_results[m]['accuracy'] for m in methods]
        aucs = [valid_results[m]['auc'] for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 准确率
        bars1 = ax1.bar(methods, accuracies, alpha=0.7)
        ax1.set_title('重构方法准确率比较')
        ax1.set_ylabel('准确率')
        ax1.tick_params(axis='x', rotation=45)
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # AUC
        bars2 = ax2.bar(methods, aucs, alpha=0.7, color='orange')
        ax2.set_title('重构方法AUC比较')
        ax2.set_ylabel('AUC')
        ax2.tick_params(axis='x', rotation=45)
        for bar, auc in zip(bars2, aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        best_method = max(valid_results.keys(), key=lambda x: valid_results[x]['balanced_accuracy'])
        print(f"\n最佳重构方法: {best_method}")
        print(f"平衡准确率: {valid_results[best_method]['balanced_accuracy']:.4f}")
        print(f"PR-AUC: {valid_results[best_method]['pr_auc']:.4f}, MCC: {valid_results[best_method]['mcc']:.4f}")
        return best_method

def main(use_simulated_data=False):
    """主函数
    Args:
        use_simulated_data: 是否使用模拟数据进行测试
    """
    import numpy as np

    # 如果使用模拟数据，跳过真实数据加载
    if use_simulated_data:
        print("[TEST MODE] 使用模拟数据进行测试...")

        # 生成模拟数据
        import numpy as np
        np.random.seed(42)  # 确保结果可重现

        n_samples = 200  # 较小的样本数量用于快速测试
        n_modals = 16
        feature_dim = 100

        # 生成模拟的多模态特征 - 按照真实数据格式: (n_samples, n_angles, n_time_points, n_modals)
        n_angles = 36  # 模拟36个角度
        n_time_points = 10  # 每个角度10个时间点

        # 生成模拟数据: (n_samples, n_angles, n_time_points, n_modals)
        modal_features = np.random.randn(n_samples, n_angles, n_time_points, n_modals)

        # 生成模拟标签 (3个类别)
        labels = np.random.randint(0, 3, n_samples)
        labels = [str(label) for label in labels]  # 转换为字符串标签

        print(f"[TEST MODE] 模拟数据生成完成: {modal_features.shape}, 标签数量: {len(labels)}")

        # 创建简单的实验管理器用于测试
        test_exp_manager = ExperimentManager()
        test_exp_manager.start_experiment(experiment_name='test_mode', config={'test': True})
        k_modals = 2
        # 创建分类器
        classifier = ProbabilityReconstructionClassifier(
            k_modals=k_modals,
            reconstruction_method='simple_normalize',
            experiment_manager=test_exp_manager
        )

        # 训练参数
        use_transformer = False
        histories = classifier.train(
            modal_features, labels,
            k_modals=k_modals,
            use_transformer_attention=use_transformer,
            learning_rate=0.001,  # 更高的学习率用于快速收敛
            batch_size=64,
            num_epochs=100  # 增加训练轮数以获得更好的收敛
        )

        # 评估 - 使用训练时的验证集进行评估
        print("[TEST MODE] 评估模型性能...")
        val_dataset = histories['val_dataset']
        val_indices = val_dataset.indices
        val_labels = np.array(labels)[val_indices]

        predictions, probabilities = classifier.predict_from_dataset(val_dataset)

        # 计算基本指标
        accuracy = accuracy_score(val_labels, predictions)
        print(f"[TEST MODE] 准确率: {accuracy:.4f}")
        print("[TEST MODE] 测试完成!")
        return classifier, None

    from DataCleanandSave16Modal_label import IntegratedIMCProcessor

    # 初始化实验管理器
    exp_manager = ExperimentManager()
    k_modals = 16  # 使用的模态数量
    # 开始实验
    experiment_config = {
        'experiment_name': 'imc_attention_classifier',
        'description': '基于注意力机制的IMC多模态分类器实验',
        'model_config': {
            'reconstruction_method': 'simple_normalize',
            'num_modals': k_modals,  # 模态数量
            'modal_selection': 'fisher_ratio_top_8'
        },
        'training_config': {
            'learning_rate': 0.0001,
            'batch_size': 64,
            'num_epochs': 200
        }
    }

    exp_id = exp_manager.start_experiment(experiment_name='imc_classifier', config=experiment_config)
    
    # 加载数据
    processor = IntegratedIMCProcessor('', 'data/processed')
    results = processor.load_complete_results('processed_imc_data')
    
    if not results:
        raise FileNotFoundError(
            "数据加载失败！请先运行数据预处理脚本生成训练数据:\n"
            "python script/DataCleanandSave16Modal_label.py"
        )

    modal_features = results['multi_modal_features']
    labels = results['labels']
    print(f"Data shape: {modal_features.shape}, Labels: {len(labels)}")

    # 创建分类器
    classifier = ProbabilityReconstructionClassifier(
        k_modals=k_modals,
        reconstruction_method='simple_normalize',
        experiment_manager=exp_manager
    )

    # 按照要求的格式训练

    use_transformer = True  # 设置为True使用Transformer注意力，False使用原始注意力
    histories = classifier.train(
        modal_features, labels,
        k_modals=k_modals,
        num_epochs=100,
        learning_rate=0.0001,
        use_transformer_attention=use_transformer
    )

    # 更新配置中的最终参数
    experiment_config['model_config']['num_modals'] = k_modals
    experiment_config['training_config']['batch_size'] = 16
    experiment_config['training_config']['learning_rate'] = 0.0001
    experiment_config['training_config']['num_epochs'] = 100  # 实际使用的训练轮数

    # 获取验证集 - 使用训练时保存的验证集（已正确预处理）
    val_dataset = histories['val_dataset']
    # 从Subset对象中提取原始数据索引，用于获取对应的标签
    val_indices = val_dataset.indices
    val_labels = np.array(labels)[val_indices]

    # 评估最佳分类器 - 使用训练时的验证集（避免重复预处理）
    print("\nEvaluating best classifier performance...")
    predictions, probabilities = classifier.predict_from_dataset(val_dataset)

    # 计算最终指标
    accuracy = accuracy_score(val_labels, predictions)
    balanced_acc = balanced_accuracy_score(val_labels, predictions)

    # 计算ROC-AUC, PR-AUC, MCC
    y_true_encoded = classifier.label_encoder.transform(val_labels)  # 用于ROC-AUC计算
    y_pred_encoded = classifier.label_encoder.transform(predictions)  # 预测结果也需要编码
    probs_array = np.array(probabilities)

    # 确保概率数组正确归一化（sklearn对多分类ROC-AUC的要求）
    if len(probs_array.shape) > 1:
        row_sums = np.sum(probs_array, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            probs_array = probs_array / row_sums[:, np.newaxis]
            probs_array = np.clip(probs_array, 0.0, 1.0)

    if classifier.num_classes == 2:
        roc_auc = roc_auc_score(y_true_encoded, probs_array[:, 1])
        precision, recall, _ = precision_recall_curve(y_true_encoded, probs_array[:, 1])
        pr_auc = auc(recall, precision)
        mcc = matthews_corrcoef(y_true_encoded, y_pred_encoded)
    else:
        roc_auc = roc_auc_score(y_true_encoded, probs_array, multi_class='ovr', average='macro')
        pr_auc = 0
        for i in range(classifier.num_classes):
            precision, recall, _ = precision_recall_curve(
                (y_true_encoded == i).astype(int), probs_array[:, i])
            pr_auc += auc(recall, precision)
        pr_auc /= classifier.num_classes
        mcc = matthews_corrcoef(y_true_encoded, y_pred_encoded)

    print(f"准确率: {accuracy:.4f}, 平衡准确率: {balanced_acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, MCC: {mcc:.4f}")

    best_method = classifier.best_classifier_type

    print(f"\nBest method ({best_method}) detailed report:")
    print(classification_report(val_labels, predictions, target_names=classifier.class_names))

    # 记录最终指标
    final_metrics = {
        'best_reconstruction_method': best_method,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'num_classes': classifier.num_classes,
        'selected_modals': len(classifier.selected_modals) if classifier.selected_modals else 0
    }

    # 保存最终的实验配置（包含所有最终参数）
    exp_manager.save_config(experiment_config)

    # 保存图表
    exp_manager.save_plot('reconstruction_methods_comparison.png')

    # 结束实验
    exp_manager.finish_experiment(final_metrics)
    
    return classifier, results

if __name__ == "__main__":
    import sys

    # 检查命令行参数
    use_test_mode = '--test' in sys.argv or '-t' in sys.argv

    if use_test_mode:
        print("启用测试模式，使用模拟数据...")
        classifier, results = main(use_simulated_data=True)
    else:
        print("运行完整实验模式...")
        classifier, results = main(use_simulated_data=False)