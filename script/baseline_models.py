"""
基线模型测试脚本
测试各种传统机器学习和简单深度学习模型的性能
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: xgboost未安装，将跳过XGBoost模型")
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# 导入数据处理模块
sys.path.append(os.path.dirname(__file__))
from DataCleanandSave16Modal_label import IntegratedIMCProcessor

class ExperimentManager:
    """实验管理器"""
    def __init__(self, base_dir='result/baseline'):
        self.base_dir = base_dir
        self.experiment_id = None
        self.experiment_dir = None
        self.start_time = None

    def start_experiment(self, experiment_name=None, config=None):
        """开始实验"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            self.experiment_id = f"{experiment_name}_{timestamp}"
        else:
            self.experiment_id = f"baseline_experiment_{timestamp}"

        self.experiment_dir = f"{self.base_dir}/{self.experiment_id}"
        os.makedirs(f'{self.experiment_dir}/metrics', exist_ok=True)
        os.makedirs(f'{self.experiment_dir}/plots', exist_ok=True)

        self.start_time = datetime.now()

        # 保存配置
        if config:
            config_path = f'{self.experiment_dir}/config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] Config file saved: {config_path}")

        return self.experiment_id

    def save_metrics(self, model_name, metrics):
        """保存指标"""
        if self.experiment_dir:
            metrics_path = f'{self.experiment_dir}/metrics/{model_name}_metrics.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

    def save_plot(self, filename):
        """保存图表"""
        if self.experiment_dir:
            plt.savefig(f'{self.experiment_dir}/plots/{filename}', dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] Plot saved: {self.experiment_dir}/plots/{filename}")

    def finish_experiment(self):
        """结束实验"""
        if self.start_time:
            duration = datetime.now() - self.start_time
            summary = {
                'experiment_id': self.experiment_id,
                'duration_seconds': duration.total_seconds(),
                'completed_at': datetime.now().isoformat()
            }
            summary_path = f'{self.experiment_dir}/experiment_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] Experiment summary saved: {summary_path}")

class SimpleCNN(nn.Module):
    """简单CNN模型"""
    def __init__(self, input_dim, num_classes, sequence_length=81):
        super().__init__()
        # input_dim 是通道数（模态数）
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(8)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimpleLSTM(nn.Module):
    """简单LSTM模型"""
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # 取最后一层的输出
        x = self.classifier(x)
        return x

class SimpleTransformer(nn.Module):
    """简单Transformer模型 - 三层注意力"""
    def __init__(self, input_dim, num_classes, seq_length=81, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.d_model = d_model

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size = x.size(0)

        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)

        # 添加位置编码
        x = x + self.positional_encoding[:, :self.seq_length, :]  # (batch_size, seq_length, d_model)

        # Transformer编码器
        x = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)

        # 全局平均池化
        x = torch.mean(x, dim=1)  # (batch_size, d_model)

        # 分类
        x = self.classifier(x)  # (batch_size, num_classes)
        return x

class SimpleMLP(nn.Module):
    """简单MLP模型"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class BaselineModels:
    """基线模型训练和评估"""

    def __init__(self, experiment_manager=None):
        self.experiment_manager = experiment_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def prepare_data(self, modal_features, labels, k_modals=8):
        """数据预处理"""
        print(f"Original data shape: {modal_features.shape}")

        # 选择最佳模态 (使用简单的方差选择)
        modal_variances = []
        for i in range(modal_features.shape[3]):
            modal_data = modal_features[:, :, 1, i]  # 使用角度数据
            variance = np.var(modal_data, axis=(0, 1))
            modal_variances.append(np.mean(variance))

        # 选择方差最大的k个模态
        selected_indices = np.argsort(modal_variances)[-k_modals:]
        selected_features = modal_features[:, :, :, selected_indices]

        print(f"Selected modalities: {[i+1 for i in selected_indices]}")

        # 展平数据用于传统机器学习模型
        n_samples, seq_len, n_angles, n_modals = selected_features.shape
        X_flat = selected_features.reshape(n_samples, -1)  # (n_samples, seq_len * n_angles * n_modals)

        # 标准化
        X_scaled = self.scaler.fit_transform(X_flat)

        # 编码标签
        y_encoded = self.label_encoder.fit_transform(labels)

        print(f"Processed data shape: {X_scaled.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")

        return X_scaled, y_encoded

    def prepare_data_for_nn(self, modal_features, labels, k_modals=8):
        """为神经网络准备数据"""
        # 选择模态
        modal_variances = []
        for i in range(modal_features.shape[3]):
            modal_data = modal_features[:, :, 1, i]
            variance = np.var(modal_data, axis=(0, 1))
            modal_variances.append(np.mean(variance))

        selected_indices = np.argsort(modal_variances)[-k_modals:]
        selected_features = modal_features[:, :, :, selected_indices]

        # 为CNN准备数据: (batch, channels, seq_len)
        X_cnn = selected_features[:, :, 1, :]  # 使用角度数据
        X_cnn = X_cnn.transpose(0, 2, 1)  # (n_samples, n_modals, seq_len)

        # 为LSTM准备数据: (batch, seq_len, features)
        X_lstm = selected_features[:, :, :, 0]  # 使用力矩数据
        X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], -1)

        # 为MLP准备数据: 展平
        X_mlp = selected_features.reshape(selected_features.shape[0], -1)

        y_encoded = self.label_encoder.fit_transform(labels)

        return {
            'cnn': X_cnn,
            'lstm': X_lstm,
            'mlp': X_mlp,
            'labels': y_encoded
        }

    def train_sklearn_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """训练sklearn模型"""
        print(f"\nTraining {model_name}...")

        # 训练
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 计算概率（如果支持）
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None

        # 计算指标
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)

        if self.experiment_manager:
            self.experiment_manager.save_metrics(model_name, metrics)

        print(f"{model_name} accuracy: {metrics['accuracy']:.4f}")
        print(f"{model_name} balanced accuracy: {metrics['balanced_accuracy']:.4f}")

        return metrics

    def train_nn_model(self, model_class, X_train, y_train, X_test, y_test, model_name, input_dim=None):
        """训练神经网络模型"""
        print(f"\nTraining {model_name}...")

        # 创建模型
        num_classes = len(np.unique(y_train))
        if 'CNN' in model_name:
            model = model_class(input_dim, num_classes).to(self.device)
        elif 'LSTM' in model_name:
            model = model_class(input_dim, num_classes).to(self.device)
        else:  # MLP
            model = model_class(input_dim, num_classes).to(self.device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 转换为tensor
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)

        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 训练
        model.train()
        for epoch in range(50):  # 较少的训练轮数用于基线测试
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # 评估
        model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算指标
        metrics = self.calculate_metrics(y_test, np.array(all_preds), np.array(all_probs))

        if self.experiment_manager:
            self.experiment_manager.save_metrics(model_name, metrics)

        print(f"{model_name} accuracy: {metrics['accuracy']:.4f}")
        print(f"{model_name} balanced accuracy: {metrics['balanced_accuracy']:.4f}")

        return metrics

    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """计算评估指标"""
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        mcc = matthews_corrcoef(y_true, y_pred)

        # ROC-AUC
        if y_prob is not None and y_prob.shape[1] > 1:
            try:
                if len(np.unique(y_true)) == 2:
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except:
                roc_auc = None
        else:
            roc_auc = None

        # PR-AUC
        if y_prob is not None and y_prob.shape[1] > 1:
            try:
                if len(np.unique(y_true)) == 2:
                    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                    pr_auc = auc(recall, precision)
                else:
                    # 多分类PR-AUC：计算每个类别的PR-AUC然后平均
                    pr_auc_values = []
                    for class_idx in range(y_prob.shape[1]):
                        y_true_binary = (y_true == class_idx).astype(int)
                        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, class_idx])
                        if len(np.unique(y_true_binary)) > 1:  # 确保有正样本和负样本
                            pr_auc_values.append(auc(recall, precision))
                    pr_auc = np.mean(pr_auc_values) if pr_auc_values else None
            except:
                pr_auc = None
        else:
            pr_auc = None

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'mcc': mcc,
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }

    def run_baseline_comparison(self, modal_features, labels, k_modals=8, test_size=0.2):
        """运行所有基线模型比较"""
        print("="*60)
        print("Starting Baseline Models Performance Comparison")
        print("="*60)

        # 数据分割
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(indices, test_size=test_size,
                                               stratify=labels, random_state=42)

        # 为传统ML准备数据
        X_ml, y_ml = self.prepare_data(modal_features, labels, k_modals)
        X_train_ml, X_test_ml = X_ml[train_idx], X_ml[test_idx]
        y_train_ml, y_test_ml = y_ml[train_idx], y_ml[test_idx]

        # 为神经网络准备数据
        data_nn = self.prepare_data_for_nn(modal_features, labels, k_modals)
        y_nn = data_nn['labels']
        y_train_nn, y_test_nn = y_nn[train_idx], y_nn[test_idx]

        # 定义模型
        models = {
            # 传统机器学习模型
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GBM': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'MLP_sklearn': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        }

        # 条件性添加XGBoost
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42)

        # 神经网络模型
        nn_models = {
            'SimpleCNN': (SimpleCNN, data_nn['cnn'][train_idx], data_nn['cnn'][test_idx]),
            'SimpleLSTM': (SimpleLSTM, data_nn['lstm'][train_idx], data_nn['lstm'][test_idx]),
            'SimpleTransformer': (SimpleTransformer, data_nn['lstm'][train_idx], data_nn['lstm'][test_idx]),  # 使用与LSTM相同的输入格式
            'SimpleMLP': (SimpleMLP, data_nn['mlp'][train_idx], data_nn['mlp'][test_idx])
        }

        results = {}

        # 训练传统ML模型
        print("\n" + "="*40)
        print("Training Traditional Machine Learning Models")
        print("="*40)

        for model_name, model in models.items():
            try:
                results[model_name] = self.train_sklearn_model(
                    model, X_train_ml, y_train_ml, X_test_ml, y_test_ml, model_name
                )
            except Exception as e:
                print(f"Training {model_name} failed: {e}")
                results[model_name] = None

        # 训练神经网络模型
        print("\n" + "="*40)
        print("Training Neural Network Models")
        print("="*40)

        for model_name, (model_class, X_train_nn, X_test_nn) in nn_models.items():
            try:
                if 'CNN' in model_name:
                    input_dim = X_train_nn.shape[1]  # 通道数
                elif 'LSTM' in model_name or 'Transformer' in model_name:
                    input_dim = X_train_nn.shape[-1]  # 特征数
                else:  # MLP
                    input_dim = X_train_nn.shape[1]  # 展平后的特征数
                results[model_name] = self.train_nn_model(
                    model_class, X_train_nn, y_train_nn, X_test_nn, y_test_nn, model_name, input_dim
                )
            except Exception as e:
                print(f"Training {model_name} failed: {e}")
                results[model_name] = None

        # Generate comparison plot
        self.plot_comparison(results)

        return results

    def plot_comparison(self, results):
        """绘制模型比较图表"""
        valid_results = {k: v for k, v in results.items() if v is not None}

        if not valid_results:
            print("没有有效的结果可以绘制")
            return

        models = list(valid_results.keys())
        accuracies = [valid_results[m]['accuracy'] for m in models]
        balanced_accuracies = [valid_results[m]['balanced_accuracy'] for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Accuracy comparison
        bars1 = ax1.bar(range(len(models)), accuracies, alpha=0.7)
        ax1.set_title('Baseline Models Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')

        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

        # Balanced accuracy comparison
        bars2 = ax2.bar(range(len(models)), balanced_accuracies, alpha=0.7, color='orange')
        ax2.set_title('Baseline Models Balanced Accuracy Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Balanced Accuracy', fontsize=12)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')

        for bar, acc in zip(bars2, balanced_accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if self.experiment_manager:
            self.experiment_manager.save_plot('baseline_models_comparison.png')

        # 不显示图片，直接关闭
        plt.close()

        # 打印详细结果
        print("\n" + "="*60)
        print("Baseline Models Performance Summary")
        print("="*60)
        print("<10")
        print("-" * 60)

        for model_name, metrics in valid_results.items():
            if metrics:
                acc = metrics.get('accuracy', 0)
                bal_acc = metrics.get('balanced_accuracy', 0)
                f1 = metrics.get('f1_score', 0)
                roc_auc = metrics.get('roc_auc', 'N/A')
                pr_auc = metrics.get('pr_auc', 'N/A')
                mcc = metrics.get('mcc', 0)
                print("<10")
            else:
                print("<10")

def main():
    """主函数"""
    # 初始化实验管理器
    exp_manager = ExperimentManager()

    # 实验配置
    base_models = [
        'SVM', 'RandomForest', 'GBM', 'KNN',
        'LogisticRegression', 'MLP_sklearn',
        'SimpleCNN', 'SimpleLSTM', 'SimpleTransformer', 'SimpleMLP'
    ]

    if XGBOOST_AVAILABLE:
        base_models.insert(3, 'XGBoost')  # 在GBM之后插入XGBoost

    experiment_config = {
        'experiment_name': 'baseline_models_comparison',
        'description': '各种基线模型性能比较实验',
        'model_config': {
            'k_modals': 8,
            'test_size': 0.2,
            'random_state': 42
        },
        'models': base_models
    }

    # 开始实验
    exp_id = exp_manager.start_experiment(config=experiment_config)
    print(f"Starting experiment: {exp_id}")
    print(f"Experiment directory: {exp_manager.experiment_dir}")

    try:
        # 加载数据
        print("\nLoading processed results...")
        processor = IntegratedIMCProcessor('', 'data/processed')
        results = processor.load_complete_results('processed_imc_data')

        if not results:
            raise FileNotFoundError("Cannot load data, please run data preprocessing script first")

        modal_features = results['multi_modal_features']
        labels = results['labels']

        print(f"Data shape: {modal_features.shape}, Number of labels: {len(labels)}")

        # 创建基线模型评估器
        baseline_evaluator = BaselineModels(exp_manager)

        # 运行基线模型比较
        results = baseline_evaluator.run_baseline_comparison(
            modal_features, labels, k_modals=8, test_size=0.2
        )

        # 保存最终结果
        final_results = {
            'experiment_id': exp_id,
            'total_models': len(results),
            'successful_models': len([r for r in results.values() if r is not None]),
            'best_model': max(results.keys(), key=lambda x: results[x]['balanced_accuracy'] if results[x] else 0)
        }

        with open(f'{exp_manager.experiment_dir}/final_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"\n[SUCCESS] Experiment completed! Results saved to: {exp_manager.experiment_dir}")

    except Exception as e:
        print(f"[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 结束实验
        exp_manager.finish_experiment()

if __name__ == "__main__":
    main()
