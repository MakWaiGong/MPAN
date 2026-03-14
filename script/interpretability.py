import pandas as pd
import numpy as np
import os
import json
import pickle
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
from scipy import stats
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, precision_recall_curve, auc
from tqdm import tqdm
from datetime import datetime

class IntegratedIMCProcessor:
    """
    整合的IMC数据处理器：结合数据归一化、异常检测、模态特征提取和标签保存
    """
    
    def __init__(self, data_path, output_dir=None):
        self.data_path = data_path
        if output_dir is None:
            # 默认使用相对于脚本位置的data/processed目录
            output_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
        self.output_dir = output_dir
        self.raw_data = None
        self.normalized_results = {}
        self.angles = np.linspace(90, 10, 81)

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """加载Excel数据"""
        print("加载数据...")
        try:
            self.raw_data = pd.read_excel(self.data_path)
            print(f"数据加载成功: {self.raw_data.shape}")
            print(f"列名: {list(self.raw_data.columns)}")
            return True
        except Exception as e:
            print(f"加载失败: {e}")
            return False
    
    def normalize_torque(self, torques):
        """扭矩归一化方法"""
        torques = np.array(torques, dtype=float)
        
        # 处理负值：加上最小负值的绝对值
        min_torque = np.min(torques)
        if min_torque < 0:
            offset = abs(min_torque)
            adjusted_torques = torques + offset
        else:
            adjusted_torques = torques
            offset = 0
        
        # 归一化到0-100%
        max_adjusted = np.max(adjusted_torques)
        if max_adjusted > 0:
            normalized = (adjusted_torques / max_adjusted) * 100
        else:
            normalized = adjusted_torques
            
        return normalized, offset, max_adjusted
    
    def process_sequence(self, angles, torques, imc_type):
        """处理单个序列并保留标签"""
        # 基础清理
        angles = np.array(angles, dtype=float)
        torques = np.array(torques, dtype=float)
        
        # 移除无效值
        valid_mask = (np.isfinite(angles) & np.isfinite(torques) & 
                     (angles >= 0) & (angles <= 120))
        
        clean_angles = angles[valid_mask]
        clean_torques = torques[valid_mask]
        
        if len(clean_angles) < 5:
            return None
        
        # 排序
        sorted_indices = np.argsort(clean_angles)
        sorted_angles = clean_angles[sorted_indices]
        sorted_torques = clean_torques[sorted_indices]
        
        # 插值到标准角度范围
        angle_min = max(10, np.min(sorted_angles))
        angle_max = min(90, np.max(sorted_angles))
        
        if angle_max - angle_min < 10:
            return None
            
        try:
            # 插值
            standard_angles = np.arange(angle_min, angle_max + 1, 1.0)
            f_interp = interp1d(sorted_angles, sorted_torques, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
            interpolated_torques = f_interp(standard_angles)
            
            # 归一化
            normalized_torques, offset, max_value = self.normalize_torque(interpolated_torques)
            
            return {
                'angles': standard_angles,
                'normalized_torques': normalized_torques,
                'original_torques': interpolated_torques,
                'imc_type': imc_type,  # 保存标签
                'offset': offset,
                'max_value': max_value
            }
            
        except Exception as e:
            print(f"处理序列时出错: {e}")
            return None
    
    def process_all_data(self):
        """处理所有数据，保留标签"""
        print("开始处理所有数据...")
        
        if self.raw_data is None:
            if not self.load_data():
                return False
        
        # 检查必需列
        required_cols = ['ID', 'currentrepetition', '患侧类型', 'torque', 'angle']
        missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
        if missing_cols:
            print(f"缺少必要列: {missing_cols}")
            return False
        
        # 处理每个患者的每次重复
        processed_count = 0
        
        for patient_id in self.raw_data['ID'].unique():
            if pd.isna(patient_id):
                continue
                
            patient_data = self.raw_data[self.raw_data['ID'] == patient_id]
            patient_results = {}
            
            for repetition in patient_data['currentrepetition'].unique():
                if pd.isna(repetition):
                    continue
                    
                rep_data = patient_data[patient_data['currentrepetition'] == repetition]
                
                if len(rep_data) >= 5:
                    angles = rep_data['angle'].values
                    torques = rep_data['torque'].values
                    imc_type = rep_data['患侧类型'].iloc[0]  # 获取患侧类型标签
                    
                    result = self.process_sequence(angles, torques, imc_type)
                    
                    if result is not None:
                        patient_results[f'rep_{repetition}'] = {
                            'angles': result['angles'].tolist(),
                            'normalized_torques': result['normalized_torques'].tolist(),
                            'original_torques': result['original_torques'].tolist(),
                            'imc_type': imc_type,  # 保存标签
                            'offset': result['offset'],
                            'max_value': result['max_value'],
                            'data_points': len(result['angles'])
                        }
                        processed_count += 1
            
            if patient_results:
                self.normalized_results[f'patient_{patient_id}'] = patient_results
        
        print(f"处理完成: {processed_count} 个序列")
        return True
    
    def convert_to_data_pairs_with_labels(self):
        """转换为data_pairs格式并保留标签"""
        if not self.normalized_results:
            print("没有处理结果可转换")
            return None, None
        
        data_pairs_list = []
        labels_list = []
        sequence_info = []
        
        for patient_key, patient_data in self.normalized_results.items():
            patient_id = patient_key.replace('patient_', '')
            
            for rep_key, rep_data in patient_data.items():
                rep_id = rep_key.replace('rep_', '')
                
                # 创建标准81点数据
                standard_data_pair = np.zeros((81, 2))
                standard_data_pair[:, 0] = self.angles  # 角度
                
                # 插值到标准81点
                angles = np.array(rep_data['angles'])
                torques = np.array(rep_data['normalized_torques'])
                
                # 确保torques被正确归一化到0-1范围
                torques_min = np.min(torques)
                torques_max = np.max(torques)
                if torques_max > torques_min:
                    torques_normalized = (torques - torques_min) / (torques_max - torques_min)
                else:
                    torques_normalized = torques / 100.0  # 如果已经是百分比形式
                
                # 插值到标准角度序列
                f_interp = interp1d(angles, torques_normalized, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
                standard_torques = f_interp(self.angles)
                
                # 确保数值在合理范围内
                standard_torques = np.clip(standard_torques, 0, 1)
                standard_data_pair[:, 1] = standard_torques
                
                data_pairs_list.append(standard_data_pair)
                labels_list.append(rep_data['imc_type'])
                sequence_info.append({
                    'patient_id': patient_id,
                    'repetition': rep_id,
                    'sequence_id': f"{patient_id}_{rep_id}",
                    'imc_type': rep_data['imc_type']
                })
        
        data_pairs = np.array(data_pairs_list)
        labels = np.array(labels_list)
        
        print(f"转换完成: {data_pairs.shape[0]} 个序列，标签分布:")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count} 个样本")
        
        return data_pairs, labels, sequence_info
    
    def detect_and_remove_outliers(self, data_pairs, labels, sequence_info, method='gradient2'):
        """检测并删除异常样本，同时保持标签对应"""
        print(f"开始异常检测，方法: {method}")
        print(f"原始数据形状: {data_pairs.shape}")

        n_samples = data_pairs.shape[0]
        outlier_indices = []
        outlier_stats = {}

        # 提取所有样本的torque曲线
        torque_curves = data_pairs[:, :, 1]  # (n_samples, 81)

        # 基于梯度的检测
        if method in ['gradient', 'gradient2', 'combined']:
            print("应用基于梯度的检测...")
            gradients = np.gradient(torque_curves, axis=1)
            gradient_variations = np.var(gradients, axis=1)
            gradient_threshold = np.percentile(gradient_variations, 3.9)

            gradient_outliers = np.where(gradient_variations < gradient_threshold)[0]

            if method == 'gradient2':
                # 检测连续相近梯度段
                segment_outliers = []
                for i in range(n_samples):
                    gradient = gradients[i]
                    outlier_score = self.detect_similar_gradient_segments(gradient)
                    if outlier_score > 0.6:
                        segment_outliers.append(i)

                all_gradient_outliers = list(set(gradient_outliers.tolist() + segment_outliers))
                outlier_indices.extend(all_gradient_outliers)
                
                outlier_stats['gradient'] = {
                    'variance_threshold': gradient_threshold,
                    'variance_outliers': gradient_outliers,
                    'segment_outliers': np.array(segment_outliers),
                    'total_outliers': np.array(all_gradient_outliers)
                }
            else:
                outlier_indices.extend(gradient_outliers)
                outlier_stats['gradient'] = {
                    'threshold': gradient_threshold,
                    'outliers': gradient_outliers
                }

        # 合并并去重异常索引
        outlier_indices = list(set(outlier_indices))
        outlier_indices.sort()

        print(f"检测到异常样本: {len(outlier_indices)} 个")

        # 删除异常样本并保持标签对应
        if len(outlier_indices) > 0:
            clean_indices = [i for i in range(n_samples) if i not in outlier_indices]
            cleaned_data_pairs = data_pairs[clean_indices]
            cleaned_labels = labels[clean_indices]
            cleaned_sequence_info = [sequence_info[i] for i in clean_indices]
            
            print(f"清理后数据形状: {cleaned_data_pairs.shape}")
            print(f"删除了 {len(outlier_indices)} 个异常样本 ({len(outlier_indices)/n_samples*100:.1f}%)")
            
            # 打印清理后的标签分布
            unique_labels, counts = np.unique(cleaned_labels, return_counts=True)
            print("清理后标签分布:")
            for label, count in zip(unique_labels, counts):
                print(f"  {label}: {count} 个样本")
        else:
            cleaned_data_pairs = data_pairs
            cleaned_labels = labels
            cleaned_sequence_info = sequence_info
            print("未检测到异常样本")

        return cleaned_data_pairs, cleaned_labels, cleaned_sequence_info, outlier_indices, outlier_stats
    
    def detect_similar_gradient_segments(self, gradient, similarity_threshold=0.001, min_segment_length=5):
        """检测曲线中连续相近梯度的段落"""
        n_points = len(gradient)
        if n_points < min_segment_length:
            return 0.0

        similar_segments = []
        current_segment_start = 0

        for i in range(1, n_points):
            gradient_diff = abs(gradient[i] - gradient[current_segment_start])

            if gradient_diff > similarity_threshold:
                segment_length = i - current_segment_start
                if segment_length >= min_segment_length:
                    similar_segments.append((current_segment_start, i-1, segment_length))
                current_segment_start = i

        final_segment_length = n_points - current_segment_start
        if final_segment_length >= min_segment_length:
            similar_segments.append((current_segment_start, n_points-1, final_segment_length))

        total_similar_points = sum(segment[2] for segment in similar_segments)
        outlier_score = total_similar_points / n_points

        return outlier_score
    
    def compute_multi_modal_features(self, data_pairs):
        """计算16种模态特征"""
        n_samples, n_points, n_dims = data_pairs.shape
        multi_modal_data = np.zeros((n_samples, n_points, n_dims, 16))

        print("计算16种模态特征...")

        for sample_idx in range(n_samples):
            angles = data_pairs[sample_idx, :, 0]
            torque = data_pairs[sample_idx, :, 1]

            # 模态1: 原始数据
            multi_modal_data[sample_idx, :, 0, 0] = angles
            multi_modal_data[sample_idx, :, 1, 0] = torque

            # 模态2: 一阶差分
            first_diff = np.gradient(torque)
            first_diff_norm = (first_diff - np.min(first_diff)) / (np.max(first_diff) - np.min(first_diff) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 1] = angles
            multi_modal_data[sample_idx, :, 1, 1] = first_diff_norm

            # 模态3: 二阶差分
            second_diff = np.gradient(first_diff)
            second_diff_norm = (second_diff - np.min(second_diff)) / (np.max(second_diff) - np.min(second_diff) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 2] = angles
            multi_modal_data[sample_idx, :, 1, 2] = second_diff_norm

            # 模态4: 局部方差
            local_var = np.zeros_like(torque)
            window = 3
            for i in range(window//2, n_points - window//2):
                local_data = torque[i-window//2:i+window//2+1]
                local_var[i] = np.var(local_data)
            local_var_norm = local_var / (np.max(local_var) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 3] = angles
            multi_modal_data[sample_idx, :, 1, 3] = local_var_norm

            # 频域模态 (5-8)
            fft_data = np.fft.fft(torque)
            
            # 模态5: 低频成分
            low_freq_cutoff = n_points // 4
            fft_low = fft_data.copy()
            fft_low[low_freq_cutoff:-low_freq_cutoff] = 0
            low_freq = np.real(np.fft.ifft(fft_low))
            low_freq_norm = (low_freq - np.min(low_freq)) / (np.max(low_freq) - np.min(low_freq) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 4] = angles
            multi_modal_data[sample_idx, :, 1, 4] = low_freq_norm

            # 模态6: 高频成分
            fft_high = np.zeros_like(fft_data)
            high_freq_cutoff = n_points // 2
            fft_high[low_freq_cutoff:high_freq_cutoff] = fft_data[low_freq_cutoff:high_freq_cutoff]
            fft_high[-high_freq_cutoff:-low_freq_cutoff] = fft_data[-high_freq_cutoff:-low_freq_cutoff]
            high_freq = np.real(np.fft.ifft(fft_high))
            high_freq_norm = high_freq / (np.max(np.abs(high_freq)) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 5] = angles
            multi_modal_data[sample_idx, :, 1, 5] = high_freq_norm

            # 模态7-8: 简化的小波和功率谱
            multi_modal_data[sample_idx, :, 0, 6] = angles
            multi_modal_data[sample_idx, :, 1, 6] = low_freq_norm  # 简化版小波
            
            power_spectrum = np.abs(fft_data)**2
            power_features = np.zeros_like(torque)
            freqs = np.fft.fftfreq(n_points)
            for i in range(len(freqs)//2):
                if freqs[i] > 0:
                    power_features += power_spectrum[i] * np.cos(2*np.pi*freqs[i]*np.arange(n_points))
            power_norm = (power_features - np.min(power_features)) / (np.max(power_features) - np.min(power_features) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 7] = angles
            multi_modal_data[sample_idx, :, 1, 7] = power_norm

            # 几何模态 (9-11)
            # 模态9: 曲率
            curvature = np.zeros_like(torque)
            for i in range(1, n_points-1):
                dx1, dy1 = 1, first_diff[i]
                dx2, dy2 = 1, first_diff[i+1] - first_diff[i] if i+1 < len(first_diff) else 0
                curvature[i] = abs(dx1*dy2 - dy1*dx2) / (dx1**2 + dy1**2)**(3/2)
            curvature_norm = curvature / (np.max(curvature) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 8] = angles
            multi_modal_data[sample_idx, :, 1, 8] = curvature_norm

            # 模态10: 分段线性
            n_segments = 5
            segment_features = np.zeros_like(torque)
            for seg in range(n_segments):
                start_idx = seg * (n_points // n_segments)
                end_idx = (seg + 1) * (n_points // n_segments)
                if end_idx > n_points:
                    end_idx = n_points
                if end_idx > start_idx + 1:
                    slope = (torque[end_idx-1] - torque[start_idx]) / (end_idx - start_idx - 1)
                    segment_features[start_idx:end_idx] = slope
            segment_norm = (segment_features - np.min(segment_features)) / (np.max(segment_features) - np.min(segment_features) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 9] = angles
            multi_modal_data[sample_idx, :, 1, 9] = segment_norm

            # 模态11: 包络线
            envelope = np.maximum.accumulate(torque)
            envelope_norm = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 10] = angles
            multi_modal_data[sample_idx, :, 1, 10] = envelope_norm

            # 动力学模态 (12-14)
            # 模态12: 振荡检测
            oscillation_features = np.zeros_like(torque)
            for freq in [2, 4, 6]:
                t = np.linspace(0, 1, n_points)
                sine_component = np.sin(2 * np.pi * freq * t)
                correlation = np.correlate(torque, sine_component, mode='same')
                if len(correlation) == n_points:
                    oscillation_features += correlation
            osc_norm = (oscillation_features - np.min(oscillation_features)) / (np.max(oscillation_features) - np.min(oscillation_features) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 11] = angles
            multi_modal_data[sample_idx, :, 1, 11] = osc_norm

            # 模态13: 梯度一致性
            gradient_consistency = np.zeros_like(torque)
            for i in range(2, n_points):
                if i >= 5:
                    recent_grads = first_diff[i-5:i]
                    if len(recent_grads) > 0:
                        consistency = np.sum(np.sign(recent_grads[0]) == np.sign(recent_grads)) / len(recent_grads)
                        gradient_consistency[i] = consistency
            grad_cons_norm = gradient_consistency / (np.max(gradient_consistency) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 12] = angles
            multi_modal_data[sample_idx, :, 1, 12] = grad_cons_norm

            # 模态14: EMD简化版
            emd_feature = np.zeros_like(torque)
            for window_size in [3, 7, 15]:
                if window_size < n_points:
                    smoothed = np.convolve(torque, np.ones(window_size)/window_size, mode='same')
                    emd_feature += smoothed / 3
            emd_norm = (emd_feature - np.min(emd_feature)) / (np.max(emd_feature) - np.min(emd_feature) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 13] = angles
            multi_modal_data[sample_idx, :, 1, 13] = emd_norm

            # 复杂度模态 (15-16)
            # 模态15: 样本熵简化版
            entropy_feature = np.zeros_like(torque)
            window_size = 10
            for i in range(window_size, n_points - window_size):
                local_data = torque[i-window_size:i+window_size]
                entropy_feature[i] = np.var(local_data)  # 简化版样本熵
            entropy_norm = entropy_feature / (np.max(entropy_feature) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 14] = angles
            multi_modal_data[sample_idx, :, 1, 14] = entropy_norm

            # 模态16: 自相关
            autocorr_feature = np.zeros_like(torque)
            max_lag = min(20, n_points // 4)
            for lag in range(1, max_lag):
                if lag < n_points:
                    autocorr = np.corrcoef(torque[:-lag], torque[lag:])[0, 1]
                    if not np.isnan(autocorr):
                        autocorr_feature[lag:] += abs(autocorr)
            autocorr_norm = autocorr_feature / (np.max(autocorr_feature) + 1e-8)
            multi_modal_data[sample_idx, :, 0, 15] = angles
            multi_modal_data[sample_idx, :, 1, 15] = autocorr_norm

        print("[SUCCESS] 计算16种模态特征完成！")
        return multi_modal_data
    
    def save_complete_results(self, data_pairs, labels, sequence_info, multi_modal_features, 
                            outlier_info=None, filename_prefix='complete_imc_data'):
        """保存完整的处理结果"""
        print("保存完整处理结果...")

        # 1. 保存为numpy格式
        np.save(os.path.join(self.output_dir, f'{filename_prefix}_data_pairs.npy'), data_pairs)
        np.save(os.path.join(self.output_dir, f'{filename_prefix}_labels.npy'), labels)
        np.save(os.path.join(self.output_dir, f'{filename_prefix}_modal_features.npy'), multi_modal_features)

        # 2. 保存序列信息
        with open(os.path.join(self.output_dir, f'{filename_prefix}_sequence_info.json'), 'w', encoding='utf-8') as f:
            json.dump(sequence_info, f, ensure_ascii=False, indent=2)

        # 3. 保存为pandas DataFrame格式
        self.save_as_dataframe(data_pairs, labels, sequence_info, multi_modal_features, filename_prefix)

        # 4. 保存处理摘要 - 修复numpy数据类型问题
        summary = {
            'total_samples': int(len(data_pairs)),  # 转换为Python int
            'data_shape': [int(dim) for dim in data_pairs.shape],  # 转换为Python list
            'modal_features_shape': [int(dim) for dim in multi_modal_features.shape],  # 转换为Python list
            'label_distribution': {str(k): int(v) for k, v in dict(zip(*np.unique(labels, return_counts=True))).items   ()},  # 转换为Python类型
            'angles_range': [float(self.angles.min()), float(self.angles.max())],  # 转换为Python float
            'processing_date': pd.Timestamp.now().isoformat(),
            'outlier_info': self._convert_numpy_types(outlier_info) if outlier_info else {}  # 递归转换
        }

        with open(os.path.join(self.output_dir, f'{filename_prefix}_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] 保存完成! 文件保存在: {self.output_dir}")
        print(f"  - 数据对数组: {filename_prefix}_data_pairs.npy")
        print(f"  - 标签数组: {filename_prefix}_labels.npy") 
        print(f"  - 16模态特征: {filename_prefix}_modal_features.npy")
        print(f"  - 序列信息: {filename_prefix}_sequence_info.json")
        print(f"  - DataFrame格式: {filename_prefix}_dataframe.pkl/csv")
        print(f"  - 处理摘要: {filename_prefix}_summary.json")

        return summary

    def _convert_numpy_types(self, obj):
        """递归地将numpy数据类型转换为Python原生类型"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def save_as_dataframe(self, data_pairs, labels, sequence_info, multi_modal_features, filename_prefix):
        """保存为DataFrame格式"""
        # 创建基础DataFrame
        all_rows = []
        
        for sample_idx in range(len(data_pairs)):
            seq_info = sequence_info[sample_idx]
            label = labels[sample_idx]
            
            for point_idx in range(81):
                angle = data_pairs[sample_idx, point_idx, 0]
                torque = data_pairs[sample_idx, point_idx, 1]
                
                row = {
                    'sample_idx': sample_idx,
                    'patient_id': seq_info['patient_id'],
                    'repetition': seq_info['repetition'],
                    'sequence_id': seq_info['sequence_id'],
                    'imc_type': label,
                    'point_idx': point_idx,
                    'angle': angle,
                    'normalized_torque': torque
                }
                
                # 添加16个模态特征
                for modal_idx in range(16):
                    modal_value = multi_modal_features[sample_idx, point_idx, 1, modal_idx]
                    row[f'modal_{modal_idx+1:02d}'] = modal_value
                
                all_rows.append(row)
        
        df = pd.DataFrame(all_rows)
        
        # 保存为多种格式
        df.to_pickle(os.path.join(self.output_dir, f'{filename_prefix}_dataframe.pkl'))
        df.to_csv(os.path.join(self.output_dir, f'{filename_prefix}_dataframe.csv'), index=False)
        
        return df
    
    def run_complete_pipeline(self, outlier_method='gradient2', save_prefix='complete_imc_data'):
        """运行完整的数据处理流水线"""
        print("="*60)
        print("    运行完整的IMC数据处理流水线")
        print("="*60)
        
        # 步骤1: 处理原始数据
        print("\n步骤1: 处理原始数据...")
        if not self.process_all_data():
            print("数据处理失败!")
            return None
        
        # 步骤2: 转换为标准格式并保留标签
        print("\n步骤2: 转换数据格式...")
        data_pairs, labels, sequence_info = self.convert_to_data_pairs_with_labels()
        if data_pairs is None:
            print("数据转换失败!")
            return None
        
        # 步骤3: 异常检测和删除
        print(f"\n步骤3: 异常检测 (方法: {outlier_method})...")
        cleaned_data_pairs, cleaned_labels, cleaned_sequence_info, outlier_indices, outlier_stats = \
            self.detect_and_remove_outliers(data_pairs, labels, sequence_info, method=outlier_method)
        
        # 步骤4: 计算16种模态特征
        print("\n步骤4: 计算16种模态特征...")
        multi_modal_features = self.compute_multi_modal_features(cleaned_data_pairs)
        
        # 步骤5: 保存结果
        print("\n步骤5: 保存处理结果...")
        outlier_info = {
            'method': outlier_method,
            'outlier_indices': outlier_indices,
            'outlier_stats': outlier_stats,
            'removed_count': len(outlier_indices),
            'removal_percentage': len(outlier_indices) / len(data_pairs) * 100 if len(data_pairs) > 0 else 0
        }
        
        summary = self.save_complete_results(
            cleaned_data_pairs, cleaned_labels, cleaned_sequence_info, 
            multi_modal_features, outlier_info, save_prefix
        )
        
        # 步骤6: 打印处理摘要
        self.print_processing_summary(summary, outlier_info)
        
        # 返回所有结果
        results = {
            'data_pairs': cleaned_data_pairs,
            'labels': cleaned_labels,
            'sequence_info': cleaned_sequence_info,
            'multi_modal_features': multi_modal_features,
            'outlier_info': outlier_info,
            'summary': summary
        }
        
        print(f"\n[SUCCESS] 完整流水线处理成功!")
        print(f"最终数据: {cleaned_data_pairs.shape[0]} 个样本, 16种模态特征")
        return results
    
    def print_processing_summary(self, summary, outlier_info):
        """打印处理摘要"""
        print("\n" + "="*50)
        print("           数据处理摘要")
        print("="*50)
        
        print(f"总样本数: {summary['total_samples']}")
        print(f"数据形状: {summary['data_shape']}")
        print(f"模态特征形状: {summary['modal_features_shape']}")
        
        print("\n标签分布:")
        for label, count in summary['label_distribution'].items():
            percentage = count / summary['total_samples'] * 100
            print(f"  {label}: {count} 个样本 ({percentage:.1f}%)")
        
        print(f"\n异常数据处理:")
        print(f"  检测方法: {outlier_info['method']}")
        print(f"  删除样本数: {outlier_info['removed_count']}")
        print(f"  删除比例: {outlier_info['removal_percentage']:.1f}%")
        
        print(f"\n角度范围: {summary['angles_range'][0]}° - {summary['angles_range'][1]}°")
        print(f"处理时间: {summary['processing_date']}")
        print("="*50)
    
    def load_complete_results(self, filename_prefix='complete_imc_data'):
        """加载完整的处理结果"""
        print(f"加载处理结果: {filename_prefix}")
        
        try:
            # 加载主要数据
            data_pairs = np.load(os.path.join(self.output_dir, f'{filename_prefix}_data_pairs.npy'))
            labels = np.load(os.path.join(self.output_dir, f'{filename_prefix}_labels.npy'))
            multi_modal_features = np.load(os.path.join(self.output_dir, f'{filename_prefix}_modal_features.npy'))
            
            # 加载序列信息
            with open(os.path.join(self.output_dir, f'{filename_prefix}_sequence_info.json'), 'r', encoding='utf-8') as f:
                sequence_info = json.load(f)
            
            # 加载摘要
            with open(os.path.join(self.output_dir, f'{filename_prefix}_summary.json'), 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print(f"[SUCCESS] 加载成功!")
            print(f"  数据形状: {data_pairs.shape}")
            print(f"  标签数量: {len(labels)}")
            print(f"  模态特征形状: {multi_modal_features.shape}")
            
            return {
                'data_pairs': data_pairs,
                'labels': labels,
                'sequence_info': sequence_info,
                'multi_modal_features': multi_modal_features,
                'summary': summary
            }
            
        except Exception as e:
            print(f"加载失败: {e}")
            return None
    
    def visualize_modal_features(self, multi_modal_data, labels, sample_idx=0, save_path=None):
        """可视化16种模态特征"""
        modal_names = [
            'Original', 'First Diff', 'Second Diff', 'Local Variance',
            'Low Freq', 'High Freq', 'Wavelet', 'Power Spectrum',
            'Curvature', 'Segment Linear', 'Envelope', 'Oscillation',
            'Gradient Consistency', 'EMD', 'Sample Entropy', 'Autocorrelation'
        ]

        fig, axes = plt.subplots(4, 4, figsize=(24, 18))  # 增大图形尺寸
        axes = axes.flatten()

        sample_label = labels[sample_idx]

        for i in range(16):
            angles = multi_modal_data[sample_idx, :, 0, i]
            features = multi_modal_data[sample_idx, :, 1, i]

            axes[i].plot(angles, features, linewidth=2, color=plt.cm.tab10(i % 10))
            axes[i].set_title(f'Modal {i+1}: {modal_names[i]}', fontsize=10)  # 减小标题字体
            axes[i].set_xlabel('Angle (degrees)', fontsize=8)  # 减小标签字体
            axes[i].set_ylabel('Normalized Value', fontsize=8)  # 减小标签字体
            axes[i].set_xlim(90, 0)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='both', which='major', labelsize=8)  # 减小刻度字体

        fig.suptitle(f'Sample {sample_idx} - Label: {sample_label} - 16 Modal Features',
                     fontsize=16, fontweight='bold', y=0.98)

        # 手动调整子图间距，避免重叠
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=0.3, hspace=0.4)

        # 保存图片到指定路径
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] 图片已保存: {save_path}")

        plt.show()
    
    def visualize_samples_by_label(self, data_pairs, labels, n_samples_per_label=3, save_path=None):
        """按标签可视化样本"""
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        
        fig, axes = plt.subplots(n_labels, n_samples_per_label, figsize=(15, 4*n_labels))
        if n_labels == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for label_idx, label in enumerate(unique_labels):
            # 找到该标签的所有样本
            label_indices = np.where(labels == label)[0]
            selected_indices = label_indices[:n_samples_per_label] if len(label_indices) >= n_samples_per_label else label_indices
            
            for sample_idx, data_idx in enumerate(selected_indices):
                if sample_idx < n_samples_per_label:
                    angles = data_pairs[data_idx, :, 0]
                    torque = data_pairs[data_idx, :, 1]
                    
                    color = colors[label_idx % len(colors)]
                    axes[label_idx, sample_idx].plot(angles, torque, color=color, linewidth=2)
                    axes[label_idx, sample_idx].set_xlim(90, 0)
                    axes[label_idx, sample_idx].set_title(f'Label {label} - Sample {sample_idx+1}')
                    axes[label_idx, sample_idx].set_xlabel('Angle (degrees)')
                    axes[label_idx, sample_idx].set_ylabel('Normalized Torque')
                    axes[label_idx, sample_idx].grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for sample_idx in range(len(selected_indices), n_samples_per_label):
                axes[label_idx, sample_idx].set_visible(False)
        
        # 根据标签数量调整布局
        if n_labels > 1:
            plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 多标签时增加间距
        else:
            plt.subplots_adjust(hspace=0.2, wspace=0.3)

        # 保存图片到指定路径
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] 图片已保存: {save_path}")

        plt.show()
    
    def get_label_statistics(self, labels):
        """获取标签统计信息"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        stats = {}
        for label, count in zip(unique_labels, counts):
            stats[label] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
        
        return stats


def quick_process_complete_pipeline(data_path, output_dir=None,
                                  outlier_method='gradient2', save_prefix='complete_imc_data'):
    """快速运行完整处理流水线的便捷函数"""
    if output_dir is None:
        # 默认使用相对于脚本位置的data/processed目录
        output_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
    processor = IntegratedIMCProcessor(data_path, output_dir)
    results = processor.run_complete_pipeline(outlier_method, save_prefix)
    return processor, results


def load_and_analyze_results(output_dir=None,
                           filename_prefix='complete_imc_data'):
    """加载并分析处理结果的便捷函数"""
    if output_dir is None:
        # 默认使用相对于脚本位置的data/processed目录
        output_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
    processor = IntegratedIMCProcessor('', output_dir)
    results = processor.load_complete_results(filename_prefix)
    
    if results:
        # 打印分析结果
        print("\n=== 数据分析 ===")
        stats = processor.get_label_statistics(results['labels'])
        print("标签分布:")
        for label, info in stats.items():
            print(f"  {label}: {info['count']} 样本 ({info['percentage']:.1f}%)")
        
        print(f"\n数据维度:")
        print(f"  原始数据: {results['data_pairs'].shape}")
        print(f"  模态特征: {results['multi_modal_features'].shape}")
        
        # 可视化样本
        label_plot_path = os.path.join(output_dir, 'samples_by_label_analysis.png')
        processor.visualize_samples_by_label(results['data_pairs'], results['labels'], save_path=label_plot_path)
        
        return processor, results
    
    return None, None


def analyze_modal_importance(modal_features, labels, output_dir):
    """
    分析每个模态的重要性 (k=1, 分别测试16个模态)
    """
    print(f"进入analyze_modal_importance函数，参数类型: modal_features={type(modal_features)}, labels={type(labels)}, output_dir={output_dir}")
    print(f"modal_features形状: {modal_features.shape}, labels长度: {len(labels)}")

    # 导入datetime到主函数作用域
    from datetime import datetime

    print("开始定义模型类...")

    # 临时直接返回测试
    print("直接返回测试结果")
    return []

    class SingleModalNet(nn.Module):
        def __init__(self, sequence_length=81):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),  # 减少通道数
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8)
            )

        def forward(self, x):
            return self.feature_extractor(x).flatten(1)

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
    def __init__(self, num_classes, num_modals, sequence_length=81):
        super().__init__()
        self.num_modals = num_modals
        self.modal_extractors = nn.ModuleList([
            SingleModalNet(sequence_length) for _ in range(num_modals)
        ])
        self.feature_dim = 128 * 8  # 修改为128

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

class MultiModalDataset(Dataset):
        def __init__(self, modal_features, labels):
            self.modal_features = torch.FloatTensor(modal_features)
            self.labels = torch.LongTensor(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                'modal_data': self.modal_features[idx].unsqueeze(0),  # (81,) -> (1, 81)
                'label': self.labels[idx]
            }

def evaluate_single_modal(modal_idx):
    """评估单个模态的性能 - 计算完整指标"""
    try:
        # 强制使用CPU避免CUDA内存问题
        device = torch.device('cpu')

        # 使用第modal_idx个模态 (k=1)
        single_modal_features = modal_features[:, :, :, modal_idx:modal_idx+1]
        modal_data = single_modal_features[:, :, 1, :]

        # 确保标签是数值类型
        if isinstance(labels[0], str):
            label_encoder = LabelEncoder()
            numeric_labels = label_encoder.fit_transform(labels)
        else:
            numeric_labels = labels

        # 数据分割
        indices = list(range(len(modal_data)))
        train_idx, val_idx = train_test_split(indices, test_size=0.3,
                                             stratify=numeric_labels, random_state=42)

        train_modal_data = modal_data[train_idx]
        train_labels = numeric_labels[train_idx]
        val_modal_data = modal_data[val_idx]
        val_labels = numeric_labels[val_idx]

        # 创建数据集 - 减少batch_size节省内存
        train_dataset = MultiModalDataset(train_modal_data, train_labels)
        val_dataset = MultiModalDataset(val_modal_data, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 从32降到16
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=False)

        # 创建和训练模型
        model = ClassifierNet(num_classes=4, num_modals=1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 训练模型 (20个epoch，提高性能)
        model.train()
        for epoch in range(20):  # 增加到20个epoch
            for batch in train_loader:
                modal_data_batch = batch['modal_data'].to(device)
                labels_batch = batch['label'].to(device)

                optimizer.zero_grad()
                outputs, _ = model(modal_data_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

            # 每个epoch后打印loss
            if (epoch + 1) % 5 == 0:
                print(f"模态{modal_idx+1} Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

        # 评估模式
        model.eval()

        # 收集预测结果和概率
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                modal_data_batch = batch['modal_data'].to(device)
                labels_batch = batch['label'].to(device)

                outputs, _ = model(modal_data_batch)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算所有要求的指标
        accuracy = accuracy_score(all_labels, all_preds)
        balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        # 计算ROC-AUC (多分类)
        try:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"ROC-AUC计算失败: {e}")
            roc_auc = 0.5

        # 计算PR-AUC (多分类)
        try:
            pr_auc = 0
            for i in range(4):  # 4个类别
                y_binary = (all_labels == i).astype(int)
                if np.sum(y_binary) > 0 and np.sum(y_binary == 0) > 0:  # 确保正负样本都存在
                    precision, recall, _ = precision_recall_curve(y_binary, all_probs[:, i])
                    pr_auc += auc(recall, precision)
            pr_auc /= 4
        except Exception as e:
            print(f"PR-AUC计算失败: {e}")
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

    # 循环测试16个模态
    results = []
    print("\n开始逐个测试16个模态 (k=1)...")
    print("="*50)
    print(f"准备测试16个模态，modal_features形状: {modal_features.shape}")

    for modal_idx in range(16):
        print(f"测试模态 {modal_idx + 1}/16...")
        try:
            result = evaluate_single_modal(modal_idx)
            if result is not None:
                results.append(result)
                print(".4f")
            else:
                print(f"模态{modal_idx + 1}测试失败")
        except Exception as e:
            print(f"模态{modal_idx + 1}测试出错: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("没有成功的测试结果")
        return

    # 保存结果
    output_file = os.path.join(output_dir, 'modal_importance_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluation_time': datetime.now().isoformat(),
            'total_modals_tested': len(results),
            'modal_results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_file}")

    # 生成可视化图表
    plot_modal_analysis(results, output_dir)

    return results

def plot_modal_analysis(results, output_dir):
    """绘制模态分析图表 - 显示所有指标"""
    modal_indices = [r['modal_idx'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    balanced_accuracies = [r['balanced_accuracy'] for r in results]
    mccs = [r['mcc'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    roc_aucs = [r['roc_auc'] for r in results]
    pr_aucs = [r['pr_auc'] for r in results]

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Modal Importance Analysis (k=1) - Complete Metrics', fontsize=18, fontweight='bold')

    x = np.arange(len(modal_indices))
    width = 0.35

    # 1. 准确率和平衡准确率
    axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, balanced_accuracies, width, label='Balanced Accuracy', color='cyan', alpha=0.7)

    axes[0, 0].set_xlabel('Modal Index', fontsize=12)
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_title('Accuracy Metrics', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'M{i}' for i in modal_indices])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.0])

    # 2. MCC和F1-Score
    axes[0, 1].bar(x - width/2, mccs, width, label='MCC', color='red', alpha=0.7)
    axes[0, 1].bar(x + width/2, f1_scores, width, label='F1-Score', color='orange', alpha=0.7)

    axes[0, 1].set_xlabel('Modal Index', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('Correlation & F1 Metrics', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'M{i}' for i in modal_indices])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([-0.2, 1.0])

    # 3. ROC-AUC和PR-AUC
    axes[1, 0].bar(x - width/2, roc_aucs, width, label='ROC-AUC', color='green', alpha=0.7)
    axes[1, 0].bar(x + width/2, pr_aucs, width, label='PR-AUC', color='purple', alpha=0.7)

    axes[1, 0].set_xlabel('Modal Index', fontsize=12)
    axes[1, 0].set_ylabel('AUC Score', fontsize=12)
    axes[1, 0].set_title('AUC Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'M{i}' for i in modal_indices])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1.0])

    # 4. 性能雷达图 (所有指标的综合视图)
    # 归一化所有指标到0-1范围
    metrics_data = np.array([accuracies, balanced_accuracies, mccs, f1_scores, roc_aucs, pr_aucs])
    metrics_data = (metrics_data - metrics_data.min(axis=1, keepdims=True)) / (metrics_data.max(axis=1, keepdims=True) - metrics_data.min(axis=1, keepdims=True) + 1e-8)

    metrics_names = ['ACC', 'BA', 'MCC', 'F1', 'ROC', 'PR']
    colors = ['blue', 'cyan', 'red', 'orange', 'green', 'purple']

    for i, (metric_name, metric_values, color) in enumerate(zip(metrics_names, metrics_data, colors)):
        axes[1, 1].plot(modal_indices, metric_values, marker='o', linewidth=2, label=metric_name, color=color, alpha=0.8)

    axes[1, 1].set_xlabel('Modal Index', fontsize=12)
    axes[1, 1].set_ylabel('Normalized Score', fontsize=12)
    axes[1, 1].set_title('Normalized Performance Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.0])

    # 5. 统计汇总
    avg_metrics = [
        np.mean(accuracies), np.mean(balanced_accuracies), np.mean(mccs),
        np.mean(f1_scores), np.mean(roc_aucs), np.mean(pr_aucs)
    ]
    max_metrics = [
        np.max(accuracies), np.max(balanced_accuracies), np.max(mccs),
        np.max(f1_scores), np.max(roc_aucs), np.max(pr_aucs)
    ]

    metrics_labels = ['Avg ACC', 'Avg BA', 'Avg MCC', 'Avg F1', 'Avg ROC', 'Avg PR']
    x_stats = np.arange(len(metrics_labels))

    axes[2, 0].bar(x_stats, avg_metrics, color='blue', alpha=0.7, label='Average')
    axes[2, 0].bar(x_stats, max_metrics, color='red', alpha=0.5, label='Maximum', width=0.4)

    axes[2, 0].set_xlabel('Metrics', fontsize=12)
    axes[2, 0].set_ylabel('Score', fontsize=12)
    axes[2, 0].set_title('Overall Statistics Summary', fontsize=14, fontweight='bold')
    axes[2, 0].set_xticks(x_stats)
    axes[2, 0].set_xticklabels(metrics_labels, rotation=45, ha='right')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    axes[2, 0].set_ylim([0, 1.0])

    # 6. Top 5模态综合排名 (基于所有指标的平均值)
    comprehensive_scores = np.mean(metrics_data, axis=0)  # 对所有归一化指标取平均
    sorted_indices = np.argsort(comprehensive_scores)[::-1]  # 降序排列
    top_modals = [modal_indices[i] for i in sorted_indices[:5]]
    top_scores = [comprehensive_scores[i] for i in sorted_indices[:5]]

    axes[2, 1].bar(range(len(top_modals)), top_scores, color='darkgreen', alpha=0.8)
    axes[2, 1].set_xlabel('Rank', fontsize=12)
    axes[2, 1].set_ylabel('Comprehensive Score', fontsize=12)
    axes[2, 1].set_title('Top 5 Modals (Comprehensive Ranking)', fontsize=14, fontweight='bold')
    axes[2, 1].set_xticks(range(len(top_modals)))
    axes[2, 1].set_xticklabels([f'M{m}' for m in top_modals])
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    axes[2, 1].set_ylim([0, 1.0])

    # 添加数值标签
    for i, score in enumerate(top_scores):
        axes[2, 1].text(i, score + 0.01, '.3f', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # 保存图表 - 降低DPI以节省内存
    svg_path = os.path.join(output_dir, 'modal_importance_analysis_complete.svg')
    png_path = os.path.join(output_dir, 'modal_importance_analysis_complete.png')

    # 先保存SVG (矢量图，不受DPI影响)
    plt.savefig(svg_path, bbox_inches='tight')
    print(f"完整的模态重要性分析图已保存到: {svg_path}")

    # 保存PNG (降低DPI以减少内存)
    try:
        plt.savefig(png_path, dpi=100, bbox_inches='tight')  # 从300降到100
        print(f"完整的模态重要性分析图已保存到: {png_path}")
    except MemoryError:
        print("内存不足，跳过PNG保存，只保存SVG格式")
        # 只保存低分辨率版本
        plt.savefig(png_path.replace('.png', '_low.png'), dpi=50, bbox_inches='tight')
        print(f"低分辨率PNG已保存到: {png_path.replace('.png', '_low.png')}")

    plt.close('all')  # 关闭所有图表释放内存

    plt.show()

# 主函数 - 修改为循环测试各个模态
if __name__ == "__main__":
    print("=== 模态可解释性分析 ===")
    print("测试每个模态单独的性能 (k=1)")
    print(f"开始时间: {datetime.now().isoformat()}")

    # 设置输出目录
    output_dir = r'D:\document\code\IMC\result\interpretability'
    os.makedirs(output_dir, exist_ok=True)

    # 加载处理好的数据
    processor, data = load_and_analyze_results(
        output_dir=os.path.join(os.path.dirname(__file__), '../data/processed'),
        filename_prefix='processed_imc_data'
    )

    if data:
        print("数据加载成功，开始模态重要性分析...")

        # 分析模态重要性
        print(f"开始分析模态重要性，模态特征形状: {data['multi_modal_features'].shape}")
        results = analyze_modal_importance(
            data['multi_modal_features'],
            data['labels'],
            output_dir
        )
        print(f"模态重要性分析完成，返回结果数量: {len(results) if results else 0}")

        if results:
            print(f"\n成功分析了 {len(results)} 个模态")
            print(f"结果已保存到: {output_dir}")

            # 输出最佳模态
            best_modal = max(results, key=lambda x: x['accuracy'])
            print(".4f")

    else:
        print("数据加载失败，无法进行模态重要性分析")

    print("\n模态可解释性分析完成!")