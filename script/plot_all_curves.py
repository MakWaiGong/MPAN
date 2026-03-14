import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CurvePlotter:
    """
    IMC曲线可视化工具：为每条曲线生成单独的图表
    """

    def __init__(self, data_path, labels_path, output_dir):
        """
        初始化曲线绘图器

        Args:
            data_path: 曲线数据文件路径 (.npy)
            labels_path: 标签数据文件路径 (.npy)
            output_dir: 输出目录路径
        """
        self.data_path = data_path
        self.labels_path = labels_path
        self.output_dir = Path(output_dir)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录按标签分类
        self.label_dirs = {}
        for label in ['a', 'b', 'c', 'd']:
            label_dir = self.output_dir / label
            label_dir.mkdir(exist_ok=True)
            self.label_dirs[label] = label_dir

    def load_data(self):
        """加载曲线数据和标签"""
        print("正在加载数据...")

        # 加载数据
        self.data = np.load(self.data_path)
        self.labels = np.load(self.labels_path)

        print(f"数据形状: {self.data.shape}")
        print(f"标签形状: {self.labels.shape}")

        # 统计信息
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print("标签分布:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count} 条曲线")

        return len(self.data)

    def plot_single_curve(self, curve_idx, angles, torques, label):
        """
        绘制单条曲线

        Args:
            curve_idx: 曲线索引
            angles: 角度数据
            torques: 扭矩数据
            label: 曲线标签
        """
        plt.figure(figsize=(10, 6))

        # 绘制曲线
        plt.plot(angles, torques, 'b-', linewidth=2, alpha=0.8)

        # 设置图表属性
        plt.title(f'IMC Curve - Index:{curve_idx:03d} Label:{label}', fontsize=14, pad=20)
        plt.xlabel('Angle (degrees)', fontsize=12)
        plt.ylabel('Normalized Torque', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 设置坐标轴范围
        plt.xlim(90, 10)  # 从大角度到小角度
        plt.ylim(0, 1.1)  # 归一化扭矩范围

        # 添加网格和样式
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        return plt.gcf()

    def save_curve_plot(self, curve_idx, label):
        """
        保存单条曲线的图表

        Args:
            curve_idx: 曲线索引
            label: 曲线标签
        """
        # 获取数据
        curve_data = self.data[curve_idx]
        angles = curve_data[:, 0]    # 第一列是角度
        torques = curve_data[:, 1]   # 第二列是扭矩

        # 创建图表
        fig = self.plot_single_curve(curve_idx, angles, torques, label)

        # 保存路径
        output_dir = self.label_dirs[label]
        filename = f"curve_{curve_idx:03d}_label_{label}.png"
        output_path = output_dir / filename

        # 保存图表
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def plot_all_curves(self, show_progress=True):
        """
        绘制所有曲线

        Args:
            show_progress: 是否显示进度
        """
        total_curves = len(self.data)
        print(f"\n开始绘制 {total_curves} 条曲线...")

        saved_files = []

        for idx in range(total_curves):
            label = self.labels[idx]

            # 保存图表
            output_path = self.save_curve_plot(idx, label)
            saved_files.append(output_path)

            if show_progress and (idx + 1) % 50 == 0:
                print(f"已处理 {idx + 1}/{total_curves} 条曲线")

        print(f"\n完成！共保存 {len(saved_files)} 个图表文件")

        # 统计信息
        self.print_summary()

        return saved_files

    def print_summary(self):
        """打印汇总信息"""
        print("\n" + "="*50)
        print("曲线图表生成汇总")
        print("="*50)

        total_files = 0
        for label, label_dir in self.label_dirs.items():
            # 统计该标签目录下的文件数
            files = list(label_dir.glob("*.png"))
            count = len(files)
            total_files += count
            print(f"{label}类型: {count} 个图表")

        print(f"\n总计: {total_files} 个图表文件")
        print(f"输出目录: {self.output_dir}")

def main():
    """主函数"""
    # 文件路径
    data_path = r"D:\document\code\IMC\data\processed\processed_imc_data_data_pairs.npy"
    labels_path = r"D:\document\code\IMC\data\processed\processed_imc_data_labels.npy"
    output_dir = r"D:\document\code\IMC\data\curves"

    # 检查输入文件是否存在
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在: {data_path}")
        return

    if not os.path.exists(labels_path):
        print(f"错误：标签文件不存在: {labels_path}")
        return

    # 创建绘图器并执行
    try:
        plotter = CurvePlotter(data_path, labels_path, output_dir)

        # 加载数据
        total_curves = plotter.load_data()

        # 绘制所有曲线
        saved_files = plotter.plot_all_curves()

        print(f"\n脚本执行完成！")
        print(f"共处理了 {total_curves} 条曲线")
        print(f"生成了 {len(saved_files)} 个图表文件")

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
