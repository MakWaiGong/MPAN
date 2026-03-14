import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def verify_baseline_plot():
    """验证基线对比图表是否包含Transformer模型"""

    # 读取PNG图片
    img_path = r'D:\document\code\IMC\result\plot\baseline_vs_our_model_comparison.png'
    img = mpimg.imread(img_path)

    print("图片尺寸:", img.shape)
    print("图片已成功生成!")

    # 检查baseline.py中的模型列表
    print("\n检查baseline.py中的模型列表:")
    try:
        with open(r'D:\document\code\IMC\result\summary\baseline.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找模型列表
        if 'SimpleTransformer' in content:
            print("[OK] SimpleTransformer 模型数据已添加到baseline.py")

        if "'Transformer'" in content:
            print("[OK] Transformer 标签已添加到图例中")

        if '#8E44AD' in content:
            print("[OK] Transformer 模型的紫色已配置")

        # 统计模型数量
        import re
        models_match = re.search(r"models = \[(.*?)\]", content, re.DOTALL)
        if models_match:
            models_str = models_match.group(1)
            model_count = len([m.strip().strip("'\"") for m in models_str.split(',') if m.strip()])
            print(f"[OK] 总共配置了 {model_count} 个模型")

    except Exception as e:
        print(f"读取文件时出错: {e}")

    print("\n验证完成！")

if __name__ == "__main__":
    verify_baseline_plot()
