import json
import os
from pathlib import Path

def summarize_baseline_results():
    """汇总基线模型性能结果"""

    # 读取所有模型的性能指标
    metrics_dir = Path(r'D:\document\code\IMC\result\baseline\baseline_experiment_20260118_091420\metrics')
    results = {}

    for metric_file in metrics_dir.glob('*_metrics.json'):
        model_name = metric_file.stem.replace('_metrics', '')
        with open(metric_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results[model_name] = {
                'accuracy': data['accuracy'],
                'balanced_accuracy': data['balanced_accuracy'],
                'f1_score': data['f1_score']
            }

    # 打印结果
    print('=' * 60)
    print('BASELINE MODELS PERFORMANCE COMPARISON')
    print('=' * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Bal.Acc':<10} {'F1-Score':<10}")
    print('-' * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    for model_name, metrics in sorted_results:
        acc = metrics['accuracy']
        bal_acc = metrics['balanced_accuracy']
        f1 = metrics['f1_score']
        print(f"{model_name:<20} {acc:.4f}   {bal_acc:.4f}   {f1:.4f}")

    print()
    print('Best performing model:', sorted_results[0][0])
    print()
    print('Transformer Model Details:')
    transformer_metrics = results.get('SimpleTransformer', {})
    if transformer_metrics:
        print(f"  Accuracy: {transformer_metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {transformer_metrics['balanced_accuracy']:.4f}")
        print(f"  F1 Score: {transformer_metrics['f1_score']:.4f}")
        print("  Note: Simple 3-layer Transformer with attention mechanism")

if __name__ == "__main__":
    summarize_baseline_results()
