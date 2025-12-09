"""
Script untuk membandingkan hasil U-Net Baseline vs U-Net Attention Gate
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import SegmentationMetrics, compare_two_models
from unet_baseline_complete import UNetBaseline, SegmentationDataset, get_transforms
from unet_attention_complete import UNetAttention

def evaluate_model(model, loader, device, threshold=0.5):
    """Evaluasi model dengan metrics lengkap"""
    model.eval()
    metrics = SegmentationMetrics(threshold=threshold)
    
    from tqdm import tqdm
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            targets = targets.unsqueeze(1).to(device)
            
            predictions = model(data)
            
            for i in range(predictions.shape[0]):
                metrics.calculate_all_metrics(predictions[i], targets[i])
    
    return metrics


def plot_comparison(comparison_df, save_path='results/comparison_plot.png'):
    """Plot bar chart perbandingan metrics"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = comparison_df['Metric'].tolist()
    baseline_means = comparison_df.iloc[:, 1].tolist()  # Column kedua
    attention_means = comparison_df.iloc[:, 3].tolist()  # Column keempat
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_means, width, label='U-Net Baseline', color='#3498db')
    bars2 = ax.bar(x + width/2, attention_means, width, label='U-Net Attention Gate', color='#e74c3c')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Perbandingan U-Net Baseline vs U-Net Attention Gate', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Tambahkan nilai di atas bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {save_path}")


def plot_improvement(comparison_df, save_path='results/improvement_plot.png'):
    """Plot improvement percentage"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = comparison_df['Metric'].tolist()
    improvements = comparison_df['Improvement_%'].tolist()
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
    bars = ax.barh(metrics, improvements, color=colors)
    
    ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Percentage Improvement: Attention Gate vs Baseline', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Tambahkan nilai
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(val, i, f' {val:+.2f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Improvement plot saved: {save_path}")


def save_detailed_report(comparison_df, baseline_metrics, attention_metrics, save_path='results/detailed_report.txt'):
    """Save detailed text report"""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LAPORAN PERBANDINGAN U-NET BASELINE VS U-NET ATTENTION GATE\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. RINGKASAN HASIL\n")
        f.write("-"*80 + "\n\n")
        
        baseline_avg = baseline_metrics.get_average_metrics()
        attention_avg = attention_metrics.get_average_metrics()
        
        f.write("U-Net Baseline:\n")
        for key in ['dice_mean', 'iou_mean', 'precision_mean', 'recall_mean']:
            metric_name = key.replace('_mean', '').upper()
            f.write(f"  {metric_name:12s}: {baseline_avg[key]:.4f} ± {baseline_avg[key.replace('mean', 'std')]:.4f}\n")
        
        f.write("\nU-Net Attention Gate:\n")
        for key in ['dice_mean', 'iou_mean', 'precision_mean', 'recall_mean']:
            metric_name = key.replace('_mean', '').upper()
            f.write(f"  {metric_name:12s}: {attention_avg[key]:.4f} ± {attention_avg[key.replace('mean', 'std')]:.4f}\n")
        
        f.write("\n\n2. PERBANDINGAN DETAIL\n")
        f.write("-"*80 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        
        f.write("\n\n3. KESIMPULAN\n")
        f.write("-"*80 + "\n\n")
        
        total_improvement = comparison_df['Improvement_%'].mean()
        if total_improvement > 0:
            f.write(f"U-Net dengan Attention Gate menunjukkan peningkatan rata-rata sebesar {total_improvement:.2f}%\n")
            f.write("dibandingkan dengan U-Net Baseline.\n\n")
            
            best_metric = comparison_df.loc[comparison_df['Improvement_%'].idxmax()]
            f.write(f"Peningkatan terbesar terjadi pada metrik: {best_metric['Metric']}\n")
            f.write(f"dengan improvement sebesar {best_metric['Improvement_%']:.2f}%\n")
        else:
            f.write("U-Net Baseline menunjukkan performa yang lebih baik pada dataset ini.\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Detailed report saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare U-Net Baseline vs Attention Gate')
    
    # Data paths
    parser.add_argument('--test_img_dir', type=str, default='data/processed/test/images')
    parser.add_argument('--test_mask_dir', type=str, default='data/processed/test/masks')
    
    # Model checkpoints
    parser.add_argument('--baseline_checkpoint', type=str, default='checkpoints/unet_baseline.pth.tar')
    parser.add_argument('--attention_checkpoint', type=str, default='checkpoints/unet_attention.pth.tar')
    
    # Parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print("PERBANDINGAN MODEL U-NET")
    print(f"{'='*80}")
    print(f"Device: {device}\n")
    
    # Prepare data
    _, test_transform = get_transforms(args.img_size)
    test_ds = SegmentationDataset(args.test_img_dir, args.test_mask_dir, test_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Test samples: {len(test_ds)}\n")
    
    # Evaluate Baseline
    print("="*80)
    print("1. EVALUASI U-NET BASELINE")
    print("="*80)
    baseline_model = UNetBaseline(in_channels=3, out_channels=1).to(device)
    baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location=device)
    baseline_model.load_state_dict(baseline_checkpoint['state_dict'])
    print("✓ Baseline model loaded")
    
    baseline_metrics = evaluate_model(baseline_model, test_loader, device, args.threshold)
    baseline_metrics.print_metrics("U-Net Baseline")
    
    # Evaluate Attention
    print("="*80)
    print("2. EVALUASI U-NET ATTENTION GATE")
    print("="*80)
    attention_model = UNetAttention(in_channels=3, out_channels=1).to(device)
    attention_checkpoint = torch.load(args.attention_checkpoint, map_location=device)
    attention_model.load_state_dict(attention_checkpoint['state_dict'])
    print("✓ Attention model loaded")
    
    attention_metrics = evaluate_model(attention_model, test_loader, device, args.threshold)
    attention_metrics.print_metrics("U-Net Attention Gate")
    
    # Compare models
    print("="*80)
    print("3. PERBANDINGAN KEDUA MODEL")
    print("="*80)
    comparison_df = compare_two_models(
        baseline_metrics, 
        attention_metrics,
        "U-Net Baseline",
        "U-Net Attention"
    )
    
    # Save results
    comparison_csv = os.path.join(args.output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"✓ Comparison saved: {comparison_csv}")
    
    # Save individual results
    baseline_df = baseline_metrics.get_metrics_summary()
    baseline_df.to_csv(os.path.join(args.output_dir, 'baseline_metrics.csv'), index=False)
    
    attention_df = attention_metrics.get_metrics_summary()
    attention_df.to_csv(os.path.join(args.output_dir, 'attention_metrics.csv'), index=False)
    
    # Generate plots
    plot_comparison(comparison_df, os.path.join(args.output_dir, 'comparison_plot.png'))
    plot_improvement(comparison_df, os.path.join(args.output_dir, 'improvement_plot.png'))
    
    # Generate detailed report
    save_detailed_report(
        comparison_df, 
        baseline_metrics, 
        attention_metrics,
        os.path.join(args.output_dir, 'detailed_report.txt')
    )
    
    print("\n" + "="*80)
    print("✓ PERBANDINGAN SELESAI!")
    print("="*80)
    print(f"\nSemua hasil disimpan di: {args.output_dir}/")
    print("Files yang dibuat:")
    print("  - model_comparison.csv       : Perbandingan lengkap")
    print("  - baseline_metrics.csv       : Metrik U-Net Baseline")
    print("  - attention_metrics.csv      : Metrik U-Net Attention")
    print("  - comparison_plot.png        : Bar chart perbandingan")
    print("  - improvement_plot.png       : Chart peningkatan")
    print("  - detailed_report.txt        : Laporan detail lengkap")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()