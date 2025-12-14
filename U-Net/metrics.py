"""
Metrics Module - Reusable untuk semua model segmentasi
Bisa digunakan untuk U-Net, DeepLab, ResNet, atau model lainnya
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

class SegmentationMetrics:
    """
    Class untuk menghitung metrik segmentasi
    Mendukung: Dice Coefficient, IoU, Precision, Recall, Specificity, F2-Score
    """
    
    def __init__(self, smooth=1e-6, threshold=0.5):
        """
        Args:
            smooth: Smoothing factor untuk menghindari division by zero
            threshold: Threshold untuk binary classification (default: 0.5)
        """
        self.smooth = smooth
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset semua accumulated metrics"""
        self.dice_scores = []
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.specificity_scores = []
        self.f2_scores = []
        self.accuracy_scores = []
    
    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Dice Coefficient (F1-Score untuk segmentasi)
        
        Formula: DSC = 2 * |X ∩ Y| / (|X| + |Y|)
        
        Args:
            pred: Predicted mask (tensor)
            target: Ground truth mask (tensor)
        
        Returns:
            dice score (float)
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return dice.item()
    
    def iou_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Intersection over Union (Jaccard Index)
        
        Formula: IoU = |X ∩ Y| / |X ∪ Y|
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
        
        Returns:
            iou score (float)
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return iou.item()
    
    def precision_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Precision (Positive Predictive Value)
        
        Formula: Precision = TP / (TP + FP)
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
        
        Returns:
            precision score (float)
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        true_positive = (pred * target).sum()
        predicted_positive = pred.sum()
        precision = (true_positive + self.smooth) / (predicted_positive + self.smooth)
        
        return precision.item()
    
    def recall_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Recall (Sensitivity / True Positive Rate)
        
        Formula: Recall = TP / (TP + FN)
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
        
        Returns:
            recall score (float)
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        true_positive = (pred * target).sum()
        actual_positive = target.sum()
        recall = (true_positive + self.smooth) / (actual_positive + self.smooth)
        
        return recall.item()
    
    def specificity_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Specificity (True Negative Rate)
        
        Formula: Specificity = TN / (TN + FP)
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
        
        Returns:
            specificity score (float)
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        true_negative = ((1 - pred) * (1 - target)).sum()
        actual_negative = (1 - target).sum()
        specificity = (true_negative + self.smooth) / (actual_negative + self.smooth)
        
        return specificity.item()
    
    def f2_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        F2-Score (memberikan bobot lebih pada Recall)
        
        Formula: F2 = 5 * (Precision * Recall) / (4 * Precision + Recall)
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
        
        Returns:
            f2 score (float)
        """
        precision = self.precision_score(pred, target)
        recall = self.recall_score(pred, target)
        
        f2 = (5 * precision * recall) / (4 * precision + recall + self.smooth)
        return f2
    
    def accuracy_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Pixel Accuracy
        
        Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
        
        Returns:
            accuracy score (float)
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        correct = (pred == target).float().sum()
        total = target.numel()
        accuracy = correct / total
        
        return accuracy.item()
    
    def calculate_all_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Hitung semua metrik sekaligus
        
        Args:
            pred: Predicted mask (raw output, akan di-threshold)
            target: Ground truth mask
        
        Returns:
            Dictionary berisi semua metrik
        """
        # Apply threshold untuk binary prediction
        pred_binary = (pred > self.threshold).float()
        
        metrics = {
            'dice': self.dice_coefficient(pred_binary, target),
            'iou': self.iou_score(pred_binary, target),
            'precision': self.precision_score(pred_binary, target),
            'recall': self.recall_score(pred_binary, target),
            'specificity': self.specificity_score(pred_binary, target),
            'f2_score': self.f2_score(pred_binary, target),
            'accuracy': self.accuracy_score(pred_binary, target)
        }
        
        # Accumulate untuk averaging nanti
        self.dice_scores.append(metrics['dice'])
        self.iou_scores.append(metrics['iou'])
        self.precision_scores.append(metrics['precision'])
        self.recall_scores.append(metrics['recall'])
        self.specificity_scores.append(metrics['specificity'])
        self.f2_scores.append(metrics['f2_score'])
        self.accuracy_scores.append(metrics['accuracy'])
        
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Dapatkan rata-rata dan standard deviation dari semua accumulated metrics
        
        Returns:
            Dictionary berisi mean dan std untuk setiap metrik
        """
        return {
            'dice_mean': np.mean(self.dice_scores),
            'dice_std': np.std(self.dice_scores),
            'iou_mean': np.mean(self.iou_scores),
            'iou_std': np.std(self.iou_scores),
            'precision_mean': np.mean(self.precision_scores),
            'precision_std': np.std(self.precision_scores),
            'recall_mean': np.mean(self.recall_scores),
            'recall_std': np.std(self.recall_scores),
            'specificity_mean': np.mean(self.specificity_scores),
            'specificity_std': np.std(self.specificity_scores),
            'f2_score_mean': np.mean(self.f2_scores),
            'f2_score_std': np.std(self.f2_scores),
            'accuracy_mean': np.mean(self.accuracy_scores),
            'accuracy_std': np.std(self.accuracy_scores)
        }
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Dapatkan summary metrics dalam format DataFrame
        
        Returns:
            Pandas DataFrame berisi summary semua metrik
        """
        avg_metrics = self.get_average_metrics()
        
        data = {
            'Metric': ['Dice Coefficient', 'IoU', 'Precision', 'Recall', 
                       'Specificity', 'F2-Score', 'Accuracy'],
            'Mean': [
                avg_metrics['dice_mean'],
                avg_metrics['iou_mean'],
                avg_metrics['precision_mean'],
                avg_metrics['recall_mean'],
                avg_metrics['specificity_mean'],
                avg_metrics['f2_score_mean'],
                avg_metrics['accuracy_mean']
            ],
            'Std': [
                avg_metrics['dice_std'],
                avg_metrics['iou_std'],
                avg_metrics['precision_std'],
                avg_metrics['recall_std'],
                avg_metrics['specificity_std'],
                avg_metrics['f2_score_std'],
                avg_metrics['accuracy_std']
            ]
        }
        
        return pd.DataFrame(data)
    
    def print_metrics(self, model_name: str = "Model"):
        """
        Print metrics dengan format yang rapi
        
        Args:
            model_name: Nama model untuk display
        """
        print(f"\n{'='*80}")
        print(f"HASIL EVALUASI - {model_name}")
        print(f"{'='*80}")
        
        avg_metrics = self.get_average_metrics()
        
        print(f"\nDice Coefficient:  {avg_metrics['dice_mean']:.4f} ± {avg_metrics['dice_std']:.4f}")
        print(f"IoU:               {avg_metrics['iou_mean']:.4f} ± {avg_metrics['iou_std']:.4f}")
        print(f"Precision:         {avg_metrics['precision_mean']:.4f} ± {avg_metrics['precision_std']:.4f}")
        print(f"Recall:            {avg_metrics['recall_mean']:.4f} ± {avg_metrics['recall_std']:.4f}")
        print(f"Specificity:       {avg_metrics['specificity_mean']:.4f} ± {avg_metrics['specificity_std']:.4f}")
        print(f"F2-Score:          {avg_metrics['f2_score_mean']:.4f} ± {avg_metrics['f2_score_std']:.4f}")
        print(f"Accuracy:          {avg_metrics['accuracy_mean']:.4f} ± {avg_metrics['accuracy_std']:.4f}")
        print(f"{'='*80}\n")


def compare_two_models(metrics1: SegmentationMetrics, metrics2: SegmentationMetrics,
                       model1_name: str = "Model 1", model2_name: str = "Model 2") -> pd.DataFrame:
    """
    Bandingkan dua model berdasarkan metrics
    
    Args:
        metrics1: Metrics dari model pertama
        metrics2: Metrics dari model kedua
        model1_name: Nama model pertama
        model2_name: Nama model kedua
    
    Returns:
        DataFrame berisi perbandingan lengkap
    """
    avg_metrics1 = metrics1.get_average_metrics()
    avg_metrics2 = metrics2.get_average_metrics()
    
    print(f"\n{'='*80}")
    print(f"PERBANDINGAN: {model1_name} vs {model2_name}")
    print(f"{'='*80}")
    
    metric_names = ['Dice Coefficient', 'IoU', 'Precision', 'Recall', 
                    'Specificity', 'F2-Score', 'Accuracy']
    metric_keys = ['dice', 'iou', 'precision', 'recall', 'specificity', 'f2_score', 'accuracy']
    
    comparison_data = []
    
    for name, key in zip(metric_names, metric_keys):
        model1_mean = avg_metrics1[f'{key}_mean']
        model1_std = avg_metrics1[f'{key}_std']
        model2_mean = avg_metrics2[f'{key}_mean']
        model2_std = avg_metrics2[f'{key}_std']
        
        diff = model2_mean - model1_mean
        improvement = (diff / model1_mean) * 100 if model1_mean > 0 else 0
        
        print(f"\n{name}:")
        print(f"  {model1_name:12s}: {model1_mean:.4f} ± {model1_std:.4f}")
        print(f"  {model2_name:12s}: {model2_mean:.4f} ± {model2_std:.4f}")
        print(f"  Selisih:      {diff:+.4f} ({improvement:+.2f}%)")
        
        comparison_data.append({
            'Metric': name,
            f'{model1_name}_Mean': model1_mean,
            f'{model1_name}_Std': model1_std,
            f'{model2_name}_Mean': model2_mean,
            f'{model2_name}_Std': model2_std,
            'Difference': diff,
            'Improvement_%': improvement
        })
    
    print(f"\n{'='*80}\n")
    
    return pd.DataFrame(comparison_data)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Test dengan dummy data
    print("Testing Metrics Module...")
    
    # Simulasi predictions dan targets
    pred = torch.rand(1, 1, 256, 256)  # Model output
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()  # Ground truth
    
    # Initialize metrics calculator
    metrics = SegmentationMetrics(threshold=0.5)
    
    # Calculate metrics
    result = metrics.calculate_all_metrics(pred, target)
    
    print("\nSingle sample metrics:")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}")
    
    # Get summary
    summary_df = metrics.get_metrics_summary()
    print("\n" + str(summary_df))
    
    print("\n✓ Metrics module berfungsi dengan baik!")
    print("✓ Siap digunakan untuk evaluasi model apapun!")