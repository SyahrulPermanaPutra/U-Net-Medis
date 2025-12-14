"""
Medical Image Visualization Module
Untuk visualisasi hasil prediksi segmentasi citra medis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from tqdm import tqdm

class MedicalImageVisualizer:
    """Class untuk visualisasi hasil segmentasi citra medis"""
    
    def __init__(self, save_dir='visualizations'):
        """
        Args:
            save_dir: Direktori untuk menyimpan hasil visualisasi
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def denormalize_image(self, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Denormalize image untuk visualisasi
        
        Args:
            image: Normalized image tensor [C, H, W]
            mean: Mean yang digunakan saat normalisasi
            std: Std yang digunakan saat normalisasi
        
        Returns:
            Denormalized image [H, W, C] dalam range [0, 255]
        """
        image = image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array(mean)
        std = np.array(std)
        image = std * image + mean
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image
    
    def overlay_mask_on_image(self, image, mask, alpha=0.5, color=(255, 0, 0)):
        """
        Overlay mask pada image dengan transparansi
        
        Args:
            image: Original image [H, W, C]
            mask: Binary mask [H, W]
            alpha: Transparansi overlay (0-1)
            color: Warna mask (R, G, B)
        
        Returns:
            Image dengan mask overlay
        """
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Buat colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Blend
        overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        
        return overlay
    
    def create_comparison_grid(self, original, ground_truth, prediction, 
                               title="", save_path=None):
        """
        Buat grid perbandingan: Original | Ground Truth | Prediction | Overlay
        
        Args:
            original: Original image [H, W, C]
            ground_truth: Ground truth mask [H, W]
            prediction: Predicted mask [H, W]
            title: Title untuk plot
            save_path: Path untuk menyimpan hasil
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Images
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(ground_truth, cmap='gray')
        axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(prediction, cmap='gray')
        axes[0, 2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Overlays
        gt_overlay = self.overlay_mask_on_image(original.copy(), ground_truth, 
                                                 alpha=0.4, color=(0, 255, 0))
        axes[1, 0].imshow(gt_overlay)
        axes[1, 0].set_title('GT Overlay (Green)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        pred_overlay = self.overlay_mask_on_image(original.copy(), prediction, 
                                                   alpha=0.4, color=(255, 0, 0))
        axes[1, 1].imshow(pred_overlay)
        axes[1, 1].set_title('Prediction Overlay (Red)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Combined overlay: GT=Green, Pred=Red, Overlap=Yellow
        combined = self.create_combined_overlay(original, ground_truth, prediction)
        axes[1, 2].imshow(combined)
        axes[1, 2].set_title('Combined (Green=GT, Red=Pred, Yellow=Both)', 
                            fontsize=10, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_combined_overlay(self, image, ground_truth, prediction):
        """
        Buat overlay kombinasi: GT (hijau), Prediction (merah), Overlap (kuning)
        
        Args:
            image: Original image [H, W, C]
            ground_truth: Ground truth mask [H, W]
            prediction: Predicted mask [H, W]
        
        Returns:
            Combined overlay image
        """
        gt_mask = (ground_truth > 0.5).astype(np.uint8)
        pred_mask = (prediction > 0.5).astype(np.uint8)
        
        # True Positive (overlap) = Yellow
        tp_mask = gt_mask & pred_mask
        # False Positive (pred only) = Red
        fp_mask = pred_mask & (~gt_mask)
        # False Negative (gt only) = Green
        fn_mask = gt_mask & (~pred_mask)
        
        overlay = image.copy()
        overlay[fn_mask > 0] = [0, 255, 0]      # Green - GT only
        overlay[fp_mask > 0] = [255, 0, 0]      # Red - Pred only
        overlay[tp_mask > 0] = [255, 255, 0]    # Yellow - Both
        
        # Blend dengan original
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        return result
    
    def save_individual_results(self, original, ground_truth, prediction, 
                                filename, model_name):
        """
        Simpan hasil individual (original, GT, prediction, overlay)
        
        Args:
            original: Original image
            ground_truth: GT mask
            prediction: Predicted mask
            filename: Base filename
            model_name: Nama model (untuk subfolder)
        """
        model_dir = os.path.join(self.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        base_name = os.path.splitext(filename)[0]
        
        # Save original
        cv2.imwrite(
            os.path.join(model_dir, f"{base_name}_original.png"),
            cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        )
        
        # Save ground truth
        cv2.imwrite(
            os.path.join(model_dir, f"{base_name}_groundtruth.png"),
            (ground_truth * 255).astype(np.uint8)
        )
        
        # Save prediction
        cv2.imwrite(
            os.path.join(model_dir, f"{base_name}_prediction.png"),
            (prediction * 255).astype(np.uint8)
        )
        
        # Save overlay
        overlay = self.overlay_mask_on_image(original, prediction, alpha=0.5)
        cv2.imwrite(
            os.path.join(model_dir, f"{base_name}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )
        
        # Save combined
        combined = self.create_combined_overlay(original, ground_truth, prediction)
        cv2.imwrite(
            os.path.join(model_dir, f"{base_name}_combined.png"),
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        )
    
    def visualize_model_predictions(self, model, dataloader, device, 
                                    model_name, num_samples=None, 
                                    save_individual=True):
        """
        Visualisasi prediksi untuk seluruh dataset
        
        Args:
            model: Model yang akan dievaluasi
            dataloader: DataLoader untuk dataset
            device: Device (cuda/cpu)
            model_name: Nama model untuk penamaan file
            num_samples: Jumlah sample yang akan divisualisasi (None = semua)
            save_individual: Apakah menyimpan hasil individual
        """
        model.eval()
        
        print(f"\n{'='*80}")
        print(f"VISUALISASI HASIL - {model_name}")
        print(f"{'='*80}\n")
        
        model_dir = os.path.join(self.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        count = 0
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Processing")):
                images = images.to(device)
                masks = masks.to(device)
                
                predictions = model(images)
                
                # Process setiap image dalam batch
                for i in range(images.shape[0]):
                    if num_samples and count >= num_samples:
                        break
                    
                    # Denormalize image
                    original = self.denormalize_image(images[i])
                    
                    # Convert masks
                    gt_mask = masks[i].cpu().numpy()
                    pred_mask = predictions[i].squeeze().cpu().numpy()
                    
                    filename = f"sample_{count:04d}"
                    
                    # Buat comparison grid
                    grid_path = os.path.join(model_dir, f"{filename}_comparison.png")
                    self.create_comparison_grid(
                        original, gt_mask, pred_mask,
                        title=f"{model_name} - Sample {count}",
                        save_path=grid_path
                    )
                    
                    # Save individual results
                    if save_individual:
                        self.save_individual_results(
                            original, gt_mask, pred_mask, 
                            filename, model_name
                        )
                    
                    count += 1
                
                if num_samples and count >= num_samples:
                    break
        
        print(f"\n✓ Visualisasi selesai! Total: {count} images")
        print(f"✓ Hasil disimpan di: {model_dir}/")
        print(f"  - Comparison grids: *_comparison.png")
        if save_individual:
            print(f"  - Individual results: *_original.png, *_prediction.png, dll")
    
    def create_summary_comparison(self, model_results, num_samples=5, 
                                  save_path=None):
        """
        Buat summary comparison untuk beberapa model
        
        Args:
            model_results: Dict {model_name: (original, gt, pred)}
            num_samples: Jumlah sample untuk summary
            save_path: Path untuk menyimpan hasil
        """
        num_models = len(model_results)
        fig, axes = plt.subplots(num_samples, num_models + 2, 
                                 figsize=(4*(num_models+2), 4*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        model_names = list(model_results.keys())
        
        for row in range(num_samples):
            # Original image (column 0)
            original = model_results[model_names[0]][row][0]
            axes[row, 0].imshow(original)
            if row == 0:
                axes[row, 0].set_title('Original', fontweight='bold')
            axes[row, 0].axis('off')
            
            # Ground truth (column 1)
            gt = model_results[model_names[0]][row][1]
            axes[row, 1].imshow(gt, cmap='gray')
            if row == 0:
                axes[row, 1].set_title('Ground Truth', fontweight='bold')
            axes[row, 1].axis('off')
            
            # Predictions dari setiap model
            for col, model_name in enumerate(model_names):
                pred = model_results[model_name][row][2]
                axes[row, col + 2].imshow(pred, cmap='gray')
                if row == 0:
                    axes[row, col + 2].set_title(model_name, fontweight='bold')
                axes[row, col + 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def visualize_single_model(model, dataloader, device, model_name, 
                          save_dir='visualizations', num_samples=None):
    """
    Fungsi helper untuk visualisasi single model
    
    Args:
        model: Model untuk divisualisasi
        dataloader: DataLoader
        device: Device (cuda/cpu)
        model_name: Nama model
        save_dir: Direktori output
        num_samples: Jumlah sample (None = semua)
    """
    visualizer = MedicalImageVisualizer(save_dir=save_dir)
    visualizer.visualize_model_predictions(
        model, dataloader, device, model_name, 
        num_samples=num_samples, save_individual=True
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Medical Image Visualizer Module")
    print("="*80)
    print("\nFitur yang tersedia:")
    print("1. Visualisasi hasil prediksi (comparison grid)")
    print("2. Save individual results (original, GT, prediction, overlay)")
    print("3. Combined overlay (GT=Green, Pred=Red, Overlap=Yellow)")
    print("4. Summary comparison untuk multiple models")
    print("\n✓ Module siap digunakan!")