"""
Quick Visualization Script
Untuk cepat membuat visualisasi hasil citra medis tanpa evaluasi full
"""

import torch
import argparse
import os
from torch.utils.data import DataLoader
from unet_baseline_complete import UNetBaseline, SegmentationDataset, get_transforms
from unet_attention_complete import UNetAttention
from visualizer import MedicalImageVisualizer

def main():
    parser = argparse.ArgumentParser(description='Quick Visualization Tool')
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['baseline', 'attention', 'both'], 
                       required=True, help='Which model to visualize')
    
    # Data
    parser.add_argument('--test_img_dir', type=str, default='data/processed/test/images')
    parser.add_argument('--test_mask_dir', type=str, default='data/processed/test/masks')
    
    # Checkpoints
    parser.add_argument('--baseline_checkpoint', type=str, 
                       default='checkpoints/unet_baseline.pth.tar')
    parser.add_argument('--attention_checkpoint', type=str,
                       default='checkpoints/unet_attention.pth.tar')
    
    # Visualization settings
    parser.add_argument('--output_dir', type=str, default='quick_visualizations',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    
    # Options
    parser.add_argument('--save_individual', action='store_true',
                       help='Save individual files (original, mask, pred, overlay)')
    parser.add_argument('--comparison_grid_only', action='store_true',
                       help='Only save comparison grids (faster)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("QUICK VISUALIZATION TOOL")
    print("="*80)
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}/")
    print(f"Samples: {args.num_samples}")
    
    # Load data
    _, test_transform = get_transforms(args.img_size)
    test_ds = SegmentationDataset(args.test_img_dir, args.test_mask_dir, test_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"Test dataset: {len(test_ds)} images\n")
    
    # Initialize visualizer
    visualizer = MedicalImageVisualizer(save_dir=args.output_dir)
    
    # Visualize based on selection
    if args.model in ['baseline', 'both']:
        print("="*80)
        print("VISUALIZING U-NET BASELINE")
        print("="*80)
        
        baseline_model = UNetBaseline(in_channels=3, out_channels=1).to(device)
        checkpoint = torch.load(args.baseline_checkpoint, map_location=device)
        baseline_model.load_state_dict(checkpoint['state_dict'])
        print("✓ Model loaded")
        
        visualizer.visualize_model_predictions(
            model=baseline_model,
            dataloader=test_loader,
            device=device,
            model_name="UNet_Baseline",
            num_samples=args.num_samples,
            save_individual=not args.comparison_grid_only and args.save_individual
        )
    
    if args.model in ['attention', 'both']:
        print("\n" + "="*80)
        print("VISUALIZING U-NET ATTENTION GATE")
        print("="*80)
        
        attention_model = UNetAttention(in_channels=3, out_channels=1).to(device)
        checkpoint = torch.load(args.attention_checkpoint, map_location=device)
        attention_model.load_state_dict(checkpoint['state_dict'])
        print("✓ Model loaded")
        
        visualizer.visualize_model_predictions(
            model=attention_model,
            dataloader=test_loader,
            device=device,
            model_name="UNet_Attention",
            num_samples=args.num_samples,
            save_individual=not args.comparison_grid_only and args.save_individual
        )
    
    # Summary
    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETED!")
    print("="*80)
    print(f"\nResults saved in: {args.output_dir}/")
    
    if args.model == 'both':
        print("\nFolders created:")
        print("  - UNet_Baseline/")
        print("  - UNet_Attention/")
    else:
        print(f"\nFolder created: UNet_{args.model.title()}/")
    
    print("\nFiles per sample:")
    print("  - *_comparison.png (always)")
    if not args.comparison_grid_only and args.save_individual:
        print("  - *_original.png")
        print("  - *_groundtruth.png")
        print("  - *_prediction.png")
        print("  - *_overlay.png")
        print("  - *_combined.png")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()