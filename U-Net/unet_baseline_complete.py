"""
U-Net Baseline - Complete Implementation
Includes: Model Architecture, Training, Evaluation, Visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from metrics import SegmentationMetrics
from visualizer import MedicalImageVisualizer

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class DoubleConv(nn.Module):
    """Double Convolution Block"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetBaseline(nn.Module):
    """U-Net Baseline (Standard Architecture)"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetBaseline, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature*2, feature))
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx//2]
            
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
        
        return torch.sigmoid(self.final_conv(x))


# =============================================================================
# DATASET
# =============================================================================

class SegmentationDataset(Dataset):
    """Dataset untuk Image Segmentation"""
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask > 0] = 1.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask


def get_transforms(img_size=256):
    """Transformasi untuk training dan validation"""
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    """Dice Loss untuk segmentasi"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Kombinasi BCE Loss dan Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    """Training untuk satu epoch"""
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.unsqueeze(1).to(device)
        
        # Forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)


def validate(loader, model, loss_fn, device):
    """Validasi model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Validation"):
            data = data.to(device)
            targets = targets.unsqueeze(1).to(device)
            
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
    
    return total_loss / len(loader)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, loader, device, threshold=0.5):
    """Evaluasi model dengan metrics lengkap"""
    model.eval()
    metrics = SegmentationMetrics(threshold=threshold)
    
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            targets = targets.unsqueeze(1).to(device)
            
            predictions = model(data)
            
            # Hitung metrik untuk setiap sample
            for i in range(predictions.shape[0]):
                metrics.calculate_all_metrics(predictions[i], targets[i])
    
    return metrics


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def train_baseline(args):
    """Training U-Net Baseline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print("TRAINING U-NET BASELINE")
    print(f"{'='*80}")
    print(f"Device: {device}")
    
    # Model
    model = UNetBaseline(in_channels=3, out_channels=1).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data
    train_transform, val_transform = get_transforms(args.img_size)
    
    train_ds = SegmentationDataset(args.train_img_dir, args.train_mask_dir, train_transform)
    val_ds = SegmentationDataset(args.val_img_dir, args.val_mask_dir, val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # Loss dan optimizer
    loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        val_loss = validate(val_loader, model, loss_fn, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, args.checkpoint_path)
            print(f"✓ Best model saved! Val Loss: {best_val_loss:.4f}")
    
    print(f"\n✓ Training completed! Best Val Loss: {best_val_loss:.4f}")


def evaluate_baseline(args):
    """Evaluasi U-Net Baseline dengan visualisasi"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print("EVALUASI U-NET BASELINE")
    print(f"{'='*80}")
    
    # Load model
    model = UNetBaseline(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("✓ Model loaded successfully")
    
    # Data
    _, test_transform = get_transforms(args.img_size)
    test_ds = SegmentationDataset(args.test_img_dir, args.test_mask_dir, test_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Test samples: {len(test_ds)}")
    
    # Evaluate metrics
    print("\n" + "="*80)
    print("MENGHITUNG METRIK")
    print("="*80)
    metrics = evaluate_model(model, test_loader, device, args.threshold)
    
    # Print results
    metrics.print_metrics("U-Net Baseline")
    
    # Save results
    summary_df = metrics.get_metrics_summary()
    summary_df.to_csv(args.output_csv, index=False)
    print(f"✓ Results saved to: {args.output_csv}")
    
    # Visualisasi hasil prediksi
    if args.visualize:
        print("\n" + "="*80)
        print("MEMBUAT VISUALISASI HASIL CITRA MEDIS")
        print("="*80)
        
        visualizer = MedicalImageVisualizer(save_dir=args.vis_dir)
        visualizer.visualize_model_predictions(
            model=model,
            dataloader=test_loader,
            device=device,
            model_name="UNet_Baseline",
            num_samples=args.num_vis_samples,
            save_individual=True
        )
        
        print(f"\n✓ Visualisasi selesai!")
        print(f"✓ Hasil disimpan di: {args.vis_dir}/UNet_Baseline/")
        print(f"  - Comparison grids (6 panel)")
        print(f"  - Original images")
        print(f"  - Predicted masks")
        print(f"  - Overlay results")
        print(f"  - Combined results (color-coded)")
    
    return metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='U-Net Baseline - Training & Evaluation')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True)
    
    # Data paths
    parser.add_argument('--train_img_dir', type=str, default='data/processed/train/images')
    parser.add_argument('--train_mask_dir', type=str, default='data/processed/train/masks')
    parser.add_argument('--val_img_dir', type=str, default='data/processed/val/images')
    parser.add_argument('--val_mask_dir', type=str, default='data/processed/val/masks')
    parser.add_argument('--test_img_dir', type=str, default='data/processed/test/images')
    parser.add_argument('--test_mask_dir', type=str, default='data/processed/test/masks')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    
    # Checkpoint
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/unet_baseline.pth.tar')
    
    # Evaluation params
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_csv', type=str, default='results/baseline_metrics.csv')
    
    # Visualization params
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Visualization directory')
    parser.add_argument('--num_vis_samples', type=int, default=None, 
                       help='Number of samples to visualize (None=all)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)
    
    if args.mode == 'train':
        train_baseline(args)
    elif args.mode == 'eval':
        evaluate_baseline(args)