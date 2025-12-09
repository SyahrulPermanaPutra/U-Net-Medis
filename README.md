research_project/
│
├── data/
│   ├── raw/                          # Data asli
│   │   ├── annotations/              # 37 file XML
│   │   └── images/                   # 37 tissue images
│   │
│   └── processed/                    # Hasil preprocessing
│       ├── train/
│       │   ├── images/               # ~26 images
│       │   └── masks/                # ~26 masks
│       ├── val/
│       │   ├── images/               # ~5-6 images
│       │   └── masks/                # ~5-6 masks
│       └── test/
│           ├── images/               # ~5-6 images
│           └── masks/                # ~5-6 masks
│
├── checkpoints/                      # Model checkpoints
│   ├── unet_baseline.pth.tar
│   └── unet_attention.pth.tar
│
├── results/                          # Hasil evaluasi
│   ├── baseline_metrics.csv
│   ├── attention_metrics.csv
│   ├── model_comparison.csv
│   ├── comparison_plot.png
│   ├── improvement_plot.png
│   └── detailed_report.txt
│
├── data_processor.py                 # XML to mask converter
├── metrics.py                        # Reusable metrics module
├── unet_baseline_complete.py         # U-Net Baseline (lengkap)
├── unet_attention_complete.py        # U-Net Attention (lengkap)
└── compare_models.py                 # Script perbandingan

# U-Net-Medis

#INSTALASI DEPENDENCIES#

# Create virtual environment (opsional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# Install packages
pip install torch torchvision
pip install albumentations
pip install opencv-python
pip install Pillow
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tqdm

STEP 1: Preprocessing Data XML
Siapkan Struktur FOlder, dan masukkan dataset yang dipakai yaitu Monuseg 2018 pada folder data/raw. Lalu
Run "python data_processor.py" 

STEP 2: Training U-Net Baseline
python unet_baseline_complete.py --mode train \
  --train_img_dir data/processed/train/images \
  --train_mask_dir data/processed/train/masks \
  --val_img_dir data/processed/val/images \
  --val_mask_dir data/processed/val/masks \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --img_size 256 \
  --checkpoint_path checkpoints/unet_baseline.pth.tar

  STEP 3: Training U-Net Attention Gate
  python unet_attention_complete.py --mode train \
  --train_img_dir data/processed/train/images \
  --train_mask_dir data/processed/train/masks \
  --val_img_dir data/processed/val/images \
  --val_mask_dir data/processed/val/masks \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --img_size 256 \
  --checkpoint_path checkpoints/unet_attention.pth.tar

  STEP 4: Evaluasi Individual Model
  Evaluasi Baseline:
python unet_baseline_complete.py --mode eval \
  --test_img_dir data/processed/test/images \
  --test_mask_dir data/processed/test/masks \
  --checkpoint_path checkpoints/unet_baseline.pth.tar \
  --threshold 0.5 \
  --output_csv results/baseline_metrics.csv
Evaluasi Attention:
python unet_attention_complete.py --mode eval \
  --test_img_dir data/processed/test/images \
  --test_mask_dir data/processed/test/masks \
  --checkpoint_path checkpoints/unet_attention.pth.tar \
  --threshold 0.5 \
  --output_csv results/attention_metrics.csv

  STEP 5: Perbandingan Kedua Model 
  python compare_models.py \
  --test_img_dir data/processed/test/images \
  --test_mask_dir data/processed/test/masks \
  --baseline_checkpoint checkpoints/unet_baseline.pth.tar \
  --attention_checkpoint checkpoints/unet_attention.pth.tar \
  --batch_size 8 \
  --img_size 256 \
  --threshold 0.5 \
  --output_dir results
