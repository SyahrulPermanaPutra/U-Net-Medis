# 🧬 U-Net Medis — Nuclei Segmentation (MoNuSeg 2018)

Repositori ini berisi pipeline lengkap untuk *semantic segmentation* inti sel (nuclei) menggunakan dua arsitektur:

* **U-Net Baseline**
* **U-Net dengan Attention Gate**

Dataset yang digunakan: **MoNuSeg 2018**
Proyek mencakup preprocessing dataset, training model, evaluasi, dan perbandingan performa.

---

## 📂 Struktur Direktori

```
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
│       │   ├── images/               # ~5–6 images
│       │   └── masks/
│       └── test/
│           ├── images/               # ~5–6 images
│           └── masks/
│
├── checkpoints/                      # Model checkpoints
│   ├── unet_baseline.pth.tar
│   └── unet_attention.pth.tar
│
├── results/                          # Hasil evaluasi & visualisasi
│   ├── baseline_metrics.csv
│   ├── attention_metrics.csv
│   ├── model_comparison.csv
│   ├── comparison_plot.png
│   ├── improvement_plot.png
│   └── detailed_report.txt
│
├── data_processor.py                 # Konversi XML → mask
├── metrics.py                        # Modul perhitungan metrik
├── unet_baseline_complete.py         # Model U-Net Baseline
├── unet_attention_complete.py        # Model U-Net dengan Attention
└── compare_models.py                 # Perbandingan model
```

---

## ⚙️ Instalasi Dependencies

### 1. Buat Virtual Environment

**Windows**

```
python -m venv venv
venv\Scripts\activate
```

**Linux / MacOS**

```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Requirements

```
pip install torch torchvision
pip install albumentations
pip install opencv-python
pip install Pillow
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tqdm
```

---

## 🧩 STEP 1 — Preprocessing (XML → Mask)

1. Siapkan folder sesuai struktur.
2. Masukkan dataset MoNuSeg 2018 ke dalam:

```
data/raw/images/
data/raw/annotations/
```

3. Jalankan preprocess:

```
python data_processor.py
```

Script akan:

* membaca XML
* membuat binary mask
* melakukan split train/val/test
* menyimpan hasilnya ke `data/processed/`

---

## 🏗️ STEP 2 — Training U-Net Baseline

```
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
```

---

## 🧠 STEP 3 — Training U-Net dengan Attention Gate

```
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
```

---

## 📊 STEP 4 — Evaluasi Model (Individual)

### Evaluasi U-Net Baseline

```
python unet_baseline_complete.py --mode eval \
  --test_img_dir data/processed/test/images \
  --test_mask_dir data/processed/test/masks \
  --checkpoint_path checkpoints/unet_baseline.pth.tar \
  --threshold 0.5 \
  --output_csv results/baseline_metrics.csv
```

### Evaluasi U-Net Attention

```
python unet_attention_complete.py --mode eval \
  --test_img_dir data/processed/test/images \
  --test_mask_dir data/processed/test/masks \
  --checkpoint_path checkpoints/unet_attention.pth.tar \
  --threshold 0.5 \
  --output_csv results/attention_metrics.csv
```

---

## ⚔️ STEP 5 — Perbandingan Model

```
python compare_models.py \
  --test_img_dir data/processed/test/images \
  --test_mask_dir data/processed/test/masks \
  --baseline_checkpoint checkpoints/unet_baseline.pth.tar \
  --attention_checkpoint checkpoints/unet_attention.pth.tar \
  --batch_size 8 \
  --img_size 256 \
  --threshold 0.5 \
  --output_dir results
```
