import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import traceback

class XMLAnnotationProcessor:
    """
    Processor untuk mengkonversi XML annotation menjadi binary mask
    dan membagi dataset menjadi train/val/test
    """
    
    def __init__(self, xml_dir, image_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Args:
            xml_dir: Direktori berisi file XML (37 files)
            image_dir: Direktori berisi tissue images (37 files)
            output_dir: Direktori output untuk menyimpan hasil
            train_ratio: Proporsi data training (default: 0.7)
            val_ratio: Proporsi data validation (default: 0.15)
            test_ratio: Proporsi data testing (default: 0.15)
        """
        self.xml_dir = xml_dir
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validasi proporsi
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            print("Warning: Total ratio = %.3f, seharusnya 1.0. Menyesuaikan..." % total)
            self.train_ratio = train_ratio / total
            self.val_ratio = val_ratio / total
            self.test_ratio = test_ratio / total
    
    def parse_xml_to_mask(self, xml_path, image_shape):
        """
        Parse XML annotation menjadi binary mask
        SESUAI DENGAN FORMAT XML ANDA (Region dengan Vertices)
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Buat mask kosong
            mask = np.zeros(image_shape, dtype=np.uint8)
            
            print("  Parsing: %s" % os.path.basename(xml_path))
            print("  Image shape: %s" % str(image_shape))
            
            # Cari semua Region - beberapa kemungkinan path
            regions = []
            
            # Coba beberapa kemungkinan struktur
            if root.find('Region') is not None:
                regions = root.findall('Region')
            elif root.find('.//Region') is not None:
                regions = root.findall('.//Region')
            else:
                # Cari semua tag Region di seluruh dokumen
                for elem in root.iter():
                    if elem.tag == 'Region':
                        regions.append(elem)
            
            print("  Ditemukan %d region(s)" % len(regions))
            
            if len(regions) == 0:
                print("  Warning: Tidak ada region ditemukan di %s" % xml_path)
                return mask
            
            for i, region in enumerate(regions):
                print("  Processing region %d" % (i+1))
                
                # Cari Vertices
                vertices_elem = region.find('Vertices')
                if vertices_elem is None:
                    # Coba cari di child
                    for elem in region.iter():
                        if elem.tag == 'Vertices':
                            vertices_elem = elem
                            break
                
                if vertices_elem is None:
                    print("    Warning: Tidak ditemukan Vertices di region %d" % (i+1))
                    continue
                
                points = []
                vertex_count = 0
                
                # Kumpulkan semua titik vertex
                for vertex in vertices_elem.findall('Vertex'):
                    x_attr = vertex.get('X')
                    y_attr = vertex.get('Y')
                    
                    if x_attr and y_attr:
                        try:
                            x = float(x_attr)
                            y = float(y_attr)
                            points.append([x, y])
                            vertex_count += 1
                        except ValueError as e:
                            print("    Error parsing vertex: %s" % e)
                            continue
                
                print("    Ditemukan %d vertices" % vertex_count)
                
                # Konversi ke integer dan buat polygon
                if len(points) >= 3:
                    points_array = np.array(points, dtype=np.int32)
                    
                    # Isi polygon dengan warna putih (255)
                    cv2.fillPoly(mask, [points_array], 255)
                    
                    print("    Polygon dibuat dengan %d titik" % len(points))
                else:
                    print("    Warning: Hanya %d titik, minimal 3" % len(points))
            
            # Cek mask hasil
            mask_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            coverage = mask_pixels / total_pixels * 100 if total_pixels > 0 else 0
            
            print("  Mask coverage: %.2f%% (%d/%d pixels)" % (coverage, mask_pixels, total_pixels))
            
            return mask
            
        except Exception as e:
            print("  ERROR parsing %s: %s" % (xml_path, str(e)))
            return np.zeros(image_shape, dtype=np.uint8)
    
    def debug_xml_structure(self, xml_file=None):
        """
        Debug fungsi untuk melihat struktur XML
        """
        if xml_file is None:
            xml_files = [f for f in os.listdir(self.xml_dir) if f.endswith('.xml')]
            if xml_files:
                xml_file = xml_files[0]
            else:
                print("ERROR: Tidak ada file XML di direktori!")
                return None
        
        xml_path = os.path.join(self.xml_dir, xml_file)
        
        if not os.path.exists(xml_path):
            print("ERROR: File tidak ditemukan: %s" % xml_path)
            return None
        
        print("\nDEBUG struktur XML: %s" % os.path.basename(xml_path))
        print("="*60)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Print informasi root
            print("Root tag: %s" % root.tag)
            print("Root attributes: %s" % str(dict(root.attrib)))
            
            # Print child langsung
            print("\nChild elements dari root:")
            for child in root:
                print("  - %s: %s" % (child.tag, str(dict(child.attrib))[:50]))
            
            # Cari Region
            regions = []
            for elem in root.iter('Region'):
                regions.append(elem)
            
            print("\nTotal Region ditemukan: %d" % len(regions))
            
            # Lihat contoh Region pertama
            if regions:
                region = regions[0]
                print("\nContoh Region pertama:")
                print("  Attributes: %s" % str(dict(region.attrib)))
                
                # Cari Vertices
                vertices = None
                for elem in region.iter():
                    if elem.tag == 'Vertices':
                        vertices = elem
                        break
                
                if vertices:
                    vertex_list = vertices.findall('Vertex')
                    print("  Jumlah Vertex: %d" % len(vertex_list))
                    
                    # Tampilkan 3 vertex pertama
                    for i, vertex in enumerate(vertex_list[:3]):
                        attrs = dict(vertex.attrib)
                        print("  Vertex %d: %s" % (i+1, attrs))
                    
                    if len(vertex_list) > 3:
                        print("  ... dan %d vertex lainnya" % (len(vertex_list)-3))
                else:
                    print("  ERROR: Tidak ditemukan Vertices!")
                    
            print("="*60)
            
            return xml_path
            
        except Exception as e:
            print("ERROR parsing XML: %s" % str(e))
            return None
    
    def check_directories(self):
        """Cek apakah direktori ada dan berisi file"""
        print("\nCHECKING DIRECTORIES")
        print("="*60)
        
        # Cek XML directory
        if not os.path.exists(self.xml_dir):
            print("ERROR: XML directory tidak ditemukan: %s" % self.xml_dir)
            return False
        
        xml_files = [f for f in os.listdir(self.xml_dir) if f.endswith('.xml')]
        print("XML directory: %s" % self.xml_dir)
        print("  Jumlah file XML: %d" % len(xml_files))
        if xml_files:
            print("  5 file pertama: %s" % str(xml_files[:5]))
        
        # Cek Image directory
        if not os.path.exists(self.image_dir):
            print("ERROR: Image directory tidak ditemukan: %s" % self.image_dir)
            return False
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(image_extensions)]
        print("\nImage directory: %s" % self.image_dir)
        print("  Jumlah file image: %d" % len(image_files))
        if image_files:
            print("  5 file pertama: %s" % str(image_files[:5]))
        
        # Cek output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print("\nOutput directory: %s" % self.output_dir)
        
        print("="*60)
        
        return len(xml_files) > 0 and len(image_files) > 0
    
    def create_directory_structure(self):
        """Buat struktur direktori untuk output"""
        splits = ['train', 'val', 'test']
        subdirs = ['images', 'masks']
        
        for split in splits:
            for subdir in subdirs:
                path = os.path.join(self.output_dir, split, subdir)
                os.makedirs(path, exist_ok=True)
        
        print("Struktur direktori dibuat di: %s" % self.output_dir)
    
    def match_files(self):
        """Match XML files dengan Image files"""
        print("\nMATCHING XML DENGAN IMAGE FILES")
        print("="*60)
        
        xml_files = sorted([f for f in os.listdir(self.xml_dir) if f.endswith('.xml')])
        image_files = sorted([f for f in os.listdir(self.image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
        
        paired_data = []
        
        # Strategy 1: Exact name match (tanpa ekstensi)
        print("\nStrategy 1: Exact name match...")
        for xml_file in xml_files:
            xml_base = os.path.splitext(xml_file)[0]
            
            for img_file in image_files:
                img_base = os.path.splitext(img_file)[0]
                
                if xml_base == img_base:
                    paired_data.append((xml_file, img_file))
                    print("  MATCH: %s -> %s" % (xml_file, img_file))
                    break
        
        # Strategy 2: Partial match (jika exact tidak cukup)
        if len(paired_data) < len(xml_files):
            print("\nStrategy 2: Partial match...")
            unmatched_xml = [f for f in xml_files if f not in [p[0] for p in paired_data]]
            unmatched_img = [f for f in image_files if f not in [p[1] for p in paired_data]]
            
            for xml_file in unmatched_xml:
                xml_base = os.path.splitext(xml_file)[0].lower()
                
                for img_file in unmatched_img:
                    img_base = os.path.splitext(img_file)[0].lower()
                    
                    # Cek jika salah satu mengandung yang lain
                    if xml_base in img_base or img_base in xml_base:
                        paired_data.append((xml_file, img_file))
                        print("  PARTIAL MATCH: %s -> %s" % (xml_file, img_file))
                        unmatched_img.remove(img_file)
                        break
        
        print("\nHasil matching:")
        print("  Total XML files: %d" % len(xml_files))
        print("  Total Image files: %d" % len(image_files))
        print("  Berhasil dipasangkan: %d" % len(paired_data))
        
        if len(paired_data) < len(xml_files):
            print("  WARNING: %d XML files tidak punya pasangan" % (len(xml_files) - len(paired_data)))
        
        return paired_data
    
    def process_and_split_dataset(self):
        """
        Proses semua XML menjadi mask dan split dataset
        """
        print("\n" + "="*80)
        print("MEMPROSES DATASET XML ANNOTATION")
        print("="*80)
        
        # 1. Cek direktori
        if not self.check_directories():
            print("ERROR: Ada masalah dengan direktori, proses dihentikan")
            return
        
        # 2. Debug struktur XML
        self.debug_xml_structure()
        
        # 3. Match files
        paired_data = self.match_files()
        
        if len(paired_data) == 0:
            print("ERROR: Tidak ada pasangan yang ditemukan!")
            return
        
        # 4. Buat struktur direktori
        self.create_directory_structure()
        
        # 5. Split dataset
        print("\n" + "="*60)
        print("SPLITTING DATASET")
        print("="*60)
        
        # Pastikan ada cukup data untuk split
        if len(paired_data) < 3:
            print("WARNING: Terlalu sedikit data untuk split!")
            print("  Semua data akan masuk ke training")
            train_data = paired_data
            val_data = []
            test_data = []
        else:
            # Split dataset
            train_val_data, test_data = train_test_split(
                paired_data, 
                test_size=self.test_ratio,
                random_state=42,
                shuffle=True
            )
            
            val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_ratio_adjusted,
                random_state=42,
                shuffle=True
            )
        
        print("\nPEMBAGIAN DATASET:")
        print("  Training:   %d files" % len(train_data))
        print("  Validation: %d files" % len(val_data))
        print("  Testing:    %d files" % len(test_data))
        
        # 6. Process dan save files
        print("\n" + "="*60)
        print("PROCESSING AND SAVING FILES")
        print("="*60)
        
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        stats = {'total': len(paired_data)}
        
        for split_name, split_data in splits.items():
            if len(split_data) == 0:
                print("\n%s: Tidak ada data" % split_name.upper())
                continue
            
            print("\nProcessing %s data (%d files):" % (split_name.upper(), len(split_data)))
            
            success_count = 0
            error_count = 0
            
            for xml_file, img_file in tqdm(split_data, desc=split_name):
                try:
                    # Baca image
                    img_path = os.path.join(self.image_dir, img_file)
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        # Coba dengan PIL
                        try:
                            pil_img = Image.open(img_path)
                            image = np.array(pil_img)
                            if len(image.shape) == 2:
                                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        except:
                            print("  ERROR: Gagal load image: %s" % img_file)
                            error_count += 1
                            continue
                    
                    # Parse XML ke mask
                    xml_path = os.path.join(self.xml_dir, xml_file)
                    mask = self.parse_xml_to_mask(xml_path, image.shape[:2])
                    
                    # Buat base name
                    base_name = os.path.splitext(img_file)[0]
                    
                    # Save image
                    img_output_path = os.path.join(self.output_dir, split_name, 'images', f"{base_name}.png")
                    cv2.imwrite(img_output_path, image)
                    
                    # Save mask (binary: 0 dan 255)
                    mask_output_path = os.path.join(self.output_dir, split_name, 'masks', f"{base_name}.png")
                    cv2.imwrite(mask_output_path, mask)
                    
                    success_count += 1
                    
                except Exception as e:
                    print("  ERROR processing %s: %s" % (xml_file, str(e)))
                    error_count += 1
            
            stats[split_name] = {
                'success': success_count,
                'error': error_count,
                'total': len(split_data)
            }
            
            print("  Success: %d, Error: %d" % (success_count, error_count))
        
        # 7. Buat summary
        print("\n" + "="*60)
        print("CREATING SUMMARY")
        print("="*60)
        
        self.create_summary_file(stats, paired_data)
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE!")
        print("="*80)
    
    def create_summary_file(self, stats, paired_data):
        """Buat file summary dataset"""
        summary_path = os.path.join(self.output_dir, "dataset_summary.txt")
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("DATASET PROCESSING SUMMARY\n")
                f.write("="*80 + "\n\n")
                
                f.write("CONFIGURATION:\n")
                f.write("  XML Directory: %s\n" % self.xml_dir)
                f.write("  Image Directory: %s\n" % self.image_dir)
                f.write("  Output Directory: %s\n" % self.output_dir)
                f.write("  Split Ratios: Train=%.2f, Val=%.2f, Test=%.2f\n\n" % 
                       (self.train_ratio, self.val_ratio, self.test_ratio))
                
                f.write("FILE STATISTICS:\n")
                xml_files_all = [f for f in os.listdir(self.xml_dir) if f.endswith('.xml')]
                img_files_all = [f for f in os.listdir(self.image_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                f.write("  Total XML files found: %d\n" % len(xml_files_all))
                f.write("  Total Image files found: %d\n" % len(img_files_all))
                f.write("  Successfully paired: %d\n\n" % stats['total'])
                
                f.write("SPLIT DISTRIBUTION:\n")
                for split in ['train', 'val', 'test']:
                    if split in stats:
                        f.write("  %s:\n" % split.upper())
                        f.write("    Total files: %d\n" % stats[split]['total'])
                        f.write("    Successfully processed: %d\n" % stats[split]['success'])
                        f.write("    Errors: %d\n" % stats[split]['error'])
                
                f.write("\nOUTPUT STRUCTURE:\n")
                f.write("  %s/\n" % self.output_dir)
                f.write("  ├── train/\n")
                f.write("  │   ├── images/     # Training images\n")
                f.write("  │   └── masks/      # Training masks\n")
                f.write("  ├── val/\n")
                f.write("  │   ├── images/     # Validation images\n")
                f.write("  │   └── masks/      # Validation masks\n")
                f.write("  ├── test/\n")
                f.write("  │   ├── images/     # Testing images\n")
                f.write("  │   └── masks/      # Testing masks\n")
                f.write("  └── dataset_summary.txt\n\n")
                
                f.write("FILE PAIRINGS:\n")
                for xml_file, img_file in paired_data:
                    f.write("  %s -> %s\n" % (xml_file, img_file))
            
            print("Summary file disimpan: %s" % summary_path)
            
        except Exception as e:
            print("ERROR membuat summary file: %s" % str(e))
    
    def create_visualization(self):
        """Buat visualisasi sederhana"""
        try:
            # Cari data di train folder
            train_img_dir = os.path.join(self.output_dir, 'train', 'images')
            train_mask_dir = os.path.join(self.output_dir, 'train', 'masks')
            
            if os.path.exists(train_img_dir):
                image_files = os.listdir(train_img_dir)
                if len(image_files) > 0:
                    # Ambil file pertama
                    sample_file = image_files[0]
                    
                    # Load image dan mask
                    img_path = os.path.join(train_img_dir, sample_file)
                    mask_path = os.path.join(train_mask_dir, sample_file)
                    
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        print("\nVisualization sample: %s" % sample_file)
                        
                        # Simpan informasi visualisasi sederhana
                        viz_path = os.path.join(self.output_dir, "visualization_info.txt")
                        with open(viz_path, 'w', encoding='utf-8') as f:
                            f.write("Sample visualization file: %s\n" % sample_file)
                            f.write("Image: %s\n" % img_path)
                            f.write("Mask: %s\n" % mask_path)
                        
                        print("Visualization info disimpan: %s" % viz_path)
                    
        except Exception as e:
            print("WARNING: Tidak bisa membuat visualisasi: %s" % str(e))

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("XML ANNOTATION PROCESSOR")
    print("="*80)
    
    # SESUAIKAN PATH INI DENGAN STRUKTUR FOLDER ANDA
    XML_DIR = "data/raw/annotations"      # Folder berisi 37 file .xml
    IMAGE_DIR = "data/raw/images"         # Folder berisi 37 tissue images
    OUTPUT_DIR = "data/processed"         # Output folder
    
    print("\nPATHS:")
    print("  XML Directory: %s" % XML_DIR)
    print("  Image Directory: %s" % IMAGE_DIR)
    print("  Output Directory: %s" % OUTPUT_DIR)
    
    # Buat output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Inisialisasi processor
    processor = XMLAnnotationProcessor(
        xml_dir=XML_DIR,
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.7,    # 70% untuk training
        val_ratio=0.15,     # 15% untuk validation
        test_ratio=0.15     # 15% untuk testing
    )
    
    # Jalankan proses
    try:
        processor.process_and_split_dataset()
        processor.create_visualization()
        
    except Exception as e:
        print("\nERROR: %s" % str(e))
    
    print("\n" + "="*80)
    print("PROCESS FINISHED")
    print("="*80)
    print("\nStruktur output yang dihasilkan:")
    print("  %s/" % OUTPUT_DIR)
    print("  ├── train/")
    print("  │   ├── images/     # Training images (~26 files)")
    print("  │   └── masks/      # Training masks (~26 files)")
    print("  ├── val/")
    print("  │   ├── images/     # Validation images (~5-6 files)")
    print("  │   └── masks/      # Validation masks (~5-6 files)")
    print("  ├── test/")
    print("  │   ├── images/     # Testing images (~5-6 files)")
    print("  │   └── masks/      # Testing masks (~5-6 files)")
    print("  └── dataset_summary.txt")
    print("="*80)