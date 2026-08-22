"""
Utility-Funktionen für Datensatz-Analyse und Verwaltung
"""
import os
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import config


class DatasetAnalyzer:
    @staticmethod
    def analyze_dataset(dataset_dir=None):
        """Analysiere den Datensatz"""
        if dataset_dir is None:
            dataset_dir = config.RAW_DATA_DIR
        
        dataset_dir = Path(dataset_dir)
        
        print("\n" + "="*60)
        print("DATENSATZ ANALYSE")
        print("="*60)
        
        if not dataset_dir.exists():
            print(f"❌ Verzeichnis nicht gefunden: {dataset_dir}")
            return
        
        # Sammle Statistiken
        category_stats = defaultdict(lambda: {
            'count': 0,
            'sizes': [],
            'formats': defaultdict(int)
        })
        
        total_images = 0
        total_size = 0
        
        for category in sorted(config.ALPHABET):
            category_path = dataset_dir / category
            
            if not category_path.exists():
                continue
            
            for img_file in category_path.glob('*'):
                if img_file.is_file():
                    try:
                        # Dateiinformationen
                        file_size = img_file.stat().st_size
                        file_ext = img_file.suffix.lower()
                        
                        # Bildinformationen
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            h, w = img.shape[:2]
                            
                            category_stats[category]['count'] += 1
                            category_stats[category]['sizes'].append((w, h))
                            category_stats[category]['formats'][file_ext] += 1
                            
                            total_images += 1
                            total_size += file_size
                    except Exception as e:
                        print(f"⚠️  Fehler bei {img_file}: {e}")
        
        # Drucke Ergebnisse
        print(f"\n📊 Gesamt-Statistik:")
        print(f"   Kategorien: {len(category_stats)}")
        print(f"   Bilder gesamt: {total_images}")
        print(f"   Größe gesamt: {total_size / (1024**2):.2f} MB")
        
        if total_images > 0:
            print(f"   Durchschnitt pro Kategorie: {total_images / len(category_stats):.1f} Bilder")
        
        print(f"\n📁 Nach Kategorie:")
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            if stats['count'] > 0:
                sizes = stats['sizes']
                avg_w = int(np.mean([s[0] for s in sizes]))
                avg_h = int(np.mean([s[1] for s in sizes]))
                formats = ", ".join([f"{fmt} ({cnt})" for fmt, cnt in stats['formats'].items()])
                
                print(f"   {category}: {stats['count']} Bilder | "
                      f"{avg_w}x{avg_h} | Format: {formats}")
        
        # Warnung bei unausgewogenen Daten
        if category_stats:
            counts = [s['count'] for s in category_stats.values()]
            min_count = min(counts)
            max_count = max(counts)
            
            if max_count / min_count > 2:
                print(f"\n⚠️  WARNUNG: Unausgewogene Daten!")
                print(f"   Minimum: {min_count}, Maximum: {max_count}")
                print(f"   Verhältnis: {max_count/min_count:.1f}x")
                print(f"   Empfehlung: Datensatz ausgleichen!")
        
        return category_stats
    
    @staticmethod
    def check_corrupted_images(dataset_dir=None):
        """Prüfe auf beschädigte Bilder"""
        if dataset_dir is None:
            dataset_dir = config.RAW_DATA_DIR
        
        dataset_dir = Path(dataset_dir)
        
        print("\n" + "="*60)
        print("PRÜFE AUF BESCHÄDIGTE BILDER")
        print("="*60)
        
        corrupted = []
        
        for category in sorted(config.ALPHABET):
            category_path = dataset_dir / category
            
            if not category_path.exists():
                continue
            
            for img_file in category_path.glob('*'):
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        corrupted.append(str(img_file))
                        print(f"❌ Beschädigt: {img_file.name}")
                except Exception as e:
                    corrupted.append(str(img_file))
                    print(f"❌ Fehler: {img_file.name} - {e}")
        
        if corrupted:
            print(f"\n⚠️  {len(corrupted)} beschädigte Bilder gefunden!")
        else:
            print(f"\n✓ Alle Bilder sind OK!")
        
        return corrupted
    
    @staticmethod
    def display_sample_images(dataset_dir=None, category='A', num_samples=4):
        """Zeige Beispiel-Bilder einer Kategorie"""
        if dataset_dir is None:
            dataset_dir = config.RAW_DATA_DIR
        
        dataset_dir = Path(dataset_dir)
        category_path = dataset_dir / category
        
        if not category_path.exists():
            print(f"❌ Kategorie {category} nicht gefunden!")
            return
        
        images = list(category_path.glob('*'))[:num_samples]
        
        if not images:
            print(f"❌ Keine Bilder in {category} gefunden!")
            return
        
        print(f"\nZeige {len(images)} Beispiel-Bilder von Kategorie '{category}'...")
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                print(f"  • {img_path.name}: {w}x{h}px")
                
                # Optional: Display
                # cv2.imshow(f"Category {category}", img)
                # cv2.waitKey(2000)
                # cv2.destroyAllWindows()


def print_project_structure():
    """Zeige Projekt-Struktur"""
    print("\n" + "="*60)
    print("PROJEKT-STRUKTUR")
    print("="*60)
    
    structure = {
        "data/raw/": "Hier Trainingsbilder einfügen (A, B, C, ..., Z)",
        "data/train/": "Automatisch generiert (Training 70%)",
        "data/val/": "Automatisch generiert (Validierung 15%)",
        "data/test/": "Automatisch generiert (Test 15%)",
        "models/": "Trainierte Modelle speichern",
        "logs/": "TensorBoard Logs",
        "output/": "Vorhersage-Ergebnisse",
        "config.py": "Konfigurationen",
        "dataset_manager.py": "Datensatz-Verwaltung",
        "model.py": "CNN + MediaPipe Modell",
        "train.py": "Training-Skript",
        "predict.py": "Vorhersagen",
        "augment.py": "Daten-Augmentation",
    }
    
    for path, description in structure.items():
        print(f"  {path:<20} - {description}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SIGN LANGUAGE RECOGNITION - UTILITY")
    print("="*80)
    
    # 1. Projekt-Struktur
    print_project_structure()
    
    # 2. Analyse
    DatasetAnalyzer.analyze_dataset()
    
    # 3. Beschädigte Bilder prüfen
    DatasetAnalyzer.check_corrupted_images()
    
    # 4. Beispiel-Bilder
    DatasetAnalyzer.display_sample_images()
    
    print("\n" + "="*80)
    print("✓ Analyse fertig!")
    print("="*80 + "\n")
