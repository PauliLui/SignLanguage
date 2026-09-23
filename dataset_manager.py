"""
Dataset Manager: Lädt Datensatz, teilt in Train/Val/Test auf
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from tqdm import tqdm
import config


class DatasetManager:
    def __init__(self):
        self.raw_data_dir = config.RAW_DATA_DIR
        self.processed_dir = config.PROCESSED_DATA_DIR
        self.train_dir = config.TRAIN_DIR
        self.val_dir = config.VAL_DIR
        self.test_dir = config.TEST_DIR
        
        # Erstelle Verzeichnisse
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_dataset_from_raw(self):
        """Laden Sie alle Bilder aus data/raw und teilen Sie sie auf"""
        print("\n" + "="*60)
        print("DATENSATZ WIRD GELADEN UND AUFGETEILT")
        print("="*60)
        
        if not self.raw_data_dir.exists():
            print(f"❌ Fehler: {self.raw_data_dir} existiert nicht!")
            print(f"Bitte erstellen Sie: {self.raw_data_dir}")
            print("und fügen Sie Unterordner für jeden Buchstaben hinzu (A, B, C, ..., Z)")
            return False
        
        # Überprüfe ob Daten vorhanden sind
        categories = [d for d in os.listdir(self.raw_data_dir) 
                     if os.path.isdir(os.path.join(self.raw_data_dir, d))]
        
        if not categories:
            print(f"❌ Keine Daten gefunden in {self.raw_data_dir}")
            return False
        
        print(f"✓ Gefundene Kategorien: {sorted(categories)}")
        print(f"✓ Anzahl Kategorien: {len(categories)}")
        
        # Sammle alle Dateipfade mit Labels
        all_files = []
        
        for category in sorted(categories):
            category_path = self.raw_data_dir / category
            image_files = self._get_valid_images(category_path)
            
            for img_file in image_files:
                all_files.append((str(img_file), category))
            
            print(f"  • {category}: {len(image_files)} Bilder")
        
        if not all_files:
            print("❌ Keine gültigen Bilder gefunden!")
            return False
        
        print(f"\n✓ Gesamt Bilder: {len(all_files)}")
        
        # Teile in Train/Val/Test auf
        self._split_and_copy_files(all_files)
        
        return True
    
    def _get_valid_images(self, directory):
        """Finde alle gültigen Bildformate"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = []
        
        for file in os.listdir(directory):
            if Path(file).suffix.lower() in valid_extensions:
                images.append(Path(directory) / file)
        
        return images
    
    def _split_and_copy_files(self, all_files):
        """Teile Dateien auf und kopiere sie"""
        train_split = config.DATASET_CONFIG['train_split']
        val_split = config.DATASET_CONFIG['val_split']
        test_split = config.DATASET_CONFIG['test_split']
        random_seed = config.DATASET_CONFIG['random_seed']
        
        # Erste Aufteilung: Train und Rest (Val + Test)
        train_files, rest_files = train_test_split(
            all_files,
            test_size=(val_split + test_split),
            random_state=random_seed,
            stratify=[label for _, label in all_files]
        )
        
        # Zweite Aufteilung: Val und Test
        val_ratio = val_split / (val_split + test_split)
        val_files, test_files = train_test_split(
            rest_files,
            test_size=1 - val_ratio,
            random_state=random_seed,
            stratify=[label for _, label in rest_files]
        )
        
        print(f"\n{'='*60}")
        print("DATENSATZ AUFTEILUNG:")
        print(f"{'='*60}")
        print(f"Training:   {len(train_files)} Bilder ({len(train_files)/len(all_files)*100:.1f}%)")
        print(f"Validierung: {len(val_files)} Bilder ({len(val_files)/len(all_files)*100:.1f}%)")
        print(f"Test:       {len(test_files)} Bilder ({len(test_files)/len(all_files)*100:.1f}%)")
        print(f"{'='*60}\n")
        
        # Kopiere Dateien
        self._copy_files_to_split(train_files, self.train_dir, "Training")
        self._copy_files_to_split(val_files, self.val_dir, "Validierung")
        self._copy_files_to_split(test_files, self.test_dir, "Test")
    
    def _copy_files_to_split(self, files, target_dir, split_name):
        """Kopiere Dateien in entsprechenden Split-Ordner"""
        print(f"📁 {split_name} wird kopiert...")
        
        # Erstelle Kategorieordner
        for category in config.ALPHABET:
            (target_dir / category).mkdir(parents=True, exist_ok=True)
        
        for src_file, category in tqdm(files, desc=f"  {split_name}"):
            dst_file = target_dir / category / Path(src_file).name
            shutil.copy2(src_file, dst_file)
        
        print(f"✓ {split_name} fertig!\n")
    
    def load_images_from_split(self, split='train'):
        """Lade Bilder aus einem Split"""
        if split == 'train':
            split_dir = self.train_dir
        elif split == 'val':
            split_dir = self.val_dir
        elif split == 'test':
            split_dir = self.test_dir
        else:
            raise ValueError(f"Ungültiger Split: {split}")
        
        images = []
        labels = []
        
        for category in sorted(config.ALPHABET):
            category_path = split_dir / category
            
            if not category_path.exists():
                continue
            img_count: int = 0
            for img_file in category_path.glob('*'):
                print(f"Lade {img_file}...")
                if img_count >= config.DATASET_CONFIG['max_images_per_category']:
                    break
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.resize(img, config.DATASET_CONFIG['image_size'])
                        images.append(img)
                        labels.append(config.ALPHABET_MAPPING[category])
                        img_count += 1
                except Exception as e:
                    print(f"Fehler beim Laden von {img_file}: {e}")
        
        return np.array(images), np.array(labels)
    
    def print_dataset_info(self):
        """Drucke Datensatz-Informationen"""
        print("\n" + "="*60)
        print("DATENSATZ INFORMATIONEN")
        print("="*60)
        
        for split_name, split_dir in [("Training", self.train_dir), 
                                      ("Validierung", self.val_dir), 
                                      ("Test", self.test_dir)]:
            print(f"\n{split_name}:")
            total_images = 0
            
            for category in sorted(config.ALPHABET):
                category_path = split_dir / category
                if category_path.exists():
                    count = len(list(category_path.glob('*')))
                    total_images += count
                    if count > 0:
                        print(f"  {category}: {count} Bilder")
            
            print(f"  Gesamt: {total_images} Bilder")


if __name__ == "__main__":
    manager = DatasetManager()
    manager.load_dataset_from_raw()
    manager.print_dataset_info()
