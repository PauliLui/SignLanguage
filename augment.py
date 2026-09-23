"""
Data Augmentation - Erweitere Trainingsdaten mit Transformationen
Nützlich wenn wenig Trainingsbilder vorhanden sind
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import config


class DataAugmenter:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or config.RAW_DATA_DIR / "augmented"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def rotate(self, image, angle):
        """Drehe Bild"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
    
    def flip(self, image, direction='horizontal'):
        """Spiegle Bild"""
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        else:
            return cv2.flip(image, 0)
    
    def scale(self, image, scale_factor):
        """Skaliere Bild"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled = cv2.resize(image, (new_w, new_h))
        return scaled
    
    def translate(self, image, tx, ty):
        """Verschiebe Bild"""
        h, w = image.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, M, (w, h))
        return translated
    
    def brightness(self, image, factor):
        """Ändere Helligkeit"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], factor)
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return brightened
    
    def contrast(self, image, factor):
        """Ändere Kontrast"""
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted
    
    def add_noise(self, image, noise_level=0.1):
        """Füge Rauschen hinzu"""
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy = image + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    def blur(self, image, kernel_size=5):
        """Unschärfe"""
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    def augment_image(self, image_path, num_augmentations=5):
        """Erstelle mehrere augmentierte Versionen eines Bildes"""
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        augmented_images = []
        
        # Original
        augmented_images.append(image)
        
        # Rotationen
        for angle in [-15, 15]:
            augmented_images.append(self.rotate(image, angle))
        
        # Spiegelung
        augmented_images.append(self.flip(image, 'horizontal'))
        
        # Skalierungen
        for scale in [0.8, 1.2]:
            scaled = self.scale(image, scale)
            h, w = image.shape[:2]
            if scaled.shape[0] < h or scaled.shape[1] < w:
                # Wenn kleiner: pad
                padded = np.zeros_like(image)
                padded[:scaled.shape[0], :scaled.shape[1]] = scaled
                augmented_images.append(padded)
            else:
                # Wenn größer: crop
                augmented_images.append(scaled[:h, :w])
        
        # Verschiebung
        augmented_images.append(self.translate(image, 10, 10))
        
        # Helligkeit
        augmented_images.append(self.brightness(image, 0.8))
        augmented_images.append(self.brightness(image, 1.2))
        
        # Rauschen
        augmented_images.append(self.add_noise(image, noise_level=0.05))
        
        return augmented_images[:num_augmentations]
    
    def augment_dataset(self, source_dir=None, multiplier=5):
        """Augmentiere alle Bilder im Datensatz"""
        if source_dir is None:
            source_dir = config.RAW_DATA_DIR
        
        source_dir = Path(source_dir)
        
        print("\n" + "="*60)
        print("DATA AUGMENTATION STARTET")
        print("="*60)
        
        categories = sorted([d for d in source_dir.iterdir() if d.is_dir()])
        
        for category in categories:
            category_name = category.name
            output_category = self.output_dir / category_name
            output_category.mkdir(parents=True, exist_ok=True)
            
            images = list(category.glob('*'))
            print(f"\n{category_name}: {len(images)} Bilder")
            
            for idx, img_path in enumerate(tqdm(images)):
                # Speichere Original
                output_path = output_category / f"{idx}_original_{img_path.name}"
                cv2.imwrite(str(output_path), cv2.imread(str(img_path)))
                
                # Erstelle augmentierte Versionen
                augmented_images = self.augment_image(img_path, multiplier)
                
                for aug_idx, aug_img in enumerate(augmented_images[1:], 1):
                    output_path = output_category / f"{idx}_aug{aug_idx}_{img_path.stem}.jpg"
                    cv2.imwrite(str(output_path), aug_img)
        
        print(f"\n✓ Augmentierung abgeschlossen!")
        print(f"✓ Ausgabe: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Data Augmentation für Sign Language Datensatz")
    parser.add_argument('--source', type=str, default=str(config.RAW_DATA_DIR),
                       help='Quelle-Verzeichnis mit Originalbildern')
    parser.add_argument('--output', type=str, help='Ausgabe-Verzeichnis')
    parser.add_argument('--multiplier', type=int, default=5,
                       help='Anzahl der Augmentierungen pro Bild')
    
    args = parser.parse_args()
    
    augmenter = DataAugmenter(args.output)
    augmenter.augment_dataset(args.source, args.multiplier)


if __name__ == "__main__":
    main()
