"""
Vorhersage-Skript für Sign Language Recognition
Klassifiziere einzelne Bilder und Videos
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
from model import SignLanguageModel
import config


class SignLanguagePredictor:
    def __init__(self, model_path=None):
        self.model = SignLanguageModel()
        self.model.load_model(model_path)
    
    def predict_image(self, image_path, show_landmarks=False):
        """Vorhersage für ein einzelnes Bild"""
        print(f"\n{'='*60}")
        print(f"Bild wird analysiert: {image_path}")
        print(f"{'='*60}")
        
        # Lade Bild
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Fehler: Bild konnte nicht geladen werden!")
            return None
        
        # Resize
        image_resized = cv2.resize(image, config.DATASET_CONFIG['image_size'])
        
        # Extrahiere Hand-Landmarks (optional)
        if show_landmarks:
            results = self.model.extract_hand_landmarks(image_resized)
            image_with_landmarks = image_resized.copy()
            image_with_landmarks = self.model.visualize_landmarks(image_with_landmarks, results)
            
            # Zeige Bild mit Landmarks
            cv2.imshow("Hand Landmarks", image_with_landmarks)
            print("✓ Hand-Landmarks erkannt (Fenster zeigt Ergebnis)")
        
        # Vorhersage
        predicted_letter, confidence = self.model.predict(image_resized)
        
        print(f"\n📝 Erkannter Buchstabe: {predicted_letter}")
        print(f"🎯 Konfidenz: {confidence*100:.2f}%")
        
        if confidence < 0.5:
            print("⚠️  Niedrige Konfidenz - Ergebnis könnte ungenau sein!")
        
        return {
            'letter': predicted_letter,
            'confidence': float(confidence),
            'image_path': str(image_path)
        }
    
    def predict_batch(self, image_dir):
        """Vorhersage für alle Bilder in einem Verzeichnis"""
        print(f"\n{'='*60}")
        print(f"Batch-Vorhersage: {image_dir}")
        print(f"{'='*60}")
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"❌ Verzeichnis existiert nicht: {image_dir}")
            return None
        
        results = []
        
        # Finde alle Bilder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(image_dir.glob(f'**/{ext}'))
        
        print(f"✓ {len(image_files)} Bilder gefunden")
        
        if len(image_files) == 0:
            print("❌ Keine Bilder gefunden!")
            return None
        
        # Lade und verarbeite Bilder
        for img_path in sorted(image_files):
            try:
                image = cv2.imread(str(img_path))
                if image is not None:
                    image_resized = cv2.resize(image, config.DATASET_CONFIG['image_size'])
                    predicted_letter, confidence = self.model.predict(image_resized)
                    
                    results.append({
                        'image': img_path.name,
                        'letter': predicted_letter,
                        'confidence': float(confidence)
                    })
                    
                    print(f"  {img_path.name}: {predicted_letter} ({confidence*100:.1f}%)")
            except Exception as e:
                print(f"  ❌ Fehler bei {img_path.name}: {e}")
        
        # Statistik
        if results:
            print(f"\n{'='*60}")
            print("BATCH-ERGEBNISSE:")
            print(f"{'='*60}")
            print(f"Verarbeitete Bilder: {len(results)}")
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"Durchschnittliche Konfidenz: {avg_confidence*100:.2f}%")
        
        return results
    
    def predict_video(self, video_path=None, show_fps=True):
        """Vorhersage für Video oder Webcam"""
        print(f"\n{'='*60}")
        if video_path is None:
            print("📹 Webcam wird aktiviert...")
            cap = cv2.VideoCapture(0)
        else:
            print(f"📹 Video wird analysiert: {video_path}")
            cap = cv2.VideoCapture(str(video_path))
        print(f"{'='*60}")
        print("Drücke 'q' um zu beenden, 's' um Foto zu speichern")
        print(f"{'='*60}\n")
        
        if not cap.isOpened():
            print("❌ Kamera/Video konnte nicht geöffnet werden!")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize für schnellere Verarbeitung
            frame_resized = cv2.resize(frame, config.DATASET_CONFIG['image_size'])
            
            # Extrahiere Landmarks
            results = self.model.extract_hand_landmarks(frame_resized)
            frame_with_landmarks = frame_resized.copy()
            frame_with_landmarks = self.model.visualize_landmarks(frame_with_landmarks, results)
            
            # Vorhersage
            if results.multi_hand_landmarks:
                predicted_letter, confidence = self.model.predict(frame_resized)
                
                # Zeige Text auf Frame
                cv2.putText(frame_with_landmarks, f"Buchstabe: {predicted_letter}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_with_landmarks, f"Konfidenz: {confidence*100:.1f}%", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # FPS anzeigen
            if show_fps:
                frame_count += 1
                cv2.putText(frame_with_landmarks, f"Frames: {frame_count}", 
                           (10, frame_with_landmarks.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Zeige Frame
            cv2.imshow("Sign Language Recognition", frame_with_landmarks)
            
            # Tastatur-Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                output_path = config.OUTPUT_DIR / f"screenshot_{frame_count}.jpg"
                cv2.imwrite(str(output_path), frame_with_landmarks)
                print(f"✓ Screenshot gespeichert: {output_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Video-Verarbeitung beendet!")


def main():
    parser = argparse.ArgumentParser(
        description="Sign Language Recognition - Vorhersage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Einzelnes Bild klassifizieren
  python predict.py --image data/test/A/image.jpg
  
  # Mit Hand-Landmarks anzeigen
  python predict.py --image data/test/A/image.jpg --landmarks
  
  # Alle Bilder in einem Verzeichnis
  python predict.py --batch data/test
  
  # Live-Webcam
  python predict.py --webcam
  
  # Video-Datei
  python predict.py --video video.mp4
        """
    )
    
    parser.add_argument('--image', type=str, help='Pfad zu einem einzelnen Bild')
    parser.add_argument('--batch', type=str, help='Pfad zu Verzeichnis mit Bildern')
    parser.add_argument('--webcam', action='store_true', help='Verwende Webcam')
    parser.add_argument('--video', type=str, help='Pfad zu Video-Datei')
    parser.add_argument('--landmarks', action='store_true', help='Zeige Hand-Landmarks')
    parser.add_argument('--model', type=str, help='Pfad zu trainiertem Modell')
    
    args = parser.parse_args()
    
    # Initialisiere Predictor
    predictor = SignLanguagePredictor(args.model)
    
    # Verarbeite basierend auf Input
    if args.image:
        predictor.predict_image(args.image, show_landmarks=args.landmarks)
    elif args.batch:
        predictor.predict_batch(args.batch)
    elif args.webcam:
        predictor.predict_video()
    elif args.video:
        predictor.predict_video(args.video)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
