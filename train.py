"""
Training-Skript für Sign Language Recognition Modell
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataset_manager import DatasetManager
from model import SignLanguageModel
import config
from pathlib import Path
import json


class Trainer:
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.model_wrapper = SignLanguageModel()
        self.history = None
        self.callbacks = []
    
    def prepare_data(self):
        """Bereite Trainingsdaten vor"""
        print("\n" + "="*60)
        print("DATEN WERDEN VORBEREITET")
        print("="*60)
        
        # Lade Datensatz
        X_train, y_train = self.dataset_manager.load_images_from_split('train')
        X_val, y_val = self.dataset_manager.load_images_from_split('val')
        
        print(f"\n✓ Training Daten: {X_train.shape}")
        print(f"✓ Validierungs Daten: {X_val.shape}")
        
        # Normalisiere
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        
        # One-Hot Encoding
        y_train = keras.utils.to_categorical(y_train, config.MODEL_CONFIG['num_classes'])
        y_val = keras.utils.to_categorical(y_val, config.MODEL_CONFIG['num_classes'])
        
        return X_train, y_train, X_val, y_val
    
    def setup_callbacks(self):
        """Richte Callbacks für Training auf"""
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        self.callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.TRAINING_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.TRAINING_CONFIG['reduce_lr_factor'],
                patience=config.TRAINING_CONFIG['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(config.LOGS_DIR),
                histogram_freq=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(config.MODELS_DIR / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
        ]
    
    def train(self, X_train, y_train, X_val, y_val):
        """Trainiere das Modell"""
        print("\n" + "="*60)
        print("TRAINING STARTET")
        print("="*60)
        
        self.model_wrapper.build_model()
        self.setup_callbacks()
        
        self.history = self.model_wrapper.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.TRAINING_CONFIG['epochs'],
            batch_size=config.TRAINING_CONFIG['batch_size'],
            callbacks=self.callbacks,
            verbose=1
        )
        
        print("\n✓ Training abgeschlossen!")
        
        # Speichere das Modell
        self.model_wrapper.save_model()
        
        return self.history
    
    def evaluate(self):
        """Bewerte das Modell auf Test-Datensatz"""
        print("\n" + "="*60)
        print("MODELL WIRD EVALUIERT")
        print("="*60)
        
        X_test, y_test = self.dataset_manager.load_images_from_split('test')
        
        if len(X_test) == 0:
            print("❌ Keine Test-Daten gefunden!")
            return None
        
        X_test = X_test.astype('float32') / 255.0
        y_test = keras.utils.to_categorical(y_test, config.MODEL_CONFIG['num_classes'])
        
        results = self.model_wrapper.model.evaluate(X_test, y_test, verbose=1)
        
        print(f"\n✓ Test Loss: {results[0]:.4f}")
        print(f"✓ Test Accuracy: {results[1]:.4f}")
        print(f"✓ Top-5 Accuracy: {results[2]:.4f}")
        
        return results
    
    def save_history(self, filename="training_history.json"):
        """Speichere Trainings-History"""
        if self.history is None:
            print("Keine Trainings-History vorhanden!")
            return
        
        output_dir = config.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        history_dict = {
            'loss': [float(x) for x in self.history.history['loss']],
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
        }
        
        with open(output_dir / filename, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"\n✓ Trainings-History gespeichert: {output_dir / filename}")


def main():
    """Hauptfunktion"""
    print("\n" + "="*80)
    print("SIGN LANGUAGE RECOGNITION - TRAINING")
    print("="*80)
    
    trainer = Trainer()
    
    # 1. Datensatz vorbereiten
    X_train, y_train, X_val, y_val = trainer.prepare_data()
    
    # 2. Datensatz-Informationen anzeigen
    trainer.dataset_manager.print_dataset_info()
    
    # 3. Training starten
    trainer.train(X_train, y_train, X_val, y_val)
    
    # 4. Speichere Trainings-History
    trainer.save_history()
    
    # 5. Evaluierung auf Test-Datensatz
    trainer.evaluate()
    
    print("\n" + "="*80)
    print("✓ TRAINING FERTIG!")
    print("="*80)
    print(f"\n📁 Modell gespeichert: {config.MODELS_DIR}")
    print(f"📊 Logs: {config.LOGS_DIR}")
    print(f"📈 History: {config.OUTPUT_DIR}")
    print(f"\nVerwende das Modell mit: python predict.py --image <path>")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
