"""
Modell: TensorFlow mit MediaPipe Integration für Sign Language Recognition
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import mediapipe as mp
import cv2
import config


class SignLanguageModel:
    def __init__(self, model_name="sign_language_model"):
        self.model_name = model_name
        self.model = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=config.MEDIAPIPE_CONFIG['static_image_mode'],
            max_num_hands=config.MEDIAPIPE_CONFIG['max_num_hands'],
            model_complexity=config.MEDIAPIPE_CONFIG['model_complexity'],
            min_detection_confidence=config.MEDIAPIPE_CONFIG['min_detection_confidence'],
            min_tracking_confidence=config.MEDIAPIPE_CONFIG['min_tracking_confidence']
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def build_model(self):
        """Baue ein tiefes CNN-Modell"""
        print("\n" + "="*60)
        print("MODELL WIRD GEBAUT")
        print("="*60)
        
        model = models.Sequential([
            # Block 1
            layers.Input(shape=config.MODEL_CONFIG['input_shape']),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Fully Connected Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config.MODEL_CONFIG['dropout_rate']),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config.MODEL_CONFIG['dropout_rate']),
            
            # Output Layer
            layers.Dense(config.MODEL_CONFIG['num_classes'], activation='softmax')
        ])
        
        self.model = model
        
        # Kompiliere Modell
        optimizer = keras.optimizers.Adam(learning_rate=config.MODEL_CONFIG['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        print("\n✓ Modell gebaut!")
        print(f"✓ Parameter: {self.model.count_params():,}")
        self.model.summary()
        
        return self.model
    
    def save_model(self, path=None):
        """Speichere das trainierte Modell"""
        if path is None:
            path = config.MODELS_DIR / f"{self.model_name}.h5"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"\n✓ Modell gespeichert: {path}")
    
    def load_model(self, path=None):
        """Lade ein gespeichertes Modell"""
        if path is None:
            path = config.MODELS_DIR / f"{self.model_name}.h5"
        
        self.model = keras.models.load_model(str(path))
        print(f"\n✓ Modell geladen: {path}")
        return self.model
    
    def extract_hand_landmarks(self, image):
        """Extrahiere Hand-Landmarks mit MediaPipe"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        return results
    
    def visualize_landmarks(self, image, results):
        """Visualisiere Hand-Landmarks auf dem Bild"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return image
    
    def predict(self, image):
        """Vorhersage für ein einzelnes Bild"""
        if self.model is None:
            raise ValueError("Modell nicht geladen!")
        
        # Normalisiere Bild
        img_array = image.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Vorhersage
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return config.REVERSE_ALPHABET_MAPPING[predicted_class], confidence
    
    def predict_batch(self, images):
        """Vorhersage für einen Batch von Bildern"""
        if self.model is None:
            raise ValueError("Modell nicht geladen!")
        
        # Normalisiere Bilder
        images_array = images.astype('float32') / 255.0
        
        # Vorhersagen
        predictions = self.model.predict(images_array, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        results = [
            (config.REVERSE_ALPHABET_MAPPING[pred_class], conf)
            for pred_class, conf in zip(predicted_classes, confidences)
        ]
        
        return results


if __name__ == "__main__":
    model = SignLanguageModel()
    model.build_model()
    model.save_model()
