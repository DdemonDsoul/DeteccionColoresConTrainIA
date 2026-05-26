# train_color.py
import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE, BATCH_SIZE, EPOCHS_RETRAIN, COLORES_CARPETAS

DATASET_PATH = os.path.join("datasets", "colores")
CORRECCIONES_PATH = os.path.join("correcciones", "colores")

def consolidar_correcciones():
    """Mueve de forma inteligente las correcciones hechas por el usuario al dataset principal antes de entrenar"""
    if not os.path.exists(CORRECCIONES_PATH):
        return
        
    print("📦 Consolidando nuevas muestras del repositorio de correcciones...")
    for carpeta in os.listdir(CORRECCIONES_PATH):
        origen = os.path.join(CORRECCIONES_PATH, carpeta)
        destino = os.path.join(DATASET_PATH, carpeta)
        
        if os.path.isdir(origen):
            os.makedirs(destino, exist_ok=True)
            for archivo in os.listdir(origen):
                shutil.move(os.path.join(origen, archivo), os.path.join(destino, archivo))
            try:
                os.rmdir(origen)
            except:
                pass
    print("✅ Dataset actualizado y unificado.")

def entrenar_sistema():
    consolidar_correcciones()
    
    # Generador de imágenes con aumento de datos (Data Augmentation) para evitar fallos por luz
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_gen = datagen.flow_from_directory(
        DATASET_PATH, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
    )
    
    num_clases = len(COLORES_CARPETAS)
    
    # Si ya existe un modelo anterior, lo cargamos para no perder su conocimiento (Fine-Tuning)
    if os.path.exists("modelo/color.keras"):
        print("🧠 Cargando modelo previo para actualización incremental...")
        model = load_model("modelo/color.keras")
    else:
        print("🏗️ Creando nueva arquitectura de Red Convolucional...")
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_clases, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
    print("🚀 Iniciando optimización del modelo con nuevas variantes cromáticas...")
    model.fit(train_gen, epochs=EPOCHS_RETRAIN)
    
    os.makedirs("modelo", exist_ok=True)
    model.save("modelo/color.keras")
    print("🎉 ¡Modelo actualizado de forma exitosa! Listo para producción en app.py.")

if __name__ == "__main__":
    entrenar_sistema()
