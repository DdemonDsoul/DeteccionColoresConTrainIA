# predict.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from config import IMG_SIZE, COLORES_CARPETAS, MAPEO_COLORES, buscar_nombre_exacto_hsv

try:
    modelo_color = load_model("modelo/color.keras")
except:
    modelo_color = None

def extraer_tonos_especificos(ruta_imagen):
    """Analiza la matriz completa pixel por pixel asignándole su sub-color exacto sin omitir nada"""
    img = cv2.imread(ruta_imagen)
    if img is None: return ""
    
    # Redimensionamos a un tamaño controlado para mantener la app veloz, pero analizando todo el espectro
    img = cv2.resize(img, (80, 80))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixeles = hsv.reshape((-1, 3))
    total_pixeles = len(pixeles)
    
    # Diccionario temporal para contar cuántos píxeles caen en cada sub-color comercial
    conteo_sub_colores = {}
    
    # Mapeo directo píxel por píxel para evitar el sesgo de promedios de K-Means
    for p in pixeles:
        nombre_exacto = buscar_nombre_exacto_hsv(p[0], p[1], p[2])
        conteo_sub_colores[nombre_exacto] = conteo_sub_colores.get(nombre_exacto, 0) + 1
        
    reporte_especifico = "\n--- SUB-TONOS ESPECÍFICOS DETECTADOS ---\n"
    
    # Ordenar los sub-colores de mayor a menor presencia
    sub_ordenados = sorted(conteo_sub_colores.items(), key=lambda x: x[1], reverse=True)
    
    for nombre, cuenta in sub_ordenados:
        porcentaje = (cuenta / total_pixeles) * 100
        # Filtro microscópico: si representa al menos el 0.1% de la imagen, SE MUESTRA
        if porcentaje > 0.1:
            reporte_especifico += f"• {nombre}: {porcentaje:.1f}%\n"
            
    return reporte_especifico

def calcular_porcentajes_reales(ruta_imagen):
    """Segmentación matricial de alta sensibilidad para el desglose general de píxeles"""
    img = cv2.imread(ruta_imagen)
    if img is None: return {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixeles = img.shape[0] * img.shape[1]
    
    rangos = {
        "red": [((0, 40, 40), (10, 255, 255)), ((165, 40, 40), (180, 255, 255))],
        "orange": [((11, 40, 40), (25, 255, 255))],
        "yellow": [((26, 40, 40), (34, 255, 255))],
        "green": [((35, 35, 35), (85, 255, 255))],
        "Blue": [((90, 40, 40), (130, 255, 255))],
        "Violet": [((131, 35, 35), (165, 255, 255))],
        "Brown": [((10, 35, 20), (20, 255, 140))],  
        "Black": [((0, 0, 0), (180, 255, 85))],      
        "White": [((0, 0, 160), (180, 45, 255))]      
    }
    
    porcentajes = {}
    for color, limites in rangos.items():
        if len(limites) == 2:
            mask1 = cv2.inRange(hsv, limites[0][0], limites[0][1])
            mask2 = cv2.inRange(hsv, limites[1][0], limites[1][1])
            mascara = cv2.bitwise_or(mask1, mask2)
        else:
            mascara = cv2.inRange(hsv, limites[0][0], limites[0][1])
        porcentajes[color] = (cv2.countNonZero(mascara) / total_pixeles) * 100
    return porcentajes

def predecir_color(ruta_imagen):
    if modelo_color is None: return "Modelo no listo.", ""
    
    # 1. Inferencia de la Red Neuronal (Keras)
    img = load_img(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predicciones = modelo_color.predict(img_array)[0]
    confianza_ia = predicciones[np.argmax(predicciones)] * 100
    color_dominante = MAPEO_COLORES[COLORES_CARPETAS[np.argmax(predicciones)]]
    
    # 2. Construcción del reporte integrado completo sin pérdidas
    reporte = f"Tendencia Dominante (IA): {color_dominante} ({confianza_ia:.1f}%)\n"
    
    # Bloque A: Desglose general (Rojo, Azul, Negro, Blanco, etc.)
    reporte += "\n--- DESGLOSE ANALÍTICO DE PÍXELES (% REAL) ---\n"
    desglose_pixeles = calcular_porcentajes_reales(ruta_imagen)
    desglose_ordenado = sorted(desglose_pixeles.items(), key=lambda x: x[1], reverse=True)
    
    for color, porc in desglose_ordenado:
        if porc > 0.01:  # Bajado al 0.01% para capturar hasta el trazo más invisible
            nombre_esp = MAPEO_COLORES.get(color, color)
            reporte += f"• Densidad de {nombre_esp}: {porc:.2f}%\n"
            
    # Bloque B: Sub-tonos comerciales exactos mapeados al 100%
    reporte += extraer_tonos_especificos(ruta_imagen)
    
    return reporte
