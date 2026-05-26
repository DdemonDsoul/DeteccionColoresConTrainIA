# config.py
import numpy as np

IMG_SIZE = 128
BATCH_SIZE = 16  
EPOCHS_RETRAIN = 5

# 1. Las carpetas base que procesa la red neuronal (Pilares cognitivos)
MAPEO_COLORES = {
    "Black": "Negro", "Blue": "Azul", "Brown": "Marrón / Café",
    "green": "Verde", "orange": "Naranja", "red": "Rojo",
    "Violet": "Violeta / Morado", "White": "Blanco", "yellow": "Amarillo"
}
COLORES_CARPETAS = sorted(list(MAPEO_COLORES.keys()))

# 2. DICCIONARIO MASIVO DE SUB-COLORES (Aquí puedes meter miles si quieres)
# Formato: "Nombre Comercial/Exacto": (H, S, V)
DICCIONARIO_MAESTRO_HSV = {
    "Rojo Carmesí": (175, 200, 150),
    "Rojo Ladrillo": (5, 180, 130),
    "Azul Celeste": (105, 100, 220),
    "Azul Marino": (115, 255, 100),
    "Azul Turquesa / Cerceta": (95, 180, 180),
    "Verde Menta": (70, 80, 200),
    "Verde Olivo": (45, 120, 100),
    "Verde Esmeralda": (65, 230, 130),
    "Amarillo Mostaza": (23, 160, 160),
    "Naranja Coral": (13, 170, 200),
    "Rosa Pastel": (170, 50, 240),
    "Fucsia / Magenta": (150, 200, 220),
    "Gris Plata": (0, 10, 190),
    "Gris Carbón": (0, 0, 70),
    "Café Chocolate": (15, 150, 60),
    "Crema / Beige": (20, 40, 240),
    "Morado Purpúra": (140, 200, 120)
}

def buscar_nombre_exacto_hsv(h, s, v):
    """Busca el color más cercano matemáticamente dentro de los miles disponibles"""
    color_mas_cercano = "Tono Desconocido"
    distancia_minima = float('inf')
    
    for nombre, valores in DICCIONARIO_MAESTRO_HSV.items():
        # Distancia euclidiana en el espacio HSV
        distancia = np.sqrt((h - valores[0])**2 + (s - valores[1])**2 + (v - valores[2])**2)
        if distancia < distancia_minima:
            distancia_minima = distancia
            color_mas_cercano = nombre
            
    return color_mas_cercano
