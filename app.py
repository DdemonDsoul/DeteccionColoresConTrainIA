# app.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import shutil
import threading  # Para entrenar en segundo plano sin congelar la app
import numpy as np

from predict import predecir_color
from config import COLORES_CARPETAS, MAPEO_COLORES
from train_color import entrenar_sistema  # Importamos tu función de entrenamiento

class AppAnalizadorCromatico:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal Color Analytics System")
        self.root.geometry("620x760")  
        self.root.configure(bg="#2c3e50")
        
        self.ruta_imagen_actual = None
        self.entrenando = False  # Bandera de control
        
        # Título principal de la interfaz
        lbl_titulo = tk.Label(
            root, 
            text="SISTEMA UNIVERSAL DE ANÁLISIS CROMÁTICO", 
            font=("Helvetica", 14, "bold"), 
            fg="#ecf0f1", 
            bg="#2c3e50"
        )
        lbl_titulo.pack(pady=10)
        
        # CONTENEDOR DE IMAGEN (Tamaño fijo en PÍXELES)
        frame_imagen = tk.Frame(root, width=340, height=260, bg="#34495e", relief="sunken", bd=2)
        frame_imagen.pack(pady=10)
        frame_imagen.pack_propagate(False) 
        
        self.canvas_img = tk.Label(
            frame_imagen, 
            text="Cargue una imagen para iniciar el escaneo...", 
            bg="#34495e", 
            fg="#bdc3c7",
            font=("Helvetica", 11, "italic")
        )
        self.canvas_img.pack(fill="both", expand=True) 
        
        # CONTENEDOR DE TEXTO (Estilo consola para reportes)
        self.result_var = tk.StringVar(value="Estado: Esperando archivo de imagen...")
        self.lbl_resultado = tk.Label(
            root, 
            textvariable=self.result_var, 
            font=("Courier", 10, "bold"), 
            fg="#2ecc71", 
            bg="#1a252f",    
            justify="left",    
            anchor="w",
            padx=15,
            pady=10,
            relief="groove"
        )
        self.lbl_resultado.pack(pady=10, fill="x", padx=40)
        
        # INDICADOR GRÁFICO DE REENTRENAMIENTO (Barra de carga infinita)
        self.frame_progreso = tk.Frame(root, bg="#2c3e50")
        self.frame_progreso.pack(pady=5, fill="x", padx=40)
        
        self.lbl_estado_ia = tk.Label(
            self.frame_progreso, 
            text="Núcleo de IA: Operativo y Listo", 
            font=("Helvetica", 9, "italic"), 
            fg="#abebc6", 
            bg="#2c3e50"
        )
        self.lbl_estado_ia.pack()
        
        self.progress_bar = ttk.Progressbar(self.frame_progreso, mode="indeterminate")
        
        # PANEL DE BOTONES
        btn_panel = tk.Frame(root, bg="#2c3e50")
        btn_panel.pack(pady=15)
        
        self.btn_cargar = tk.Button(
            btn_panel, 
            text="Cargar Imagen 📷", 
            font=("Helvetica", 11, "bold"), 
            bg="#3498db", 
            fg="white", 
            width=20, 
            command=self.cargar_imagen
        )
        self.btn_cargar.grid(row=0, column=0, padx=10)
        
        self.btn_corregir = tk.Button(
            btn_panel, 
            text="Corregir Modelo ⚠️", 
            font=("Helvetica", 11, "bold"), 
            bg="#e74c3c", 
            fg="white", 
            width=20, 
            command=self.abrir_ventana_correccion
        )
        self.btn_corregir.grid(row=0, column=1, padx=10)
        
    def cargar_imagen(self):
        if self.entrenando:
            messagebox.showwarning("IA Ocupada", "Por favor espera a que termine el reentrenamiento automático.")
            return
            
        ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
        if not ruta:
            return
            
        self.ruta_imagen_actual = ruta
        
        img = Image.open(ruta)
        img = img.resize((320, 240), Image.Resampling.LANCZOS) 
        img_tk = ImageTk.PhotoImage(img)
        
        self.canvas_img.configure(image=img_tk, text="")
        self.canvas_img.image = img_tk
        
        reporte_completo = predecir_color(ruta)
        self.result_var.set(reporte_completo)

    def abrir_ventana_correccion(self):
        if self.entrenando:
            messagebox.showwarning("IA Ocupada", "El sistema se está optimizando en este momento.")
            return
        if not self.ruta_imagen_actual:
            messagebox.showwarning("Advertencia", "No hay ninguna imagen cargada para clasificar.")
            return
            
        ventana_pop = tk.Toplevel(self.root)
        ventana_pop.title("Retroalimentación del Sistema")
        ventana_pop.geometry("320x280")
        ventana_pop.configure(bg="#34495e")
        ventana_pop.grab_set() 
        
        tk.Label(
            ventana_pop, 
            text="Selecciona la clasificación real:", 
            font=("Helvetica", 10, "bold"), 
            fg="white", 
            bg="#34495e"
        ).pack(pady=12)
        
        seleccion = tk.StringVar(value=COLORES_CARPETAS[0])
        
        for clave in COLORES_CARPETAS:
            tk.Radiobutton(
                ventana_pop, 
                text=MAPEO_COLORES[clave], 
                variable=seleccion, 
                value=clave, 
                fg="white", 
                bg="#34495e", 
                selectcolor="#2c3e50", 
                font=("Helvetica", 9)
            ).pack(anchor="w", padx=40, background="#34495e")
            
        def guardar_y_reentrenar():
            color_real = seleccion.get()
            carpeta_destino = os.path.join("correcciones", "colores", color_real)
            os.makedirs(carpeta_destino, exist_ok=True)
            
            nombre_archivo = os.path.basename(self.ruta_imagen_actual)
            shutil.copy(self.ruta_imagen_actual, os.path.join(carpeta_destino, nombre_archivo))
            
            ventana_pop.destroy()
            
            # Disparar el reentrenamiento automático en segundo plano usando un Hilo
            self.ejecutar_reentrenamiento_async()
            
        tk.Button(
            ventana_pop, 
            text="Confirmar y Optimizar IA", 
            font=("Helvetica", 10, "bold"), 
            bg="#2ecc71", 
            fg="white", 
            command=guardar_y_reentrenar
        ).pack(pady=15)

    def ejecutar_reentrenamiento_async(self):
        """Prepara la interfaz e inicia el entrenamiento en un hilo separado"""
        self.entrenando = True
        self.lbl_estado_ia.configure(text="⚠️ REENTRENANDO MODELO EN SEGUNDO PLANO... POR FAVOR ESPERE", fg="#f1c40f")
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.start(10)
        
        # Desactivar botones temporalmente por seguridad estructural
        self.btn_cargar.configure(state="disabled")
        self.btn_corregir.configure(state="disabled")
        
        # Crear y arrancar el hilo
        hilo_entrenamiento = threading.Thread(target=self._hilo_entrenamiento_worker)
        hilo_entrenamiento.start()

    def _hilo_entrenamiento_worker(self):
        """Código que ejecuta el proceso pesado sin trabar las ventanas gráficas"""
        try:
            entrenar_sistema() # Llama directamente la función de tu train_color.py
            exitosa = True
        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            exitosa = False
            
        # Regresar al hilo principal para actualizar los elementos visuales de la App
        self.root.after(0, self._finalizar_reentrenamiento_ui, exitosa)

    def _finalizar_reentrenamiento_ui(self, resultado_exitoso):
        """Devuelve la interfaz a su estado normal tras concluir las épocas de Keras"""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.entrenando = False
        
        self.btn_cargar.configure(state="normal")
        self.btn_corregir.configure(state="normal")
        
        if resultado_exitoso:
            self.lbl_estado_ia.configure(text="✅ Optimización Concluida. Modelo .keras Actualizado.", fg="#2ecc71")
            messagebox.showinfo("Sistema Optimizado", "La red neuronal ha absorbido la muestra y se ha reentrenado con éxito automáticamente.")
            
            # Recargar el análisis de la imagen actual con los nuevos pesos de la IA
            if self.ruta_imagen_actual:
                # Forzar recarga del modelo en predict importando dinámicamente o corriendo la predicción de nuevo
                import predict
                from importlib import reload
                reload(predict) # Forzamos a predict.py a cargar el nuevo archivo de Keras del disco
                self.result_var.set(predict.predecir_color(self.ruta_imagen_actual))
        else:
            self.lbl_estado_ia.configure(text="❌ Error crítico durante el proceso de optimización.", fg="#e74c3c")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppAnalizadorCromatico(root)
    root.mainloop()
