from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# --- Inicializar FastAPI ---
app = FastAPI()

# --- Cargar modelo CNN ---
model = tf.keras.models.load_model("modelo_deslizamientos.h5")

# --- Esquema de entrada ---
class CuboEntrada(BaseModel):
    data: list

# --- Ruta principal ---
@app.get("/")
def inicio():
    return {"mensaje": "API para predicción de deslizamientos"}

# --- Ruta de predicción ---
@app.post("/predict")
def predecir(entrada: CuboEntrada):
    cubo = np.array(entrada.data)
    cubo = cubo.reshape(1, 13, 13, 7)  # Asegúrate que coincida con la entrada del modelo
    resultado = model.predict(cubo)
    prediccion = int(resultado[0][0] > 0.5)
    return {"resultado": prediccion}
