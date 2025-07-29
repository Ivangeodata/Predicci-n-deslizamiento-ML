from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import tensorflow as tf

# --- Inicializar FastAPI ---
app = FastAPI(title="API CNN Deslizamientos", version="1.1")

# --- Cargar modelo CNN ---
try:
    model = tf.keras.models.load_model("modelo_deslizamientos.h5")
except Exception as e:
    raise RuntimeError(f"❌ Error al cargar el modelo: {e}")

# --- Esquema de entrada validado ---
class CuboEntrada(BaseModel):
    data: List[List[List[float]]] = Field(..., description="Arreglo 13x13x7")

# --- Ruta principal ---
@app.get("/")
def inicio():
    return {"mensaje": "✅ API lista para recibir cubos 13x13x7 y predecir deslizamientos."}

# --- Ruta de predicción ---
@app.post("/predict")
def predecir(entrada: CuboEntrada):
    try:
        cubo = np.array(entrada.data, dtype=np.float32)

        # Validación de forma
        if cubo.shape != (13, 13, 7):
            raise HTTPException(status_code=422, detail=f"❌ Cubo debe tener forma (13, 13, 7). Recibido: {cubo.shape}")

        # Validación de valores válidos (no NaN ni inf)
        if not np.isfinite(cubo).all():
            raise HTTPException(status_code=422, detail="❌ El cubo contiene valores NaN o Inf.")

        # Redimensionar para predicción batch
        cubo = cubo.reshape(1, 13, 13, 7)

        # Predicción
        resultado = model.predict(cubo, verbose=0)
        probabilidad = float(resultado[0][0])
        prediccion = int(probabilidad > 0.5)

        return {
            "prediccion": prediccion,
            "probabilidad": round(probabilidad, 4),
            "clase": "Deslizamiento" if prediccion == 1 else "No deslizamiento"
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error interno en la predicción: {str(e)}")
