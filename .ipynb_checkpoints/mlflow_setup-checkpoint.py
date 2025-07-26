import mlflow.tensorflow
import tensorflow as tf

mlflow.set_experiment("modelo_deslizamientos")

with mlflow.start_run():
    mlflow.tensorflow.log_model(tf.keras.models.load_model("modelo_deslizamientos.h5"), "modelo_api")
    mlflow.log_param("modelo", "CNN 3 capas")
    mlflow.log_param("input_shape", "(13,13,6)")
    mlflow.log_param("activacion", "sigmoid")
