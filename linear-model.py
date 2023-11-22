import numpy as np
import os
import tensorflow as tf

# Datos de ejemplo (reemplaza estos datos con tu propio conjunto de datos)
L = np.arange(100, 600, 100)  # Número de líneas de código
C = np.array([3, 5, 4, 6, 7])  # Nivel de complejidad
Errores = np.array([10, 15, 12, 18, 20])  # Número real de errores

train_end = int(0.6 * len(L))
test_start = int(0.8 * len(L))

L_train, C_train, Errores_train = L[:train_end], C[:train_end], Errores[:train_end]
L_test, C_test, Errores_test = L[test_start:], C[test_start:], Errores[test_start:]
L_val, C_val, Errores_val = L[train_end:test_start], C[train_end:test_start], Errores[train_end:test_start]

tf.keras.backend.clear_session()

# Definir el modelo para predecir errores o bugs
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2], name='Single')
])

linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)
print(linear_model.summary())

# Entrenar el modelo para predecir errores o bugs
linear_model.fit(x=np.column_stack((L_train, C_train)), y=Errores_train,
              validation_data=(np.column_stack((L_val, C_val)), Errores_val), epochs=20)

# Predicción de errores o bugs para algunos valores
predictions = linear_model.predict(np.column_stack(([0.0, 200, 310, 420, 520], [3, 5, 4, 6, 7]))).tolist()
print("Predicciones:", predictions)

# Guardar el modelo entrenado
export_path = 'linear-model/1/'
tf.saved_model.save(linear_model, os.path.join('./', export_path))
