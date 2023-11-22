
import numpy as np
import os
#import matplotlib.pyplot as plt

# TensorFlow for calculate the weight 
import tensorflow as tf
 
print(tf.__version__)

X = np.arange(1, 11, 1)  
np.random.shuffle(X)
y = X * 8.91  

train_end = int(0.6 * len(X))
test_start = int(0.8 * len(X))
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
])
linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)
print(linear_model.summary())

linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
print(linear_model.predict([[1.0], [2.0], [3.0], [4.0], [5.0]]).tolist())   

export_path = 'linear-model/1/'
tf.saved_model.save(linear_model, os.path.join('./', export_path))