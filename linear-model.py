# adsoft 
import numpy as np
import os
import math
import pandas as pd
#import matplotlib.pyplot as plt

# TensorFlow
import tensorflow as tf

def circulo(num_datos=500, R=1, cLat=0, cLon=0 ):
    pi = np.pi
    theta = np.random.uniform(0, 2 * pi, size=num_datos)
    positive = np.abs(R * np.sqrt(np.random.normal(0, 1, size=num_datos)**2))

    x = np.cos(theta)* positive + cLat
    y = np.sin(theta)* positive + cLon

    x = np.round(x, 4)
    y = np.round(y, 4)

    #df = pd.DataFrame({'lat': y, 'lon':x})
    df = np.column_stack([x,y])
    return df


N=150

datos_1 = circulo(num_datos = N, R = 2, cLat=-23.33770680846116, cLon=-58.18941513029103)
datos_2 = circulo(num_datos = N, R = 0.5, cLat=23.69017773392101, cLon= 45.487493833255385)
X = np.concatenate([datos_1,datos_2])
x = np.round(X,3)
y = np.concatenate([np.zeros(800), np.ones(100), np.ones(100)])

train_end = int(0.6 * len(X))
test_start = int(0.8 * len(X))
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([
                                           tf.keras.layers.Dense(units=4, input_shape=[2], activation='relu'
                                           ),
                                           tf.keras.layers.Dense(units=8,activation='relu'
                                           ),
                                           tf.keras.layers.Dense(units=1,activation='sigmoid'
                                           )
                                           ])

linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)
print(linear_model.summary())

linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=27)

all_predictions = linear_model.predict(X).tolist()

gps_points_paraguay = [[-26.17071248262649, -56.189903411278564], [-25.18057529657293, -55.68453231746222], [-24.162281761201704, -58.497032317831426], [-22.649999275426865, -57.00289169263528], [-22.04032102270884, -60.01314559928044]]
gps_points_arabia = [[26.2395947532087, 42.32343133283999], [23.971567725067743, 42.36737664534576], [21.49945384359213, 48.168157896107225], [27.416023726593824, 46.366400083370706], [18.98460587544843, 42.36737664534576]]

predictions_paraguay = linear_model.predict(gps_points_paraguay).tolist()
predictions_arabia = linear_model.predict(gps_points_arabia).tolist()

print("\nPredictions for Paraguay:")
print(predictions_paraguay)

print("\nPredictions for Arabaia:")
print(predictions_arabia)

export_path = 'linear-model/1/'
tf.saved_model.save(linear_model, os.path.join('./',export_path))
