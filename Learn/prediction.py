import tensorflow as tf
# Framework by Google
import numpy as np
# Loading the variables into a single unit
from tensorflow import keras


def house_model(y_new):
    xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    ys = np.array([1.000, 1.500, 2.000, 2.500, 3.000, 4.000], dtype=float)
    # Loading the data into xs and ys
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    # Defining the model
    model.compile(opmtimizer='sgd', loss="mean_squared_error")
    model.fit(xs, ys, epochs=10000)
    return model.predict(y_new)[0]
    # Returning the value that is predicted


prediction = house_model([7])
print(prediction)
