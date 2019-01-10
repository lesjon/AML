import numpy as np
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf
import json


if __name__ == '__main__':
    print("Starting project!")

    read_file = open("logs/log.json", "r")
    data = json.load(read_file)
#    for sample in data:
#        print(sample)
    mnist = tf.keras.datasets.mnist
#    print(mnist.load_data())
#    print(mnist.load_data()[0][0])
#    print(mnist.load_data()[0][0][0])
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(x_train[0])
    print(y_train)
    exit()

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
