import numpy as np
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from gamedrawer import DrawGame


def json2input(path):
    def dict2lists(dict_to_split):
        k, v = list(dict_to_split.keys()), list(dict_to_split.values())
        # print(list(zip(k, v)))
        return k, v

    read_file = open(path, "r")
    json_file = json.load(read_file)
    return_list = []
    keys, robots_yellow_keys, robots_blue_keys, balls_keys = [], [], [], []

    for json_object in json_file:
        keys, values = dict2lists(json_object)

        robots_yellow_keys, robots_yellow_values, robots_blue_keys, robots_blue_values, balls_keys = [], [], [], [], []

        for robot in values[keys.index("robots_yellow")]:
            k, v = dict2lists(robot)
            robots_yellow_keys.extend(k[1:])
            robots_yellow_values.extend(v[1:])

        for robot in values[keys.index("robots_blue")]:
            k, v = dict2lists(robot)
            robots_blue_keys.extend(k[1:])
            robots_blue_values.extend(v[1:])

        balls_keys, balls_values = dict2lists(values[keys.index("balls")][0])

        return_list.append(robots_yellow_values + robots_blue_values + balls_values)

    return return_list, robots_yellow_keys + robots_blue_keys + balls_keys


if __name__ == '__main__':
    print("Starting project!")
    dg = DrawGame()

    read_file = open("logs/testWriterOutput.json", "r")
    json_file = json.load(read_file)

    dg.draw_game_from_json(json_file)

    print("load data from json")
    data, data_keys = json2input("logs/testWriterOutput.json")

    print(list(zip(data_keys, data[0])))

    index = int(0.8*len(data))
    data_for_train = data[:index]
    data_for_test = data[index:]

    (data_train, category_train) = data_for_train[:-1], data_for_train[1:]
    (data_test, category_test) = data_for_test[:-1],  data_for_test[1:]
    dg.wait_till_close()
    exit()
    print("create model")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(len(data_train[0]), activation=tf.keras.activations.linear),  # tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(category_train[0]), activation=tf.keras.activations.linear)
    ])

    print(model)
    print("compile model")
    model.compile(optimizer='adam',
                  loss=tf.losses.mean_squared_error,
                  metrics=['accuracy'])

    model.fit(data_train, category_train, epochs=5, verbose=2)
    model.evaluate(data_test, category_test)

    exit()

    mnist = tf.keras.datasets.mnist
#    print(mnist.load_data())
#    print(mnist.load_data()[0][0])
#    print(mnist.load_data()[0][0][0])
    (data_train, category_train), (data_test, category_test) = mnist.load_data()

    data_train, data_test = data_train / 255.0, data_test / 255.0

    print(data_train[0])
    print(category_train)
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

    model.fit(data_train, category_train, epochs=5)
    model.evaluate(data_test, category_test)
