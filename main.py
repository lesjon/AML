import numpy as np
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf
import gamedrawer
import jsonToNNInput

if __name__ == '__main__':
    print("Starting project!")
    keep_display_on = True

    dg = gamedrawer.DrawGame()

    NN_input = jsonToNNInput.NNInput("logs/testWriterOutput.json")

    try:
        dg.draw_game_from_json(NN_input.json_data)
    except gamedrawer.TclError:
        print("stopped showing match")
        keep_display_on = False

    print(list(zip(NN_input.data_keys, NN_input.data[0])))

    index = int(0.8 * len(NN_input.data))
    data_for_train = NN_input.data[:index]
    data_for_test = NN_input.data[index:]

    (data_train, category_train) = data_for_train[:-1], data_for_train[1:]
    (data_test, category_test) = data_for_test[:-1],  data_for_test[1:]
    if keep_display_on:
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
