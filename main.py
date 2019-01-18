import gamedrawer
import jsonGameProcessor
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    print("Starting project!")
    keep_display_on = False
    play_whole_match = True

    dg = gamedrawer.GameDrawer()

    # NN_input = jsonGameProcessor.JsonToArray("logs/testWriterOutput.json")

    NN_input = jsonGameProcessor.JsonToArray("logs/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.log")
    frame_data_size = len(NN_input.data[0])
    print(frame_data_size)
    # if play_whole_match:
    #     try:
    #         dg.draw_game_from_json(NN_input.json_data)
    #     except gamedrawer.TclError:
    #         print("stopped showing match")
    #         keep_display_on = False


    if play_whole_match:
        for frame in NN_input.data:
            dg.draw_json(NN_input.data_frame_to_dict(frame[0:50]))
            dg.clear_canvas()

    print("create model")
    # Example of one output for whole sequence

    # define model where LSTM is also output layer
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(1, return_sequences=True, activation=None, input_shape=(frame_data_size, 1)))
    model.add(tf.keras.layers.LSTM(1, return_sequences=True, activation=None))
    model.add(tf.keras.layers.LSTM(1, return_sequences=True, activation=None))
    model.compile(optimizer='adam', loss='mse')
    # input time steps
    np_data = np.array(NN_input.data)
    print("np_data shape",np_data.shape)
    index = int(0.8 * np_data.shape[0])
    print("index",index)
    data_for_train = np_data[:index].reshape((-1, frame_data_size, 1))
    data_for_test = np_data[index:].reshape((-1, frame_data_size, 1))
    print("data_for_train shape", data_for_train.shape)
    print("data_for_test shape", data_for_test.shape)
    (x_train, y_train) = data_for_train[:-1], data_for_train[1:]
    (x_test, y_test) = data_for_test[:-1],  data_for_test[1:]
    print("x_train.shape", x_train.shape)
    print("y_train.shape", y_train.shape)

    model.fit(x_train, y_train, epochs=1, batch_size=1)
    prediction = list(model.predict(data_for_test[:1]))
    dg.draw_json(NN_input.data_frame_to_dict(prediction))
    dg.draw_json(NN_input.data_frame_to_dict(list(data_for_test[0])))
    # make and show prediction

    if keep_display_on:
        dg.wait_till_close()
