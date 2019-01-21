"""Example script showing how to use a stateful LSTM model
and how its stateless counterpart performs.

More documentation about the Keras LSTM model can be found at
https://keras.io/layers/recurrent/#lstm
"""
import numpy as np
from scipy.ndimage.interpolation import shift
from keras.models import Sequential
from keras.layers import Dense, LSTM
import jsonGameProcessor
import jsonGameProcessorV2
import gamedrawer
from save_load_nn import *
import matplotlib.pyplot as plt


# when lahead > 1, need to convert the input to "rolling window view"
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
def create_input_of_right_length(x, y, n_samples_input):
    print('before repeat Input shape:', x.shape)
    if n_samples_input > 1:
        x = np.repeat(x, repeats=n_samples_input, axis=1)
        x = x
        for i, c in enumerate(range(x.shape[1])):
            x[c] = shift(x[c], i, cval=np.NaN)

    # drop the the just created NaNs
    y = y[n_samples_input:]
    x = x[n_samples_input:]

    # check if there are no NaN left in array
    if np.argwhere(np.isnan(x)) or np.argwhere(np.isnan(y)):
        print(np.argwhere(np.isnan(x)))
        print(np.argwhere(np.isnan(y)))
    return x, y


def create_model(stateful):
    model = Sequential()
    model.add(LSTM(50,
                   input_shape=(input_seq_len, input_len_frame),
                   batch_size=batch_size,
                   stateful=stateful,
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal',
                   bias_initializer='zeros',
                   unit_forget_bias=True,
                   kernel_regularizer=None,
                   recurrent_regularizer=None,
                   bias_regularizer=None,
                   activity_regularizer=None,
                   kernel_constraint=None,
                   recurrent_constraint=None,
                   bias_constraint=None,
                   dropout=0.01,
                   recurrent_dropout=0.01,
                   implementation=1,
                   return_sequences=False,
                   return_state=False,
                   go_backwards=False,
                   unroll=False))
    model.add(Dense(input_len_frame))
    model.compile(loss='mse', optimizer='adam')
    return model


# split train/test data
def split_data(x, y, ratio=0.8):
    to_train = int(input_len_sequence * ratio)
    # tweak to match with batch_size
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]

    # tweak to match with batch_size
    to_drop = x.shape[0] % batch_size
    if to_drop > 0:
        x_test = x_test[:-1 * to_drop]
        y_test = y_test[:-1 * to_drop]

    return (x_train, y_train), (x_test, y_test)


# split train/test data
def do_not_split_data(x, y):
    return (x, y)


def reshape_2(x, input_len_frame):
    return x.reshape((x.shape[0], input_len_frame))  # 1


# for reproducability
np.random.seed(1995)

# NN_input = jsonGameProcessor.JsonToArray("logs/testWriterOutput.json")
# NN_input = jsonGameProcessor.JsonToArray('Resources/Logs/RD_RT.json')
# NN_input = jsonGameProcessor.JsonToArray("logs/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.log")
# NN_input = jsonGameProcessor.JsonToArray("logs/2018-06-20_11-18_TIGERs_Mannheim-vs-ER-Force.log")
NN_input = jsonGameProcessorV2.JsonToArray()
NN_input.add_file_to_data("Resources/LogsCut/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.json", verbose=0)
NN_input.add_file_to_data("Resources/LogsCut/2018-06-18_14-04_ER-Force-vs-UMass_Minutebots.json", verbose=0)
NN_input.add_file_to_data("Resources/LogsCut/2018-06-19_15-34_ER-Force-vs-ZJUNlict.json", verbose=0)
NN_input.add_file_to_data("Resources/LogsCut/2018-06-19_19-24_CMμs-vs-TIGERs_Mannheim.json", verbose=0)
NN_input.add_file_to_data("Resources/LogsCut/2018-06-20_18-08_UMass_Minutebots-vs-ER-Force.json", verbose=0)

# plt.hist(jsonGameProcessorV2.x_vels, bins=100, range=(-2, 2))
# plt.figure()
# plt.hist(jsonGameProcessorV2.y_vels, bins=100, range=(-2, 2))
# plt.show()
# The input sequence length that the LSTM is trained on for each output point
input_seq_len = 1

input_len_frame = len(NN_input.data[0][0])
# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 1000
x_test, y_test = None, None

print('Creating Stateful Model...')
model_stateful = create_model(stateful=True)

print('Training')
for i in range(epochs):
    print('Epoch', i + 1, '/', epochs)
    # make each fragment a sequence
    for temp in NN_input.data:
        if len(temp) < 20:
            print("temp length too short:", len(temp))
            continue
        model_stateful.reset_states()

        prepackaged = np.array([[row] for row in temp])
        # length of input sequence
        input_len_sequence = prepackaged.shape[0]
        # # length of input frame
        # input_len_frame = prepackaged.shape[2]

        # shift input_seq_len frames for prediction
        data_input = prepackaged[:-input_seq_len]
        expected_output = prepackaged[input_seq_len:]

        data_input, expected_output = create_input_of_right_length(data_input, expected_output, input_seq_len)

        expected_output = reshape_2(expected_output, input_len_frame)
        # print('Showing first frame input and expected output')
        # plt.subplot(411)
        # plt.hist(jsonGameProcessor.x_vels, bins=100, range=(-2, 2))
        # plt.subplot(412)
        # plt.hist(jsonGameProcessor.y_vels, bins=100, range=(-2, 2))
        # plt.subplot(413)
        # plt.plot(jsonGameProcessor.x_vels)
        # plt.subplot(414)
        # plt.plot(jsonGameProcessor.y_vels)
        # plt.show()

        # print("before split_data:", data_input.shape, expected_output.shape)
        # (x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
        x_train, y_train = data_input, expected_output
        if None is x_test:
            print("created test")
            x_test, y_test = x_train, y_train
            continue
        print('x_train.shape: ', x_train.shape)
        print('y_train.shape: ', y_train.shape)
        print('x_test.shape: ', x_test.shape)
        print('y_test.shape: ', y_test.shape)

        model_stateful.fit(x_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=1,
                           verbose=1,
                           validation_data=(x_test, y_test),
                           shuffle=False)

print('Predicting')
# predicted_stateful = model_stateful.predict(x_test, batch_size=batch_size)

save_nn(model_stateful, name="lstm")

# dg = gamedrawer.GameDrawer()
#
# print('Creating Stateless Model...')
# model_stateless = create_model(stateful=False)
#
# print('Training')
# model_stateless.fit(x_train,
#                     y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test),
#                     shuffle=False)
#
# print('Predicting')
# predicted_stateless = model_stateless.predict(x_test, batch_size=batch_size)
#
# # ----------------------------
#
# print('Plotting Results')
# plt.subplot(3, 1, 1)
# plt.plot(y_test)
# plt.title('Expected')
# plt.subplot(3, 1, 2)
# # drop the first "tsteps-1" because it is not possible to predict them
# # since the "previous" timesteps to use do not exist
# plt.plot((y_test - predicted_stateful).flatten()[tsteps - 1:])
# plt.title('Stateful: Expected - Predicted')
# plt.subplot(3, 1, 3)
# plt.plot((y_test - predicted_stateless).flatten())
# plt.title('Stateless: Expected - Predicted')
#
# plt.figure()
# plt.plot(y_test)
# plt.plot(predicted_stateful.flatten()[tsteps - 1:])
# plt.plot(predicted_stateless.flatten())
# plt.legend(("expected", "stateful", "stateless"))
# plt.show()