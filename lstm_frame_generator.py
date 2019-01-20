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
import json
import gamedrawer

# for reproducability
np.random.seed(1995)

# NN_input = jsonGameProcessor.JsonToArray("logs/testWriterOutput.json")
# NN_input = jsonGameProcessor.JsonToArray('Resources/Logs/RD_RT.json')
# NN_input = jsonGameProcessor.JsonToArray("logs/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.log")
NN_input = jsonGameProcessor.JsonToArray("logs/2018-06-20_11-18_TIGERs_Mannheim-vs-ER-Force.log")
# NN_input = jsonGameProcessorV2.JsonToArray("Resources/LogsCut/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.json")


# The input sequence length that the LSTM is trained on for each output point
input_seq_len = 2

# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 100

# make each fragment a sequence
temp = NN_input.data#[0]

prepackaged = np.array([[row] for row in temp])
# length of input sequence
input_len_sequence = len(temp)
# length of input frame
input_len_frame = len(temp[0])

# shift input_seq_len frames for prediction
data_input = prepackaged[:-input_seq_len]
expected_output = prepackaged[input_seq_len:]


# when lahead > 1, need to convert the input to "rolling window view"
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
def create_input_of_right_length(x, y, n_samples_input):
    print('before repeat Input shape:', x.shape)
    if n_samples_input > 1:
        x = np.repeat(x, repeats=n_samples_input, axis=1)
        x = x
        for i, c in enumerate(range(x.shape[1])):
            x[c] = shift(x[c], i, cval=np.NaN)

    # drop the nan
    y = y[n_samples_input:]
    x = x[n_samples_input:]

    # check if there are no NaN left in array
    if np.argwhere(np.isnan(x)) or np.argwhere(np.isnan(y)):
        print(np.argwhere(np.isnan(x)))
        print(np.argwhere(np.isnan(y)))
    return x, y


data_input, expected_output = create_input_of_right_length(data_input, expected_output, input_seq_len)
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


def create_model(stateful):
    model = Sequential()
    model.add(LSTM(50,
                   input_shape=(input_seq_len, input_len_frame),
                   batch_size=batch_size,
                   stateful=stateful))
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

    # some reshaping
    reshape_3 = lambda x: x.reshape((x.shape[0], x.shape[1], input_len_frame))  # 1
    x_train = reshape_3(x_train)
    x_test = reshape_3(x_test)

    reshape_2 = lambda x: x.reshape((x.shape[0], input_len_frame))  # 1
    y_train = reshape_2(y_train)
    y_test = reshape_2(y_test)

    return (x_train, y_train), (x_test, y_test)


print("before split_data:", data_input.shape, expected_output.shape)
(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)


print('Creating Stateful Model...')
model_stateful = create_model(stateful=True)

print(np.argwhere(np.isnan(x_train)))
print(np.argwhere(np.isnan(y_train)))

print('Training')
for i in range(epochs):
    print('Epoch', i + 1, '/', epochs)
    # Note that the last state for sample i in a batch will
    # be used as initial state for sample i in the next batch.
    # Thus we are simultaneously training on batch_size series with
    # lower resolution than the original series contained in data_input.
    # Each of these series are offset by one step and can be
    # extracted with data_input[i::batch_size].
    model_stateful.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=1,
                       verbose=1,
                       validation_data=(x_test, y_test),
                       shuffle=False)
    model_stateful.reset_states()

print('Predicting')
predicted_stateful = model_stateful.predict(x_test, batch_size=batch_size)

# dg = gamedrawer.GameDrawer()
frame = x_test[500][None]
with open("logs/lstm_creation.json", 'w') as f:
    f.write('[\n')
    last_value = None
    for x in range(1000):
        f.write(json.dumps(NN_input.data_frame_to_dict(frame[0][1])))
        frame = np.append(frame[0], model_stateful.predict(frame), axis=0)[None]
        frame = frame[0][1:][None]

        last_value = f.tell()
        f.write(',\n')
    f.seek(last_value)
    f.truncate()

    f.write('\n]')
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