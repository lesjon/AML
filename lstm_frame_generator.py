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
import json
import gamedrawer


np.random.seed(1986)

# NN_input = jsonGameProcessor.JsonToArray("logs/testWriterOutput.json")
# NN_input = jsonGameProcessor.JsonToArray('Resources/Logs/RD_RT.json')
# NN_input = jsonGameProcessor.JsonToArray("logs/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.log")
NN_input = jsonGameProcessor.JsonToArray("logs/2018-06-20_11-18_TIGERs_Mannheim-vs-ER-Force.log")
# NN_input = jsonGameProcessorV2.JsonToArray("Resources/LogsCut/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.json")


# The input sequence length that the LSTM is trained on for each output point
lahead = 2

# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 100

# for reproducability

print("original shape:", np.array(NN_input.data).shape)
prepackaged = np.array([[row] for row in NN_input.data])
print("reshaped prepackaged.shape:", np.array(prepackaged).shape)
# length of input sequence
input_len_sequence = len(NN_input.data)
# length of input frame
input_len_frame = len(NN_input.data[0])
print("input_lens", input_len_sequence, input_len_frame)

data_input = prepackaged[:-lahead] #gen_uniform_amp(amp=0.1, xn=input_len + to_drop)

print("input_lens", len(data_input), len(data_input[0]))
# set the target to be a N-point average of the input
expected_output = prepackaged[lahead:]#data_input.rolling(window=tsteps, center=False).mean()

# when lahead > 1, need to convert the input to "rolling window view"
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html

print('before repeat Input shape:', data_input.shape)
if lahead > 1:
    data_input = np.repeat(data_input, repeats=lahead, axis=1)
    data_input = data_input
    for i, c in enumerate(range(data_input.shape[1])):
        data_input[c] = shift(data_input[c], i, cval=np.NaN)

# drop the nan
expected_output = expected_output[lahead:]
data_input = data_input[lahead:]

print(np.argwhere(np.isnan(data_input)))
print(np.argwhere(np.isnan(expected_output)))

print('Input shape:', data_input.shape)
print('Output shape:', expected_output.shape)
# print('Input head: ')
# print(data_input.head(1))
# print('Output head: ')
# print(expected_output.head(1))
# print('Input tail: ')
# print(data_input.tail(1))
# print('Output tail: ')
# print(expected_output.tail(1))

# print('Showing first frame input and expected output')
# dg.draw_json(NN_input.data_frame_to_dict(data_input.iloc[100, :].tolist()))
# dg.draw_json(NN_input.data_frame_to_dict(expected_output.iloc[100, :].tolist()), dash=(1, 1))
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
              input_shape=(lahead, input_len_frame),
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