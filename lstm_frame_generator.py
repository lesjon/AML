"""Example script showing how to use a stateful LSTM model
and how its stateless counterpart performs.

More documentation about the Keras LSTM model can be found at
https://keras.io/layers/recurrent/#lstm
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape
import jsonGameProcessorV2
# import gamedrawer
from save_load_nn import *
import json


def create_input_of_right_length(x, y, n_samples_input, n_samples_output):
    # print('before repeat Input shape:', x.shape)
    def reshape_and_shift(fragment, n_samples):
        fragment_shape = list(fragment.shape)
        fragment_shape[0] -= (n_samples - 1)
        fragment_shape[1] += (n_samples - 1)
        tensor = np.zeros(fragment_shape)
        for i in range(fragment.shape[0] - (n_samples - 1)):
            tensor[i] = [fragment[i + n] for n in range(n_samples)]
        return tensor

    x_out = reshape_and_shift(x, n_samples_input)
    y_out = reshape_and_shift(y, n_samples_output)
    # cut both to the same length:
    if x_out.shape[0] > y_out.shape[0]:
        x_out = x_out[0: y_out.shape[0]]
    else:
        y_out = y_out[0: x_out.shape[0]]

    return x_out, y_out


def create_model(stateful, batch_size, input_seq_len, input_len_frame, output_seq_len):
    model = Sequential()
    average_input_output = input_seq_len * input_len_frame + output_seq_len * input_len_frame
    average_input_output //= 2
    model.add(LSTM(average_input_output,
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
                   return_sequences=True,  # Needs be true for all but the last layer
                   return_state=False,
                   go_backwards=False,
                   unroll=False))
    model.add(LSTM(average_input_output,
                   dropout=0.01,
                   recurrent_dropout=0.01,
                   implementation=1,
                   return_sequences=True))  # Needs be true for all but the last layer)
    model.add(LSTM(average_input_output))
    model.add(Dense(output_seq_len * input_len_frame,
                    activation=None))
    model.add(Reshape((output_seq_len, input_len_frame)))
    model.compile(loss='mse', optimizer='adam')
    return model


def split_train_test(all_xy, ratio):
    split_index = int(all_xy.shape[0] * ratio)
    return all_xy[:split_index], all_xy[split_index:]


def load_data_reader():
    data_reader = jsonGameProcessorV2.JsonToArray(keys_to_ignore=('robot_id', 'x_vel', 'y_vel'))
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.json", verbose=0)
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_14-04_ER-Force-vs-UMass_Minutebots.json", verbose=0)
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_15-34_ER-Force-vs-ZJUNlict.json", verbose=0)
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_19-24_CMμs-vs-TIGERs_Mannheim.json", verbose=0)
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_18-08_UMass_Minutebots-vs-ER-Force.json", verbose=0)
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_21-21_TIGERs_Mannheim-vs-CMμs.json", verbose=0)
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-21_11-36_TIGERs_Mannheim-vs-ZJUNlict.json", verbose=0)
    # data_reader.add_file_to_data("logs/testfile.json", verbose=2)
    return data_reader


def data_to_input_output(data_reader, minimum_seq_len, input_seq_len, output_seq_len):
    xy = []
    for fragment in data_reader.data:
        if len(fragment) < minimum_seq_len:
            print("sequence length too short:", len(fragment))
            continue

        fragment = np.array([np.array([row]) for row in fragment])

        # shift input_seq_len frames for prediction
        data_input = fragment[:-input_seq_len]
        expected_output = fragment[input_seq_len:]
        xy.append(create_input_of_right_length(data_input, expected_output, input_seq_len, output_seq_len))
    return np.array(xy)


def main():
    # The input sequence length that the LSTM is trained on for each output point
    input_seq_len = 1
    # The output sequence length that the LSTM is trained on
    output_seq_len = 30

    minimum_seq_len = 100  # 30 frames per second,
    batch_size = 1
    epochs = 5

    save_model = True
    save_model_at_epochs = [2 ** i for i in range(int(np.log2(epochs) + 1))]
    save_model_at_epochs.append(epochs-1)
    save_model_at_epochs.insert(0, 0)

    # for reproducibility
    np.random.seed(1995)

    print("Loading games...")
    data_reader = load_data_reader()
    input_len_frame = len(data_reader.data[0][0])  # the length of a single frame
    print("Loading done!")
    print("features used from dataset:", set(data_reader.data_keys))

    print('Creating Stateful Model...')
    model_stateful = create_model(True, batch_size, input_seq_len, input_len_frame, output_seq_len)
    model_stateful.summary()

    # create the trainable data:
    print("preparing data...")
    xy = data_to_input_output(data_reader, minimum_seq_len, input_seq_len, output_seq_len)
    print("produced data with shape", xy.shape)

    print("splitting data...")
    xy_train, xy_test = split_train_test(xy, 0.8)
    print("produced train and test set, sizes:", np.shape(xy_train), np.shape(xy_test))

    print('Training')
    with open("logs/lstmlog.txt", "w") as logfile:
        model_stateful.summary(print_fn=lambda line: logfile.write(line + '\n'))
        for epoch in range(epochs):
            print('Epoch', epoch + 1, '/', epochs)
            # shuffle the fragments
            np.random.shuffle(xy_train)
            for x, y in xy_train:
                # reset the memory of the network at the start of each continuous fragment
                model_stateful.reset_states()
                # choose one of the test sequences:
                test_index = np.random.randint(0, xy_test.shape[0])
                history_callback = model_stateful.fit(x,
                                                      y,
                                                      batch_size=batch_size,
                                                      epochs=1,
                                                      verbose=1,
                                                      validation_data=(xy_test[test_index][0], xy_test[test_index][1]),
                                                      shuffle=False)
                out_dict = {"epoch": epoch}
                out_dict.update(history_callback.history)
                logfile.write(json.dumps(out_dict))
                logfile.write('\n')
                logfile.flush()
            if save_model:
                if epoch in save_model_at_epochs:
                    save_nn(model_stateful, name="lstm" + str(epoch))


if __name__ == '__main__':
    main()
