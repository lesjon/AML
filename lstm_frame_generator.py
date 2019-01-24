"""Example script showing how to use a stateful LSTM model
and how its stateless counterpart performs.

More documentation about the Keras LSTM model can be found at
https://keras.io/layers/recurrent/#lstm
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape
from keras.regularizers import l2
from keras.optimizers import adam
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

    return [x_out, y_out]


def create_model(stateful, batch_size, input_seq_len, input_len_frame, output_seq_len, dropout):
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
                   dropout=dropout,
                   recurrent_dropout=0.,
                   implementation=1,
                   return_sequences=True,  # Needs be true for all but the last layer
                   return_state=False,
                   go_backwards=False,
                   unroll=False))
    model.add(LSTM(average_input_output,
                   dropout=dropout,
                   recurrent_dropout=0.,
                   implementation=1,
                   kernel_regularizer=None,
                   recurrent_regularizer=None,
                   return_sequences=True))  # Needs be true for all but the last layer)
    model.add(LSTM(average_input_output,
                   dropout=dropout,
                   recurrent_dropout=0.,
                   implementation=1,
                   kernel_regularizer=None,
                   recurrent_regularizer=None))
    model.add(Dense(output_seq_len * input_len_frame,
                    kernel_regularizer=None,
                    activation=None))
    model.add(Reshape((output_seq_len, input_len_frame)))
    optim = adam(lr=0.00001)
    model.compile(loss='mse', optimizer=optim)
    return model


def split_train_test(all_xy, ratio):
    split_index = int(all_xy.shape[0] * ratio)
    return all_xy[:split_index], all_xy[split_index:]


def load_data_reader():
    data_reader = jsonGameProcessorV2.JsonGameReader()
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_11-09_TIGERs_Mannheim-vs-RoboTeam_Twente.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_12-41_KIKS-vs-Immortals.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_14-04_ER-Force-vs-UMass_Minutebots.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_15-55_CMμs-vs-RoboTeam_Twente.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_17-22_ZJUNlict-vs-KIKS.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_19-15_TIGERs_Mannheim-vs-RoboDragons.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_21-13_Immortals-vs-ER-Force.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_09-50_KIKS-vs-ER-Force.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_11-36_RoboDragons-vs-CMμs.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_13-23_UMass_Minutebots-vs-Immortals.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_15-34_ER-Force-vs-ZJUNlict.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_16-35_RoboDragons-vs-RoboTeam_Twente.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_18-01_UMass_Minutebots-vs-KIKS.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_19-24_CMμs-vs-TIGERs_Mannheim.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_20-30_ZJUNlict-vs-Immortals.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_09-13_KIKS-vs-RoboDragons.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_11-18_TIGERs_Mannheim-vs-ER-Force.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_12-37_CMμs-vs-ZJUNlict.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_14-11_UMass_Minutebots-vs-RoboTeam_Twente.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_15-57_RoboDragons-vs-Immortals.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_18-08_UMass_Minutebots-vs-ER-Force.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_19-27_Immortals-vs-ZJUNlict.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_19-39_Immortals-vs-ZJUNlict.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_21-21_TIGERs_Mannheim-vs-CMμs.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-21_09-12_ER-Force-vs-ZJUNlict.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-21_11-36_TIGERs_Mannheim-vs-ZJUNlict.json")
    # data_reader.add_file_to_data("Resources/LogsCut/2018-06-21_14-09_ZJUNlict-vs-CMμs.json")

    raw_data = jsonGameProcessorV2.JsonToRawData(keys_to_ignore=('robot_id', 'x_vel', 'y_vel'))
    raw_data.json_game_reader_to_raw(data_reader, verbose=0)
    return raw_data


def data_to_input_output(data_reader, minimum_seq_len, input_seq_len, output_seq_len, batch_size):
    xy = []
    for fragment in data_reader.data:
        if len(fragment) < minimum_seq_len or len(fragment) < input_seq_len + output_seq_len + batch_size:
            print("sequence length too short:", len(fragment))
            continue

        fragment = np.array([np.array([row]) for row in fragment])
        print("shape(fragment):", np.shape(fragment))
        # shift input_seq_len frames for prediction
        data_input = fragment[:-input_seq_len]
        expected_output = fragment[input_seq_len:]

        xy.append(create_input_of_right_length(data_input, expected_output, input_seq_len, output_seq_len))

        print("shape of xy:", len(xy), len(xy[-1]), len(xy[-1][0]), len(xy[-1][0][0]), len(xy[-1][0][0][0]))
        cut_off = len(xy[-1][0]) % batch_size
        print("cut_off:", cut_off)
        if 0 is not cut_off:
            xy[-1][0] = xy[-1][0][:-cut_off]
            xy[-1][1] = xy[-1][1][:-cut_off]
        print("shape of xy:", len(xy), len(xy[-1]), len(xy[-1][0]), len(xy[-1][0][0]), len(xy[-1][0][0][0]))
    return np.array(xy)


def main():
    # The input sequence length that the LSTM is trained on for each output point
    input_seq_len = 1
    # The output sequence length that the LSTM is trained on
    output_seq_len = 30

    minimum_seq_len = 50  # 30 frames per second,
    batch_size = 32
    epochs = 5
    dropout = 0.3

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
    model_stateful = create_model(True, batch_size, input_seq_len, input_len_frame, output_seq_len, dropout)
    model_stateful.summary()

    # create the trainable data:
    print("preparing data...")
    xy = data_to_input_output(data_reader, minimum_seq_len, input_seq_len, output_seq_len, batch_size)
    print("produced data with shape", xy.shape)

    print("splitting data...")
    np.random.shuffle(xy)  # shuffle first to get varied test sequences
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
