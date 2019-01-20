import gamedrawer
import jsonGameProcessor
import jsonGameProcessorV2
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt


def save_nn(nn_model, name="model"):
    # serialize model to JSON
    model_json = nn_model.to_json()
    with open("Saved_models/" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    nn_model.save_weights("Saved_models/" + name + ".h5")
    print("Saved model " + name + " to disk")


def load_nn(name="model"):
    # load json and create model
    json_file = open("Saved_models/" + name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Saved_models/" + name + ".h5")
    print("Loaded model " + name + " from disk")
    return loaded_model


if __name__ == '__main__':
    print("Starting project!")
    keep_display_on = True
    play_whole_match = True
    retrain_nn = False


    # NN_input = jsonGameProcessor.JsonToArray("logs/testWriterOutput.json")
    # NN_input = jsonGameProcessor.JsonToArray('Resources/Logs/RD_RT.json')
    # NN_input = jsonGameProcessor.JsonToArray("logs/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.log")
    # NN_input = jsonGameProcessor.JsonToArray("logs/lstm_creation.json")

    dg = gamedrawer.GameDrawer()
    # if play_whole_match:
    #     for frame in NN_input.data:
    #         dg.draw_json(NN_input.data_frame_to_dict(frame))
    #         dg.clear_canvas()

    NN_input = jsonGameProcessorV2.JsonToArray("Resources/LogsCut/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.json")
    NN_input.add_file_to_data("Resources/LogsCut/2018-06-18_11-09_TIGERs_Mannheim-vs-RoboTeam_Twente.json")
    if play_whole_match:
        for fragment in NN_input.data:
            for frame in fragment:
                dg.draw_json(NN_input.data_frame_to_dict(frame))
                dg.clear_canvas()

    if keep_display_on:
        dg.wait_till_close()
