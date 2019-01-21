import gamedrawer
import jsonGameProcessor
import jsonGameProcessorV2
import numpy as np
import tensorflow as tf
from time import sleep
# import matplotlib.pyplot as plt


if __name__ == '__main__':
    print("Starting project!")
    keep_display_on = True
    play_whole_match = True
    retrain_nn = False


    # NN_input = jsonGameProcessor.JsonToArray("logs/testWriterOutput.json")
    # NN_input = jsonGameProcessor.JsonToArray('Resources/Logs/RD_RT.json')
    # NN_input = jsonGameProcessor.JsonToArray("logs/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.log")
    # NN_input = jsonGameProcessor.JsonToArray("logs/lstm_creation.json")
    #
    dg = gamedrawer.GameDrawer()
    sleep(1)
    #
    # if play_whole_match:
    #     for frame in NN_input.data:
    #         dg.draw_json(NN_input.data_frame_to_dict(frame))
    #         dg.clear_canvas()
    #         sleep(1/30)

    NN_input = jsonGameProcessorV2.JsonToArray("logs/lstm_creation.json")
    if play_whole_match:
        for fragment in NN_input.data:
            for frame in fragment:
                dg.draw_json(NN_input.data_frame_to_dict(frame))
                dg.clear_canvas()

    if keep_display_on:
        dg.wait_till_close()
