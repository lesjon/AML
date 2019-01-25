import gamedrawer
import jsonGameProcessor
import jsonGameProcessorV2
import numpy as np
from time import sleep
# import matplotlib.pyplot as plt
import lstm_frame_generator


if __name__ == '__main__':
    print("Starting project!")
    keep_display_on = True
    play_whole_match = True

    data_reader = jsonGameProcessorV2.JsonGameReader()
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.json")
    dg = gamedrawer.GameDrawer()
    dg.json_data = data_reader.json_data
    sleep(1)
    if play_whole_match:
        dg.draw_game_from_json()

    if keep_display_on:
        dg.wait_till_close()
