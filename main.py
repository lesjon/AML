import gamedrawer
import jsonGameProcessor
import jsonGameProcessorV2
import numpy as np
import tensorflow as tf
from time import sleep
# import matplotlib.pyplot as plt
# import lstm_frame_generator


def load_data_reader():
    data_reader = jsonGameProcessorV2.JsonGameReader()
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_11-09_TIGERs_Mannheim-vs-RoboTeam_Twente.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_12-41_KIKS-vs-Immortals.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_14-04_ER-Force-vs-UMass_Minutebots.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_15-55_CMμs-vs-RoboTeam_Twente.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_17-22_ZJUNlict-vs-KIKS.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_19-15_TIGERs_Mannheim-vs-RoboDragons.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_21-13_Immortals-vs-ER-Force.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_09-50_KIKS-vs-ER-Force.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_11-36_RoboDragons-vs-CMμs.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_13-23_UMass_Minutebots-vs-Immortals.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_15-34_ER-Force-vs-ZJUNlict.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_16-35_RoboDragons-vs-RoboTeam_Twente.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_18-01_UMass_Minutebots-vs-KIKS.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_19-24_CMμs-vs-TIGERs_Mannheim.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-19_20-30_ZJUNlict-vs-Immortals.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_09-13_KIKS-vs-RoboDragons.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_11-18_TIGERs_Mannheim-vs-ER-Force.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_12-37_CMμs-vs-ZJUNlict.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_14-11_UMass_Minutebots-vs-RoboTeam_Twente.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_15-57_RoboDragons-vs-Immortals.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_18-08_UMass_Minutebots-vs-ER-Force.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_19-27_Immortals-vs-ZJUNlict.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_19-39_Immortals-vs-ZJUNlict.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-20_21-21_TIGERs_Mannheim-vs-CMμs.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-21_09-12_ER-Force-vs-ZJUNlict.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-21_11-36_TIGERs_Mannheim-vs-ZJUNlict.json")
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-21_14-09_ZJUNlict-vs-CMμs.json")

    raw_data = jsonGameProcessorV2.JsonToRawData(keys_to_ignore=('robot_id', 'x_vel', 'y_vel'))
    raw_data.json_game_reader_to_raw(data_reader, verbose=0)
    return raw_data


if __name__ == '__main__':
    print("Starting project!")
    keep_display_on = True
    play_whole_match = True


    # NN_input = jsonGameProcessor.JsonToArray("logs/testWriterOutput.json")
    # NN_input = jsonGameProcessor.JsonToArray('Resources/Logs/RD_RT.json')
    # NN_input = jsonGameProcessor.JsonToArray("logs/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.log")
    # NN_input = jsonGameProcessor.JsonToArray("logs/lstm_creation.json")
    #
    #
    # if play_whole_match:
    #     for frame in NN_input.data:
    #         dg.draw_json(NN_input.data_frame_to_dict(frame))
    #         dg.clear_canvas()
    #         sleep(1/30)

    # NN_input = jsonGameProcessorV2.JsonToArray("logs/lstm_creation.json")
    NN_input = load_data_reader()
    dg = gamedrawer.GameDrawer()
    sleep(1)
    if play_whole_match:
        for n, fragment in enumerate(NN_input.data):
            print(n, "size of fragment:", np.shape(fragment))
            for frame in fragment:
                dg.draw_json(NN_input.data_frame_to_dict(frame))
                dg.clear_canvas()

    if keep_display_on:
        dg.wait_till_close()
