import jsonGameProcessorV2
import json
from save_load_nn import *
import numpy as np

model_stateful = load_nn(name="lstm")

NN_input = jsonGameProcessorV2.JsonToArray()
NN_input.add_file_to_data("Resources/LogsCut/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.json", verbose=0)

frame = NN_input.data[0][0]
with open("logs/lstm_creation.json", 'w') as f:
    f.write('[\n')
    last_value = None
    for x in range(1000):
        f.write(json.dumps(NN_input.data_frame_to_dict(frame)))
        frame = model_stateful.predict(np.reshape(frame, (1, 1, len(frame))))[0].tolist()

        last_value = f.tell()
        f.write(',\n')
    f.seek(last_value)
    f.truncate()

    f.write('\n]')