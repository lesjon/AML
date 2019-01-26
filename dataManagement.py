import json
import numpy as np

# Takes the file from string 'resource' and transforms the data into a 5 dimensional array: dataOrdered[set][frame][team1, team2, ball][robotID][x,y,x_vel,etc]
def import_file(resource, order):
    read_file = open(resource, "r")
    data = json.load(read_file)
    dataOrdered = []
    for set in data:
        dataOrdered.append([])
        for frame in set:
            dataOrdered[-1].append([])

            if order == 0:
                dataOrdered[-1][-1].append( np.zeros((len(frame["robots_yellow"]), 6)) )
                for j in range(len(frame["robots_yellow"])):
                    dataOrdered[-1][-1][-1][j,0] = np.float32(frame["robots_yellow"][j]["x"])
                    dataOrdered[-1][-1][-1][j,1] = np.float32(frame["robots_yellow"][j]["y"])
                    dataOrdered[-1][-1][-1][j,2] = np.float32(frame["robots_yellow"][j]["x_vel"]*100)
                    dataOrdered[-1][-1][-1][j,3] = np.float32(frame["robots_yellow"][j]["y_vel"]*100)
                    dataOrdered[-1][-1][-1][j,4] = np.float32(frame["robots_yellow"][j]["x_orien"])
                    dataOrdered[-1][-1][-1][j,5] = np.float32(frame["robots_yellow"][j]["y_orien"])
                dataOrdered[-1][-1][-1] = np.float32(dataOrdered[-1][-1][-1])

                dataOrdered[-1][-1].append( np.zeros((len(frame["robots_blue"]), 6)) )
                for j in range(len(frame["robots_blue"])):
                    dataOrdered[-1][-1][-1][j,0] = np.float32(frame["robots_blue"][j]["x"])
                    dataOrdered[-1][-1][-1][j,1] = np.float32(frame["robots_blue"][j]["y"])
                    dataOrdered[-1][-1][-1][j,2] = np.float32(frame["robots_blue"][j]["x_vel"]*100)
                    dataOrdered[-1][-1][-1][j,3] = np.float32(frame["robots_blue"][j]["y_vel"]*100)
                    dataOrdered[-1][-1][-1][j,4] = np.float32(frame["robots_blue"][j]["x_orien"])
                    dataOrdered[-1][-1][-1][j,5] = np.float32(frame["robots_blue"][j]["y_orien"])
                dataOrdered[-1][-1][-1] = np.float32(dataOrdered[-1][-1][-1])

            if order == 1:
                dataOrdered[-1][-1].append( np.zeros((len(frame["robots_blue"]), 6)) )
                for j in range(len(frame["robots_blue"])):
                    dataOrdered[-1][-1][-1][j,0] = np.float32(-frame["robots_blue"][j]["x"])
                    dataOrdered[-1][-1][-1][j,1] = np.float32(-frame["robots_blue"][j]["y"])
                    dataOrdered[-1][-1][-1][j,2] = np.float32(-frame["robots_blue"][j]["x_vel"]*100)
                    dataOrdered[-1][-1][-1][j,3] = np.float32(-frame["robots_blue"][j]["y_vel"]*100)
                    dataOrdered[-1][-1][-1][j,4] = np.float32(-frame["robots_blue"][j]["x_orien"])
                    dataOrdered[-1][-1][-1][j,5] = np.float32(-frame["robots_blue"][j]["y_orien"])
                dataOrdered[-1][-1][-1] = np.float32(dataOrdered[-1][-1][-1])

                dataOrdered[-1][-1].append( np.zeros((len(frame["robots_yellow"]), 6)) )
                for j in range(len(frame["robots_yellow"])):
                    dataOrdered[-1][-1][-1][j,0] = np.float32(-frame["robots_yellow"][j]["x"])
                    dataOrdered[-1][-1][-1][j,1] = np.float32(-frame["robots_yellow"][j]["y"])
                    dataOrdered[-1][-1][-1][j,2] = np.float32(-frame["robots_yellow"][j]["x_vel"]*100)
                    dataOrdered[-1][-1][-1][j,3] = np.float32(-frame["robots_yellow"][j]["y_vel"]*100)
                    dataOrdered[-1][-1][-1][j,4] = np.float32(-frame["robots_yellow"][j]["x_orien"])
                    dataOrdered[-1][-1][-1][j,5] = np.float32(-frame["robots_yellow"][j]["y_orien"])
                dataOrdered[-1][-1][-1] = np.float32(dataOrdered[-1][-1][-1])

            dataOrdered[-1][-1].append(np.zeros(4))

            for j in frame["balls"]:
                dataOrdered[-1][-1][-1][0] = np.float32(j["x"])
                dataOrdered[-1][-1][-1][1] = np.float32(j["y"])
                dataOrdered[-1][-1][-1][2] = np.float32(j["x_vel"])
                dataOrdered[-1][-1][-1][3] = np.float32(j["y_vel"])
            dataOrdered[-1][-1][-1] = np.float32(dataOrdered[-1][-1][-1])
    return dataOrdered

def file_to_x_y(file, frames):
    x = []
    y = []
    for set in file:
        for frame in range(len(set[:-frames])):
            if len(set[frame][0]) == len(set[frame+frames][0]):
                x.append(set[frame])
                y.append(set[frame+frames])
    return x, y

def make_pairs(x, y):
    pairedSet = np.stack((x, y))
    pairedSet = pairedSet.transpose(1,0,2)
    return pairedSet

def get_data(resource, frames, order):
    data = import_file(resource, order)
    x,y = file_to_x_y(data, 30)
    pairs = make_pairs(x,y)
    return pairs
