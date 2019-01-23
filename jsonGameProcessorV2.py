import json
import copy
import numpy as np


x_vels = []
y_vels = []
x_scaling = .01#.0060
y_scaling = .01#.0045
clip = True


class JsonToArray:
    """
    JsonToArray is an object that takes a json file with frames from a RoboCup SSL league game
    it will then hold transform the data into a flat array so it can be passed to a neural network
    it provides functions to switch between the json style dicts and the array of naked numbers
    """
    data = []  # [0 for z in range(1)]
    data_keys = []
    json_data = []
    keys_to_ignore = []
    group_sizes = []

    object_group_keys = ["robots_yellow", "robots_blue", "balls"]

    def __init__(self, path_to_file='', keys_to_ignore=()):
        """read the data from the location pointed to by path_to_file and process the file
        :param: path_to_file path to the first file to load
        :return: this object
        """
        self.keys_to_ignore = keys_to_ignore
        if path_to_file:
            self.add_file_to_data(path_to_file)

    def add_file_to_data(self, path, verbose=2):
        def dict2lists(dict_to_split):
            return list(dict_to_split.keys()), list(dict_to_split.values())

        with open(path, "r") as read_file:
            fragments = json.load(read_file)
        self.json_data.append(fragments)
        # keys, robots_yellow_keys, robots_blue_keys, balls_keys = [], [], [], []
        for fragment_n, fragment in enumerate(fragments):
            for json_frame in fragment:
                keys, values = dict2lists(json_frame)

                if len(values[keys.index(self.object_group_keys[0])]) is not 8 \
                        or len(values[keys.index(self.object_group_keys[1])]) is not 8 \
                        or len(values[keys.index(self.object_group_keys[2])]) is not 1:
                    if verbose > 0:
                        print("warning: the line did not have the expected amount of balls and blue and yellow robots\n"
                              "this line is ignored, data sizes were:",
                              len(values[keys.index(self.object_group_keys[0])]),
                              len(values[keys.index(self.object_group_keys[1])]),
                              len(values[keys.index(self.object_group_keys[2])]),
                              "\n")
                    continue

                robots_yellow_values = []
                robots_blue_values = []
                robots_yellow_keys = []
                robots_blue_keys = []
                balls_keys = []
                balls_values = []
                for robot in values[keys.index(self.object_group_keys[0])]:
                    # robot['x_vel'] = robot['x_vel'] / x_scaling
                    # robot['y_vel'] = robot['y_vel'] / y_scaling
                    # x_vels.append(robot['x_vel'])
                    # y_vels.append(robot['y_vel'])
                    k, v = dict2lists(robot)
                    robots_yellow_keys.extend(k)#[1:])
                    robots_yellow_values.extend(v)#[1:])

                for robot in values[keys.index(self.object_group_keys[1])]:
                    # robot['x_vel'] = robot['x_vel'] / x_scaling
                    # robot['y_vel'] = robot['y_vel'] / y_scaling
                    # x_vels.append(robot['x_vel'])
                    # y_vels.append(robot['y_vel'])
                    k, v = dict2lists(robot)
                    robots_blue_keys.extend(k)#[1:])
                    robots_blue_values.extend(v)#[1:])

                for ball in values[keys.index(self.object_group_keys[2])]:
                    # ball['x_vel'] = ball['x_vel'] / x_scaling
                    # ball['y_vel'] = ball['y_vel'] / y_scaling
                    # x_vels.append(ball['x_vel'])
                    # y_vels.append(ball['y_vel'])
                    balls_keys, balls_values = dict2lists(ball)
                combined_values = robots_yellow_values + robots_blue_values + balls_values
                self.data_keys = robots_yellow_keys + robots_blue_keys + balls_keys
                self.group_sizes = [len(robots_yellow_keys), len(robots_blue_keys), len(balls_keys)]
                for key in self.keys_to_ignore:
                    # print("key:", key)
                    # print("key:", self.data_keys)
                    indices = [i for i, x in enumerate(self.data_keys) if x == key]
                    # print("indices:", indices)
                    for index in sorted(indices, reverse=True):
                        del combined_values[index]
                        del self.data_keys[index]
                        total = 0
                        for i, group_size in enumerate(self.group_sizes):
                            total += group_size
                            if index < total:
                                self.group_sizes[i] -= 1
                                break

                if not self.data:
                    self.data = [[combined_values]]
                    # print("if not shape of self.data", np.shape(self.data))
                elif len(self.data) <= fragment_n:
                    self.data.append([combined_values])
                    # print("elif shape of self.data", np.shape(self.data))
                else:
                    self.data[fragment_n].append(combined_values)
                    # print("else shape of self.data", np.shape(self.data))

    def data_frame_to_dict(self, data_frame, timestamp=0, stage="", command=""):
        """
        data_frame_to_dict, takes an array of float in sequence from which is created when instantiating NNInput
        it transforms the array of unordered data back into the readable dict
        """
        # create a copy of the input so we can pop on it without editing the source
        data_frame_copy = list(data_frame)
        # copy first frame as template for dict
        return_dict = copy.deepcopy(self.json_data[0][0][0])
        # create copy of the list of keys
        all_keys = copy.copy(self.data_keys)
        # set the not stored parameters
        return_dict['timestamp'] = timestamp
        return_dict['stage'] = stage
        return_dict['command'] = command
        # TODO also add this for the the robot_id's

        key = all_keys.pop(0)
        for group_key in self.object_group_keys:
            print("group_key", group_key)
            print("return_dict", return_dict)
            if len(all_keys) < 1:
                break
            first_key = None
            for robot_or_ball in return_dict[group_key]:
                print("robot_or_ball", robot_or_ball)
                if len(all_keys) < 1:
                    break

                while True:
                    robot_or_ball[key] = data_frame_copy.pop(0)
                    key = all_keys.pop(0)
                    if len(all_keys) < 1:
                        break
                    if first_key is key:
                        break
                    if None is first_key:
                        first_key = key

        # for group_key in self.object_group_keys:
        #         for n, data_key in enumerate(self.data_keys):
        #             length_of_object = len(return_dict[group_key][0])
        #             index_in_object_array = n // length_of_object
        #             if len(data_frame_copy):
        #                 return_dict[group_key][index_in_object_array][data_key] = data_frame_copy.pop(0)
        #             else:
        #                 return_dict[group_key][index_in_object_array][data_key] = -1

        return return_dict
