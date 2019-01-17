import json
import copy


class NNInput:
    data = []
    data_keys = None
    json_data = None

    object_group_keys = ["robots_yellow", "robots_blue", "balls"]

    def __init__(self, path_to_file):

        def dict2lists(dict_to_split):
            return list(dict_to_split.keys()), list(dict_to_split.values())

        read_file = open(path_to_file, "r")
        self.json_data = json.load(read_file)

        keys, robots_yellow_keys, robots_blue_keys, balls_keys = [], [], [], []

        for json_object in self.json_data:
            keys, values = dict2lists(json_object)

            robots_yellow_keys, robots_yellow_values = [], []
            for robot in values[keys.index(self.object_group_keys[0])]:
                k, v = dict2lists(robot)
                robots_yellow_keys.extend(k[1:])
                robots_yellow_values.extend(v[1:])

            robots_blue_keys, robots_blue_values = [], []
            for robot in values[keys.index(self.object_group_keys[1])]:
                k, v = dict2lists(robot)
                robots_blue_keys.extend(k[1:])
                robots_blue_values.extend(v[1:])

            balls_keys, balls_values = dict2lists(values[keys.index(self.object_group_keys[2])][0])

            self.data.append(robots_yellow_values + robots_blue_values + balls_values)
        self.data_keys = [robots_yellow_keys, robots_blue_keys, balls_keys]

    def data_frame_to_dict(self, data_frame, timestamp=0, stage="", command=""):
        # copy first frame as template for dict
        return_dict = copy.deepcopy(self.json_data[0])
        # set the not stored parameters
        return_dict['timestamp'] = timestamp
        return_dict['stage'] = stage
        return_dict['command'] = command

        print(self.object_group_keys, self.data_keys)
        for group_key, keys_per_group in zip(self.object_group_keys, self.data_keys):
            for n, (data_key, value) in enumerate(zip(keys_per_group, data_frame)):
                print(group_key, data_key, value)
                length_of_object = len(return_dict[group_key][0])
                return_dict[group_key][round((n / length_of_object) - 0.5)][data_key] = value
        return return_dict
