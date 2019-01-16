import json


class NNInput:
    data = []
    data_keys = None
    json_data = None

    def __init__(self, path_to_file):

        def dict2lists(dict_to_split):
            return list(dict_to_split.keys()), list(dict_to_split.values())

        read_file = open(path_to_file, "r")
        self.json_data = json.load(read_file)

        keys, robots_yellow_keys, robots_blue_keys, balls_keys = [], [], [], []

        for json_object in self.json_data:
            keys, values = dict2lists(json_object)

            robots_yellow_keys, robots_yellow_values = [], []
            for robot in values[keys.index("robots_yellow")]:
                k, v = dict2lists(robot)
                robots_yellow_keys.extend(k[1:])
                robots_yellow_values.extend(v[1:])

            robots_blue_keys, robots_blue_values = [], []
            for robot in values[keys.index("robots_blue")]:
                k, v = dict2lists(robot)
                robots_blue_keys.extend(k[1:])
                robots_blue_values.extend(v[1:])

            balls_keys, balls_values = dict2lists(values[keys.index("balls")][0])

            self.data.append(robots_yellow_values + robots_blue_values + balls_values)
        self.data_keys = robots_yellow_keys + robots_blue_keys + balls_keys
