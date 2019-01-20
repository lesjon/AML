import json
import copy


x_vels = []
y_vels = []


class JsonToArray:
    """
    JsonToArray is an object that takes a json file with frames from a RoboCup SSL league game
    it will then hold transform the data into a flat array so it can be passed to a neural network
    it provides functions to switch between the json style dicts and the array of naked numbers
    """
    data = []
    data_keys = None
    json_data = None

    object_group_keys = ["robots_yellow", "robots_blue", "balls"]

    def __init__(self, path_to_file):
        """
        read the data from the location pointed to by path_to_file and process the file
        """

        def dict2lists(dict_to_split):
            return list(dict_to_split.keys()), list(dict_to_split.values())

        read_file = open(path_to_file, "r")
        self.json_data = json.load(read_file)

        keys, robots_yellow_keys, robots_blue_keys, balls_keys = [], [], [], []
        x_scaling = 60
        y_scaling = 45
        if isinstance(self.json_data, dict):
            self.json_data = self.json_data['data']
        for json_object in self.json_data:
            keys, values = dict2lists(json_object)

            if len(values[keys.index(self.object_group_keys[0])]) is not 8 \
                    or len(values[keys.index(self.object_group_keys[1])]) is not 8 \
                    or len(values[keys.index(self.object_group_keys[2])]) is not 1:
                print("warning: the line did not have the expected amount of balls and blue and yellow robots\n"
                      "data sizes were:", len(values[keys.index(self.object_group_keys[0])]),
                      len(values[keys.index(self.object_group_keys[1])]),
                      len(values[keys.index(self.object_group_keys[2])]))
                continue

            robots_yellow_values = []
            robots_blue_values = []
            robots_yellow_keys = []
            robots_blue_keys = []
            balls_keys = []
            balls_values = []
            for robot in values[keys.index(self.object_group_keys[0])]:
                robot['x_vel'] = robot['x_vel'] / x_scaling
                robot['y_vel'] = robot['y_vel'] / y_scaling
                x_vels.append(robot['x_vel'])
                y_vels.append(robot['y_vel'])
                k, v = dict2lists(robot)
                robots_yellow_keys.extend(k[1:])
                robots_yellow_values.extend(v[1:])

            for robot in values[keys.index(self.object_group_keys[1])]:
                robot['x_vel'] = robot['x_vel'] / x_scaling
                robot['y_vel'] = robot['y_vel'] / y_scaling
                x_vels.append(robot['x_vel'])
                y_vels.append(robot['y_vel'])
                k, v = dict2lists(robot)
                robots_blue_keys.extend(k[1:])
                robots_blue_values.extend(v[1:])

            for ball in values[keys.index(self.object_group_keys[2])]:
                ball['x_vel'] = ball['x_vel'] / x_scaling
                ball['y_vel'] = ball['y_vel'] / y_scaling
                x_vels.append(ball['x_vel'])
                y_vels.append(ball['y_vel'])
                balls_keys, balls_values = dict2lists(ball)
            self.data.append(robots_yellow_values + robots_blue_values + balls_values)
        self.data_keys = [robots_yellow_keys, robots_blue_keys, balls_keys]

    def data_frame_to_dict(self, data_frame, timestamp=0, stage="", command=""):
        """
        data_frame_to_dict, takes an array of float in sequence from which is created when instantiating NNInput
        it transforms the array of unordered data back into the readable dict
        """
        data_frame_copy = list(data_frame)
        # copy first frame as template for dict
        return_dict = copy.deepcopy(self.json_data[0])
        # set the not stored parameters
        return_dict['timestamp'] = timestamp
        return_dict['stage'] = stage
        return_dict['command'] = command
        # TODO also add this for the the robot_id's

        for group_key, keys_per_group in zip(self.object_group_keys, self.data_keys):
                for n, data_key in enumerate(keys_per_group):
                    length_of_object = len(return_dict[group_key][0])
                    if group_key in ["robots_yellow", "robots_blue"]:
                        length_of_object -= 1  # the id was ignored
                    index_in_object_array = int(n / length_of_object)
                    if len(data_frame_copy):
                        return_dict[group_key][index_in_object_array][data_key] = data_frame_copy.pop(0)
                    else:
                        return_dict[group_key][index_in_object_array][data_key] = -1

        return return_dict
