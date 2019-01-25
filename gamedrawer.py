from tkinter import *
import time


def create_circle(canvas, x, y, r, **kwargs):
    return canvas.create_oval(x - r, y - r, x + r, y + r, **kwargs)


class GameDrawer:
    """
    Object with functions to draw SSL game frames on a TKinter canvas
    """
    master = None
    canvas = None
    width = 1200
    height = 900
    field_width = 12/6
    field_height = 9/4.5
    field_image = None
    delete_each_frame = []
    FPS_counter = None
    frame_count = 0
    n_frames = 10
    timer = 0
    json_data = []
    fragment_n = 0
    frame_n = 0

    def __init__(self):
        self.master = Tk()

        self.scroll_bar = Scrollbar(self.master, orient="horizontal")
        self.scroll_bar.pack()
        self.text_frame = Frame(self.master)
        self.FPS_counter = Text(self.text_frame, width=10, height=1, borderwidth=0)
        self.FPS_counter.pack(side=RIGHT)

        self.fragment_frame = Text(self.text_frame, width=30, height=1, borderwidth=0)
        self.fragment_frame.pack(side=LEFT)
        self.text_frame.pack(anchor="n")

        self.canvas = Canvas(self.master, width=self.width, height=self.height, bg="green", xscrollcommand=self.scroll_bar.set)
        self.field_image = PhotoImage(file="images/field.gif")
        self.canvas.create_image(-30, -30, image=self.field_image, anchor=NW)
        self.canvas.pack()

        self.scroll_bar.config(command=self.canvas.xview)
        self.master.bind_all('<Left>', self.arrow_press)
        self.master.bind_all('<Right>', self.arrow_press)
        self.update()

    def arrow_press(self, event):
        if event.keysym == 'Left':
            if self.frame_n is not 0:
                self.frame_n -= 1
            else:
                if self.fragment_n is not 0:
                    self.fragment_n -= 1
                    self.frame_n = len(self.json_data[self.fragment_n]) - 1
        if event.keysym == 'Right':
            if self.frame_n < len(self.json_data[self.fragment_n]) - 1:
                self.frame_n += 1
            else:
                if self.fragment_n < len(self.json_data[self.fragment_n][self.fragment_n]) - 1:
                    self.fragment_n += 1
                    self.frame_n = 0
        self.clear_canvas()
        self.draw_json(self.json_data[self.fragment_n][self.frame_n])
        self.fragment_frame.delete(1.0, END)
        self.fragment_frame.insert(INSERT, "Fragment:" + str(self.fragment_n) + ",frame:" + str(self.frame_n))


    def update(self):
        self.master.update()

    @staticmethod
    def wait_till_close():
        mainloop()

    def draw_game_from_nparray(self, nparray):
            frame = {}
            frame['robots_yellow'] = []
            frame['robots_blue'] = []
            frame['robots_orange'] = []
            frame['balls'] = []
            if nparray.size % 6 is not 0:
                print("[draw_game_from_nparray] Warning! Expected 6 parameters per robot. %d out of %d parameters unaccounted for" % (nparray.size % 6, nparray.size))

            # Fill frame with robots
            for i in range(0, int( nparray.size / 6)):
                offset = i * 6
                if i<=7:
                    frame['robots_yellow'].append({})     # Create new robot in robots_yellow
                    frame['robots_yellow'][i]['x']          = nparray[offset + 0]
                    frame['robots_yellow'][i]['y']          = nparray[offset + 1]
                    frame['robots_yellow'][i]['x_vel']      = nparray[offset + 2]
                    frame['robots_yellow'][i]['y_vel']      = nparray[offset + 3]
                    frame['robots_yellow'][i]['x_orien']    = nparray[offset + 4]
                    frame['robots_yellow'][i]['y_orien']    = nparray[offset + 5]
                elif i<= 15:
                    frame['robots_blue'].append({})     # Create new robot in robots_yellow
                    frame['robots_blue'][i-8]['x']          = nparray[offset + 0]
                    frame['robots_blue'][i-8]['y']          = nparray[offset + 1]
                    frame['robots_blue'][i-8]['x_vel']      = nparray[offset + 2]
                    frame['robots_blue'][i-8]['y_vel']      = nparray[offset + 3]
                    frame['robots_blue'][i-8]['x_orien']    = nparray[offset + 4]
                    frame['robots_blue'][i-8]['y_orien']    = nparray[offset + 5]
                else:
                    frame['robots_orange'].append({})     # Create new robot in robots_yellow
                    frame['robots_orange'][i-16]['x']          = nparray[offset + 0]
                    frame['robots_orange'][i-16]['y']          = nparray[offset + 1]
                    frame['robots_orange'][i-16]['x_vel']      = nparray[offset + 2]
                    frame['robots_orange'][i-16]['y_vel']      = nparray[offset + 3]
                    frame['robots_orange'][i-16]['x_orien']    = nparray[offset + 4]
                    frame['robots_orange'][i-16]['y_orien']    = nparray[offset + 5]
            # Draw frame
            self.draw_json(frame)

    def draw_game_from_json(self):
        # This function is to play a whole match
        print("draw game from json")
        for fragment in self.json_data:
            for json_object in fragment:
                self.draw_json(json_object)
                self.clear_canvas()
                # time.sleep(1/60)
                '''self.master.after(1000/60, self.draw_game_from_json)
                yield'''

    def draw_json(self, json_data, dash=None):
        """
        This function draws one frame in json/ dict format
        :param json_data: one frame in a dict/ json
        :param dash: show these with this dash
        :return:
        """
        def draw_object(game_object, size, color):
            # y_vel_vector_factor = 45
            # x_vel_vector_factor = 60
            # vel_vector_factor = 2  # 0.01
            x_pos = (game_object['x'] + self.field_width / 2) * self.width / self.field_width
            y_pos = (game_object['y'] + self.field_height / 2) * self.height / self.field_height
            x_vel = game_object['x_vel']  #  + self.field_width / 2) * self.width / self.field_width
            y_vel = game_object['y_vel']  # + self.field_height / 2) * self.height / self.field_height
            # y_vel *= y_vel_vector_factor * vel_vector_factor
            # x_vel *= x_vel_vector_factor * vel_vector_factor
            self.delete_each_frame.append(create_circle(self.canvas, x_pos, y_pos, size, fill=color, dash=dash))
            self.delete_each_frame.append(self.canvas.create_line(x_pos, y_pos, x_pos + x_vel, y_pos + y_vel, width=1, dash=dash))

        for robot in json_data['robots_yellow']:
            draw_object(robot, 9, "yellow")
        for robot in json_data['robots_blue']:
            draw_object(robot, 9, "blue")
        # for robot in json_data['robots_orange']:
        #     draw_object(robot, 10, "orange")
        for ball in json_data['balls']:
            draw_object(ball, 4, "orange")
        self.update()

    def clear_canvas(self):
        """
        clear the objects which were drawn for one frame
        :return:
        """
        self.canvas.delete(*self.delete_each_frame)
        self.delete_each_frame = []

        self.frame_count += 1
        if not self.frame_count % self.n_frames:
            self.FPS_counter.delete(1.0, END)
            fps = round(self.n_frames / (time.time() - self.timer))
            self.FPS_counter.insert(INSERT, str(fps) + "FPS")
            self.timer = time.time()


if __name__ == "__main__":
    # test the code
    import jsonGameProcessorV2
    data_reader = jsonGameProcessorV2.JsonGameReader()
    data_reader.add_file_to_data("Resources/LogsCut/2018-06-18_09-06_ZJUNlict-vs-UMass_Minutebots.json")
    dg = GameDrawer()
    dg.json_data = data_reader.json_data
    dg.wait_till_close()
