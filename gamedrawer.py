from tkinter import *
import time


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

    def __init__(self):

        def _create_circle(self, x, y, r, **kwargs):
            return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
        Canvas.create_circle = _create_circle

        self.master = Tk()

        self.scroll_bar = Scrollbar(self.master, orient="horizontal")
        self.scroll_bar.pack()

        self.FPS_counter = Text(self.master, width=10, height=1, borderwidth=0)
        self.FPS_counter.pack(anchor="nw")

        self.canvas = Canvas(self.master, width=self.width, height=self.height, bg="green", xscrollcommand=self.scroll_bar.set)
        self.field_image = PhotoImage(file="images/field.gif")
        self.canvas.create_image(-30, -30, image=self.field_image, anchor=NW)
        self.canvas.pack()

        self.scroll_bar.config(command=self.canvas.xview)

        self.update()

    def update(self):
        self.master.update()

    @staticmethod
    def wait_till_close():
        mainloop()

    def draw_game_from_json(self, json_file):
        """
        This function is to play a whole match
        :param json_file: a file containing frames from a game
        :return:
        """
        print("draw game from json")
        for s, json_object in enumerate(json_file):
            self.canvas.xview_moveto(s/len(json_file))
            self.draw_json(json_object)
            self.clear_canvas()
            # time.sleep(1/60)
            '''self.master.after(1000/60, self.draw_game_from_json)
            yield'''

    def draw_json(self, json_data):
        """
        This function draws one frame in json/ dict format
        :param json_data: one frame in a dict/ json
        :return:
        """
        def draw_object(game_object, size, color):
            vel_vector_factor = 0.01
            x_pos = (game_object['x'] + self.field_width / 2) * self.width / self.field_width
            y_pos = (game_object['y'] + self.field_height / 2) * self.height / self.field_height
            x_vel = (game_object['x_vel'] + self.field_width / 2) * self.width / self.field_width
            y_vel = (game_object['y_vel'] + self.field_height / 2) * self.height / self.field_height
            y_vel *= vel_vector_factor
            x_vel *= vel_vector_factor
            self.delete_each_frame.append(self.canvas.create_circle(x_pos, y_pos, size, fill=color))
            self.delete_each_frame.append(self.canvas.create_line(x_pos, y_pos, x_pos + x_vel, y_pos + y_vel, width=1))

        for robot in json_data['robots_yellow']:
            draw_object(robot, 10, "yellow")
        for robot in json_data['robots_blue']:
            draw_object(robot, 10, "blue")
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

