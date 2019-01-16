from tkinter import *
from time import sleep


class DrawGame:
    master = None
    canvas = None
    width = 1200
    height = 900
    field_width = 12/4
    field_height = 9/4

    def __init__(self):

        def _create_circle(self, x, y, r, **kwargs):
            return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
        Canvas.create_circle = _create_circle

        self.master = Tk()
        self.scroll_bar = Scrollbar(self.master, orient="horizontal")
        self.scroll_bar.pack()

        self.canvas = Canvas(self.master, width=self.width, height=self.height)
        self.canvas.pack()

        self.update()

    def update(self):
        self.master.update()

    @staticmethod
    def wait_till_close():
        mainloop()

    def draw_game_from_json(self, json_file):
        print("draw game from json")
        for json_object in json_file:
            self.draw_json(json_object)
            self.update()
            self.clear_canvas()
            sleep(1/60)
            '''self.master.after(1000/60, self.draw_game_from_json)
            yield'''

    def draw_json(self, json_data):

        def draw_object(game_object, size, color):
            vel_vector_factor = 0.01
            x_pos = (game_object['x'] + self.field_width / 2) * self.width / self.field_width
            y_pos = (game_object['y'] + self.field_height / 2) * self.height / self.field_height
            x_vel = (game_object['x_vel'] + self.field_width / 2) * self.width / self.field_width
            y_vel = (game_object['y_vel'] + self.field_height / 2) * self.height / self.field_height
            y_vel *= vel_vector_factor
            x_vel *= vel_vector_factor
            self.canvas.create_circle(x_pos, y_pos, size, fill=color)
            self.canvas.create_line(x_pos, y_pos, x_pos + x_vel, y_pos + y_vel, width=1)

        for robot in json_data['robots_yellow']:
            draw_object(robot, 10, "yellow")
        for robot in json_data['robots_blue']:
            draw_object(robot, 10, "blue")
        for ball in json_data['balls']:
            draw_object(ball, 4, "orange")

    def clear_canvas(self):
        self.canvas.delete("all")