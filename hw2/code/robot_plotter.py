from math import cos, sin

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from hw2.code.parameters import *

class RobotPlotter:
    def __init__(self):
        pass

    def init_plot(self, x, y, theta):
        plt.ion()
        fig, ax = plt.subplots()
        main_ax = ax
        self._fig = fig
        self._main_ax = main_ax
        main_ax.set_xlim(-20,20)
        main_ax.set_ylim(-20,20)
        main_ax.scatter(
            [LANDMARK_1_LOCATION[0],LANDMARK_2_LOCATION[0], LANDMARK_3_LOCATION[0]],
             [LANDMARK_1_LOCATION[1],LANDMARK_2_LOCATION[1], LANDMARK_3_LOCATION[1]])
        self.plot_robot(x, y, theta)
        plt.draw()

    def update_plot(self, x, y, theta):
        self.remove_for_next_step()
        self.plot_robot(x, y, theta)
        self._fig.canvas.draw_idle()
        plt.pause(0.0001)

    def plot_robot(self, x, y, theta):
        radius = 1
        self.circle = plt.Circle((x, y), radius, fill=False)
        self._main_ax.add_artist(self.circle)
        # self.arrow = patches.Arrow(x, y, x+cos(radius), y+sin(radius))
        # self._main_ax.add_patch(self.arrow)

    def remove_for_next_step(self):
        self.circle.remove()
        # self.arrow.remove()