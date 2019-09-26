from math import cos, sin, atan2, sqrt, pi, exp

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import normal

from hw2.code.parameters import *
from hw2.code.robot import Robot
from hw2.code.ekf import EKF


def main():
    robot = Robot()
    ekf = EKF()
    total_time_steps = int(TOTAL_TIME/SAMPLE_PERIOD)
    for time_step in range(total_time_steps):
        theta_prev = ekf.mean_belief[2]
        t = time_step*SAMPLE_PERIOD
        robot.update_true_position_and_heading(t)
        ekf.prediction_step(theta_prev, robot.vc, robot.wc)
        v_measured = robot.v + v_measurement_noise()
        w_measured = robot.w + w_measurement_noise()
        ekf.measurement_step(v_measured, w_measured)

def v_measurement_noise():
    normal(0, STD_DEV_LOCATION_RANGE)

def w_measurement_noise():
    normal(0, STD_DEV_LOCATION_BEARING)

# def plot_robot(x, y):
#     plt.ion()
#     fig, ax = plt.subplots()
#     main_ax = ax
#     self._fig = fig
#     self._main_ax = main_ax
#     main_ax.set_xlim(-50,50)
#     main_ax.set_ylim(-50,50)

#     circle = plt.Circle(x, y, 3, color='b', fill=False)
#     self._main_ax.add_artist(circle)
#     circle.remove()

if __name__ == "__main__":
    main()