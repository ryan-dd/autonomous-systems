from math import cos, sin, atan2, sqrt, pi, exp

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import normal

from hw2.code.parameters import *
from hw2.code.robot import Robot
from hw2.code.ekf import EKF
from hw2.code.plotter import Plotter


def main():
    robot = Robot()
    ekf = EKF()
    plotter = Plotter()
    total_time_steps = int(TOTAL_TIME/SAMPLE_PERIOD)
    plotter.init_plot(robot.x, robot.y, robot.theta)
    for time_step in range(total_time_steps):
        theta_prev = ekf.mean_belief[2]
        t = time_step*SAMPLE_PERIOD
        robot.update_true_position_and_heading(t)
        ekf.prediction_step(theta_prev, robot.vc, robot.wc)
        v_measured = robot.v + v_measurement_noise()
        w_measured = robot.w + w_measurement_noise()
        ekf.measurement_step(v_measured, w_measured)
        plotter.update_plot(robot.x, robot.y, robot.theta)
        

def v_measurement_noise():
    return normal(0, STD_DEV_LOCATION_RANGE)

def w_measurement_noise():
    return normal(0, STD_DEV_LOCATION_BEARING)

if __name__ == "__main__":
    main()