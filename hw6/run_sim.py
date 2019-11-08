from math import cos, sin, atan2, sqrt, pi, exp

import numpy as np
from matplotlib import pyplot as plt

from heading_range_robot.parameters import *
from hw6.extended_kalman_filter_slam import EKF
from heading_range_robot.robot import Robot
from heading_range_robot.robot_plotter_hw6 import RobotPlotter, plot_summary


def main():
    robot = Robot(INITIAL_X, INITIAL_Y, INITIAL_THETA, SAMPLE_PERIOD, ALPHA1, ALPHA2, ALPHA3, ALPHA4)
    ekf = EKF(SAMPLE_PERIOD)
    robot_plotter = RobotPlotter()
    total_time_steps = int(TOTAL_TIME/SAMPLE_PERIOD)
    robot_plotter.init_plot(robot.actual_position, ekf.mean_belief, ekf.all_features)

    all_mean_belief = []
    all_covariance_belief = []
    all_kalman_gain = []
    all_true_state = []
    for time_step in range(total_time_steps):
        t = time_step*SAMPLE_PERIOD
        robot.update_true_position_and_heading(t)
        ekf.prediction_step(robot.vc, robot.wc)
        
        ekf.measurement_step(robot.actual_position)      
        robot_plotter.update_plot(robot.actual_position, ekf.mean_belief, ekf.covariance_belief)

        all_true_state.append(np.copy(np.copy(robot.actual_position)))
        all_mean_belief.append(np.copy(ekf.mean_belief))
        all_covariance_belief.append(np.copy(ekf.covariance_belief))
        all_kalman_gain.append(np.copy(ekf.kt))
    plot_summary(all_true_state, all_mean_belief, all_covariance_belief, all_kalman_gain, SAMPLE_PERIOD)

if __name__ == "__main__":
    main()