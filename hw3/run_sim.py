from math import cos, sin, atan2, sqrt, pi, exp

import numpy as np
from matplotlib import pyplot as plt

from heading_range_robot.parameters import *
from hw3.unscented_kalman_filter import UKF
from heading_range_robot.robot import Robot
from heading_range_robot.robot_plotter import RobotPlotter, plot_summary


def main():
    matlab_name = "hw3_5_soln_data.mat"
    robot = Robot(INITIAL_X, INITIAL_Y, INITIAL_THETA, SAMPLE_PERIOD, ALPHA1, ALPHA2, ALPHA3, ALPHA4, truth_file=matlab_name)
    ukf = UKF(SAMPLE_PERIOD)
    robot_plotter = RobotPlotter()
    total_time_steps = int(TOTAL_TIME/SAMPLE_PERIOD)
    robot_plotter.init_plot(robot.x, robot.y, robot.theta, LANDMARKS)

    all_mean_belief = []
    all_covariance_belief = []
    all_kalman_gain = []
    all_true_state = []
    for time_step in range(total_time_steps):
        t = time_step*SAMPLE_PERIOD
        robot.update_true_position_and_heading(t)
        ukf.prediction_step(robot)
        
        ukf.measurement_step(robot.actual_position, robot)      
        robot_plotter.update_plot(robot.x, robot.y, robot.theta)

        all_true_state.append(np.copy(np.copy(robot.actual_position)))
        all_mean_belief.append(np.copy(ukf.mean_belief))
        all_covariance_belief.append(np.copy(ukf.covariance_belief))
        all_kalman_gain.append(np.copy(ukf.kt))
    plot_summary(all_true_state, all_mean_belief, all_covariance_belief, all_kalman_gain, SAMPLE_PERIOD)

if __name__ == "__main__":
    main()