from math import cos, sin, atan2, sqrt, pi, exp
from statistics import mean, variance

import numpy as np
from matplotlib import pyplot as plt

from heading_range_robot.parameters import *
from fast_slam.fast_SLAM import FastSLAM
from heading_range_robot.robot import Robot
from heading_range_robot.robot_plotter_fast_slam import RobotPlotter, plot_summary


def main():
    robot = Robot(INITIAL_X, INITIAL_Y, INITIAL_THETA, SAMPLE_PERIOD, ALPHA1, ALPHA2, ALPHA3, ALPHA4) 
    robot_plotter = RobotPlotter()
    total_time_steps = int(TOTAL_TIME/SAMPLE_PERIOD)
    all_features = np.random.random((2,2))*40-20
    # mx = np.array([6, -7, 6, 0, -8, 1, 4, -5])[:,np.newaxis]
    # my = np.array([4, 8, -1, -9, -5, 1, -4, 3])[:,np.newaxis]
    # all_features = np.append(mx,my,axis=1)
    pkf = FastSLAM(SAMPLE_PERIOD, 1000, all_features)
    robot_plotter.init_plot(robot.x, robot.y, robot.theta, all_features, particles=pkf.particles)

    all_mean_belief = []
    all_covariance_belief = []
    all_true_state = []
    for time_step in range(total_time_steps):
        t = time_step*SAMPLE_PERIOD
        robot.update_true_position_and_heading(t)
        first_step = False
        if time_step==0:
            first_step = True      
        pkf.particle_filter(robot, first_step)      
        robot_plotter.update_plot(robot.x, robot.y, robot.theta, pkf.particles, pkf.particle_to_plot)

        all_true_state.append(np.copy(robot.actual_position))
        all_mean_belief.append(np.copy(pkf.mean_belief))
        all_covariance_belief.append(np.copy(pkf.covariance_belief))
    plot_summary(all_true_state, all_mean_belief, all_covariance_belief, SAMPLE_PERIOD)

if __name__ == "__main__":
    main()
