from math import cos, sin, atan2, sqrt, pi, exp
from statistics import mean, variance

import numpy as np
from matplotlib import pyplot as plt

from heading_range_robot.parameters import *
from hw4.particle_filter import ParticleFilter
from heading_range_robot.robot import Robot
from heading_range_robot.robot_plotter import RobotPlotter, plot_summary


def main():
    robot = Robot(INITIAL_X, INITIAL_Y, INITIAL_THETA, SAMPLE_PERIOD, ALPHA1, ALPHA2, ALPHA3, ALPHA4) 
    pkf = ParticleFilter(SAMPLE_PERIOD, number_of_particles=1000)
    robot_plotter = RobotPlotter()
    total_time_steps = int(TOTAL_TIME/SAMPLE_PERIOD)
    robot_plotter.init_plot(robot.x, robot.y, robot.theta, LANDMARKS, particles=pkf.particles)

    all_mean_belief = []
    all_covariance_belief = []
    all_kalman_gain = []
    all_true_state = []
    for time_step in range(total_time_steps):
        t = time_step*SAMPLE_PERIOD
        robot.update_true_position_and_heading(t)        
        pkf.particle_filter(robot)      
        robot_plotter.update_plot(robot.x, robot.y, robot.theta, particles=pkf.particles)

        all_true_state.append(np.copy(robot.actual_position))
        all_mean_belief.append(np.copy(pkf.mean_belief))
        all_covariance_belief.append(np.copy(pkf.covariance_belief))
    plot_summary(all_true_state, all_mean_belief, all_covariance_belief, SAMPLE_PERIOD)

if __name__ == "__main__":

    main()