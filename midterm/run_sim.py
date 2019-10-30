from math import cos, sin, atan2, sqrt, pi, exp

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from midterm.extended_information_filter import EIF
from heading_range_robot.robot_plotter import RobotPlotter

def main():
    data = loadmat('midterm/midterm_data.mat')
    # Unpack data
    true_state = data['X_tr']
    landmarks = data['m']
    w_c = data['om_c'][0]
    w = data['om'][0]
    t = data['t'][0]
    v = data['v'][0]
    v_c = data['v_c'][0]
    true_bearing = data['bearing_tr']
    true_range = data['range_tr']

    eif = EIF(landmarks.T)
    # Initialize plots
    robot_plotter = RobotPlotter()
    robot_plotter.init_plot(true_state[0], true_state[1], true_state[2], landmarks.T)

    # Initialize history for plotting
    all_mean_belief = []
    all_covariance_belief = []
    # Go through data
    for time_step in range(1, len(t)):
        t_curr = t[time_step]
        change_t = t[time_step] - t[time_step-1]

        eif.prediction_step(v_c[time_step], w_c[time_step], change_t)
        
        eif.measurement_step(np.vstack((true_range[time_step], true_bearing[time_step])))      
        robot_plotter.update_plot(eif.mean_belief[0], eif.mean_belief[1], eif.mean_belief[2])

        all_mean_belief.append(np.copy(ekf.mean_belief))
        all_covariance_belief.append(np.copy(ekf.covariance_belief))

if __name__ == "__main__":
    main()