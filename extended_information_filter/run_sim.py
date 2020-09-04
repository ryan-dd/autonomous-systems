from math import cos, sin, atan2, sqrt, pi, exp

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from extended_information_filter.extended_information_filter import EIF
from heading_range_robot.robot_plotter_eif import RobotPlotter
from tools.wrap import wrap


def main():
    data = loadmat('extended_information_filter/eif_data.mat')
    # Unpack data
    true_state = data['X_tr']
    true_state = wrap(data['X_tr'], index=2)
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
    robot_plotter.init_plot(true_state, eif.mean_belief, landmarks.T)

    # Initialize history for plotting
    all_mean_belief = [np.copy(eif.mean_belief)]
    all_covariance_belief = [np.copy(eif.covariance_belief)]
    all_information_vector = [np.copy(eif.info_vector)]
    # Go through data
    for time_step in range(1, len(t)):
        t_curr = t[time_step]
        change_t = t[time_step] - t[time_step-1]

        eif.prediction_step(v_c[time_step], w_c[time_step], change_t)
        eif.measurement_step(np.vstack((true_range[time_step], true_bearing[time_step])))

        robot_plotter.update_plot(true_state[:, time_step], eif.mean_belief)

        all_mean_belief.append(np.copy(wrap(eif.mean_belief, index=2)))
        all_covariance_belief.append(np.copy(eif.covariance_belief))
        all_information_vector.append(np.copy(eif.info_vector))
    # Plot summary
    plot_summary(true_state, all_mean_belief, all_covariance_belief, t, all_information_vector)


def plot_summary(all_true_states, all_mean_belief, all_variance_belief, time_steps_in_seconds, all_info_vector):
    # TODO information vector plots
    all_true_states = np.array(all_true_states)
    all_mean_belief = np.array(all_mean_belief).reshape((301, 3)).T
    all_variance_belief = np.array(all_variance_belief)
    all_info_vector = np.array(all_info_vector).T.reshape((3,-1))

    info_x = all_info_vector[0]
    info_y = all_info_vector[1]
    info_theta = all_info_vector[2]

    true_x = all_true_states[0]
    true_y = all_true_states[1]
    true_theta = all_true_states[2]

    mean_beliefs_about_x = all_mean_belief[0]
    mean_beliefs_about_y = all_mean_belief[1]
    mean_beliefs_about_theta = all_mean_belief[2]
    
    var_beliefs_about_x = all_variance_belief[:, 0, 0]
    var_beliefs_about_y = all_variance_belief[:, 1, 1]
    var_beliefs_about_theta = all_variance_belief[:, 2, 2]

    # Add static plots
    _, axes = plt.subplots(2, 2, figsize=(15, 15))
    ax1 = axes[0, 0]
    ax2 = axes[1, 0]
    ax3 = axes[1, 1]
    ax4 = axes[0, 1]

    ax1.plot(time_steps_in_seconds, true_x)
    ax1.plot(time_steps_in_seconds, mean_beliefs_about_x, '--')
    ax1.plot(time_steps_in_seconds, true_y)
    ax1.plot(time_steps_in_seconds, mean_beliefs_about_y, '--')
    ax1.plot(time_steps_in_seconds, true_theta)
    ax1.plot(time_steps_in_seconds, mean_beliefs_about_theta, '--')
    ax1.set_title("State vs Mean belief about State")
    ax1.set_xlabel("Time (s)")
    ax1.legend(["Actual X", "Mean X Belief",
                "Actual Y", "Mean Y Belief",
                 "Actual Theta", "Mean Theta Belief"])

    x_error = [
        (xt-mean_beliefs_about_x[i]) for i, xt in enumerate(true_x)]
    ax2.plot(time_steps_in_seconds, x_error)
    ax2.plot(time_steps_in_seconds, np.sqrt(var_beliefs_about_x)*2, 'b--')
    ax2.plot(time_steps_in_seconds, np.negative(
        np.sqrt(var_beliefs_about_x)*2), 'b--')
    ax2.legend(["X Error", "X Variance"])
    ax2.set_title("Error from X and mean belief")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("X (m)")
    ax2.set_ylim(-0.5, 0.5)

    y_error = [
        (vt-mean_beliefs_about_y[i])for i, vt in enumerate(true_y)]
    ax3.plot(time_steps_in_seconds, y_error)
    ax3.plot(time_steps_in_seconds, 
        np.sqrt(var_beliefs_about_y)*2, 'y--')
    ax3.plot(time_steps_in_seconds, 
        np.negative(np.sqrt(var_beliefs_about_y)*2), 'y--')
    ax3.legend(["Y Error", "Y Variance"])
    ax3.set_title("Error from Y and mean belief")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Y (m)")
    ax3.set_ylim(-0.5, 0.5)

    theta_error = [
        (vt-mean_beliefs_about_theta[i])for i, vt in enumerate(true_theta)]
    ax4.plot(time_steps_in_seconds, theta_error)
    ax4.plot(time_steps_in_seconds, 
        np.sqrt(var_beliefs_about_theta)*2, 'y--')
    ax4.plot(time_steps_in_seconds, 
        np.negative(np.sqrt(var_beliefs_about_theta)*2), 'y--')
    ax4.legend(["Theta Error", "Theta Variance"])
    ax4.set_title("Error from theta and mean belief")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Theta (radians)")
    ax4.set_ylim(-0.5, 0.5)

    plt.subplots()
    plt.plot(time_steps_in_seconds, info_x)
    plt.plot(time_steps_in_seconds, info_y)
    plt.plot(time_steps_in_seconds, info_theta)
    plt.title("Information Vector")
    plt.legend(["X", "Y", "Theta"])
    plt.xlabel("Time (s)")
    plt.ylabel("Value of information")

    plt.show()
    plt.pause(200)

if __name__ == "__main__":
    main()
