from math import cos, sin, atan2, sqrt, pi, exp

import numpy as np
from matplotlib import pyplot as plt

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

    all_mean_belief = []
    all_covariance_belief = []
    all_kalman_gain = []
    all_true_state = []
    for time_step in range(total_time_steps):
        theta_prev = ekf.mean_belief[2]
        t = time_step*SAMPLE_PERIOD
        robot.update_true_position_and_heading(t)
        ekf.prediction_step(theta_prev, robot.vc, robot.wc)
        
        ekf.measurement_step(robot.actual_position)      
        plotter.update_plot(robot.x, robot.y, robot.theta)

        all_true_state.append(np.copy(np.copy(robot.actual_position)))
        all_mean_belief.append(np.copy(ekf.mean_belief))
        all_covariance_belief.append(np.copy(ekf.covariance_belief))
        all_kalman_gain.append(np.copy(ekf.kt))
    plot_everything(all_true_state, all_mean_belief, all_covariance_belief, all_kalman_gain)

def plot_everything(all_true_states, all_mean_belief, all_variance_belief, all_kt):
    time_steps = list(range(len(all_true_states)))
    time_steps_in_seconds = [t*SAMPLE_PERIOD for t in time_steps]

    all_true_states = np.array(all_true_states)
    all_mean_belief = np.array(all_mean_belief)
    all_variance_belief = np.array(all_variance_belief)

    true_x = all_true_states[:, 0, 0]
    true_y = all_true_states[:, 1, 0]
    true_theta = all_true_states[:, 2, 0]

    mean_beliefs_about_x = all_mean_belief[:, 0, 0]
    mean_beliefs_about_y = all_mean_belief[:, 1, 0]
    mean_beliefs_about_theta = all_mean_belief[:, 2, 0]
    
    var_beliefs_about_x = all_variance_belief[:, 0, 0]
    var_beliefs_about_y = all_variance_belief[:, 1, 1]
    var_beliefs_about_theta = all_variance_belief[:, 2, 2]

    # Add static plots
    _, axes = plt.subplots(3, 2, figsize=(15, 15))
    ax1 = axes[0, 0]
    ax2 = axes[1, 0]
    ax3 = axes[1, 1]
    ax4 = axes[0, 1]
    ax5 = axes[2, 0]
    ax6 = axes[2, 1]

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

    ax5.plot(time_steps_in_seconds, np.array(all_kt)[:, 0])
    ax5.plot(time_steps_in_seconds, np.array(all_kt)[:, 1])
    ax5.plot(time_steps_in_seconds, np.array(all_kt)[:, 2])
    ax5.set_title("Kalman filter gain for position")
    ax5.legend(["X kalman gain", "Y kalman gain", "Theta Kalman Gain"])
    ax5.set_xlabel("Time (s)")

    plt.show()
    plt.pause(200)
    i=0

if __name__ == "__main__":
    main()