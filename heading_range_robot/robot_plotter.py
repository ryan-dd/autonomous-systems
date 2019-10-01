from math import cos, sin

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class RobotPlotter:
    def __init__(self):
        pass

    def init_plot(self, x, y, theta, landmarks):
        plt.ion()
        fig, ax = plt.subplots()
        main_ax = ax
        self._fig = fig
        self._main_ax = main_ax
        main_ax.set_xlim(-20,20)
        main_ax.set_ylim(-20,20)
        landmarks = np.array(landmarks)
        main_ax.scatter(
            landmarks[:,0], landmarks[:,1])
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

def plot_summary(all_true_states, all_mean_belief, all_variance_belief, all_kt, sample_period):
    time_steps = list(range(len(all_true_states)))
    time_steps_in_seconds = [t*sample_period for t in time_steps]

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