from math import cos, sin

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np

class RobotPlotter:
    def __init__(self):
        pass

    def init_plot(self, x, y, theta, landmarks, particles):
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
        true_hist = main_ax.plot(np.array([[x]]), np.array([[y]]))
        self.plot_robot(x, y, theta)
        if particles is not None:
            p_plot = np.array([particle[0] for particle in particles])
            self.particles_scatter = main_ax.scatter(p_plot[:,0], p_plot[:,1])
        
        self.estimated_landmarks = ax.scatter([], [])
        self.error_bounds_ellipses = []

        self.data = np.array([[x, y]])
        plt.draw()

    def update_plot(self, x, y, theta, particles, particle_to_plot):
        self.remove_for_next_step()
        self.plot_robot(x, y, theta)
        self.data = np.append(self.data, np.array([[x,y]]).reshape(1,2), axis=0)
        self._main_ax.plot(self.data[:,0], self.data[:,1])
        self._fig.canvas.draw_idle()
        p_plot = np.array([particle[0] for particle in particles])
        self.particles_scatter.set_offsets(p_plot.reshape(len(p_plot),-1)[:,:2])
        
        particle_position = particle_to_plot[0]
        landmark_beliefs = particle_to_plot[1]
        landmark_covars = particle_to_plot[2]
        initialized = particle_to_plot[3]
        for index, est_landmark in enumerate(landmark_beliefs):
            if not initialized[index]:
                continue
            cov = landmark_covars[index]
            self.error_bounds_ellipses.append(confidence_ellipse(est_landmark[0], est_landmark[1], cov, self._main_ax))
        self.estimated_landmarks.set_offsets(landmark_beliefs)
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
        for shape in self.error_bounds_ellipses:
            shape.remove()
        self.error_bounds_ellipses = []

def plot_summary(all_true_states, all_mean_belief, all_variance_belief, sample_period, all_kt=None):
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
    var_beliefs_about_y = all_variance_belief[:, 1, 0]
    var_beliefs_about_theta = all_variance_belief[:, 2, 0]

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
        np.sqrt(np.abs(var_beliefs_about_x))*2), 'b--')
    ax2.legend(["X Error", "X Variance"])
    ax2.set_title("Error from X and mean belief")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("X (m)")
    ax2.set_ylim(-0.5, 0.5)

    y_error = [
        (vt-mean_beliefs_about_y[i])for i, vt in enumerate(true_y)]
    ax3.plot(time_steps_in_seconds, y_error)
    ax3.plot(time_steps_in_seconds, 
        np.sqrt(np.abs(var_beliefs_about_y))*2, 'y--')
    ax3.plot(time_steps_in_seconds, 
        np.negative(np.sqrt(np.abs(var_beliefs_about_y))*2), 'y--')
    ax3.legend(["Y Error", "Y Variance"])
    ax3.set_title("Error from Y and mean belief")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Y (m)")
    ax3.set_ylim(-0.5, 0.5)

    theta_error = [
        (vt-mean_beliefs_about_theta[i])for i, vt in enumerate(true_theta)]
    ax4.plot(time_steps_in_seconds, theta_error)
    ax4.plot(time_steps_in_seconds, 
        np.sqrt(np.abs(var_beliefs_about_theta))*2, 'y--')
    ax4.plot(time_steps_in_seconds, 
        np.negative(np.sqrt(np.abs(var_beliefs_about_theta))*2), 'y--')
    ax4.legend(["Theta Error", "Theta Variance"])
    ax4.set_title("Error from theta and mean belief")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Theta (radians)")
    ax4.set_ylim(-0.174, 0.174)

    if all_kt is not None:
        ax5.plot(time_steps_in_seconds, np.array(all_kt)[:, 0, 0])
        ax5.plot(time_steps_in_seconds, np.array(all_kt)[:, 1, 0])
        ax5.plot(time_steps_in_seconds, np.array(all_kt)[:, 2, 0])
        ax5.plot(time_steps_in_seconds, np.array(all_kt)[:, 0, 1])
        ax5.plot(time_steps_in_seconds, np.array(all_kt)[:, 1, 1])
        ax5.plot(time_steps_in_seconds, np.array(all_kt)[:, 2, 1])
        ax5.set_title("Kalman filter gain for position")
        ax5.legend(["X kalman gain range", "Y kalman gain range", "Theta Kalman Gain range",
                "X kalman gain bearing", "Y kalman gain bearing", "Theta Kalman Gain bearing"])

    sc = plt.imshow(all_variance_belief[-1], cmap='Blues', interpolation='nearest', origin='lower')
    plt.colorbar(sc)
    plt.show()
    plt.pause(200)

# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(mean_x, mean_y, cov, ax, n_std=2.0, facecolor='r', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        alpha=0.5,
        **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)