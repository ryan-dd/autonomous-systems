from math import sin, cos, pi, sqrt
import numpy as np
from numpy.random import normal


class Robot():
    def __init__(self, x, y, theta, change_t, alpha1, alpha2, alpha3, alpha4):
        self._initialize_position_and_heading(x, y, theta, change_t)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

    def commanded_translational_velocity(self, t):
        return 1 + 0.5*cos(2*pi*(0.2)*t)

    def commanded_rotational_velocity(self, t):
        return -0.2 + 2*cos(2*pi*(0.6)*t)

    def translational_noise(self, v, w):
        translational_error_variance = self.alpha1*(v**2)+self.alpha2*(w**2)
        return normal(0, sqrt(translational_error_variance))

    def rotational_noise(self, v, w):
        rotational_error_variance = self.alpha3*(v**2)+self.alpha4*(w**2)
        return normal(0, sqrt(rotational_error_variance))

    def _initialize_position_and_heading(self, x, y, theta, change_t):
        self.x = x
        self.y = y
        self.theta = theta
        self.change_t = change_t
        self.actual_position = np.vstack((x, y, theta))

    def update_true_position_and_heading(self, t):
        vc = self.commanded_translational_velocity(t)
        wc = self.commanded_rotational_velocity(t)
        v = vc + self.translational_noise(vc, wc)
        w = wc + self.rotational_noise(vc, wc)
        self.vc = vc
        self.wc = wc
        self.v = v
        self.w = w
        self.actual_position = self._next_position_from_state(
            self.x, self.y, self.theta, v, w, self.change_t)

    def _next_position_from_state(self, x, y, theta, vt, wt, change_t):
        x_next = x + (-vt/wt)*sin(theta) + (vt/wt)*sin(theta + wt*change_t)
        y_next = y + (vt/wt)*cos(theta) - (vt/wt)*cos(theta + wt*change_t)
        theta_next = theta + wt*change_t
        self.x = x_next
        self.y = y_next
        self.theta = theta_next
        return np.vstack((x_next, y_next, theta_next))
