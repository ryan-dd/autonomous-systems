from math import sin, cos, pi

from hw2.code.parameters import *


class Robot():
    def __init__(self):
        self._initialize_position_and_heading()

    def commanded_translational_velocity(self, t):
        return 1 + 0.5*cos(2*pi*(0.2)*t)

    def commanded_rotational_velocity(self, t):
        return -0.2 + 2*cos(2*pi*(0.6)*t)

    def translational_noise(self, v, w):
        translational_error_variance = ALPHA1*(v**2)+ALPHA2*(w**2)
        return normal(0, sqrt(translational_error_variance))

    def rotational_noise(self, v, w):
        rotational_error_variance = ALPHA3*(v**2)+ALPHA4*(w**2)
        return normal(0, sqrt(rotational_error_variance))

    def _initialize_position_and_heading(self):
        self.x = INITIAL_X
        self.y = INITIAL_Y
        self.theta = INITIAL_THETA
        self.change_t = SAMPLE_PERIOD
        self.actual_position = np.vstack((INITIAL_X, INITIAL_Y, INITIAL_THETA))

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
            self.x, self.y, self.theta, v, w, SAMPLE_PERIOD)

    def _next_position_from_state(self, x, y, theta, vt, wt, change_t):
        x_next = x + (-vt/wt)*sin(theta) + (vt/wt)*sin(theta + wt*change_t)
        y_next = y + (vt/wt)*cos(theta) - (vt/wt)*cos(theta + wt*change_t)
        theta_next = theta + wt*change_t
        self.x = x_next
        self.y = y_next
        self.theta = theta_next
        return np.vstack((x_next, y_next, theta_next))
