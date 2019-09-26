from math import sin, cos

from hw2.code.parameters import *


class Robot():
    def __init__(self):
        self._initialize_position_and_heading

    def _initialize_position_and_heading(self):
        self.x = INITIAL_X
        self.y = INITIAL_Y
        self.theta = INITIAL_THETA
        self.change_t = SAMPLE_PERIOD
        self.actual_position = np.vstack((INITIAL_X, INITIAL_Y, INITIAL_THETA))

    def update_true_position_and_heading(self, t):
        vc = commanded_translational_velocity(t)
        wc = commanded_rotational_velocity(t)
        v = vc + translational_noise(vc, wc)
        w = wc + rotational_noise(vc, wc)
        self.vc = vc
        self.wc = wc
        self.v = v
        self.w = w
        self.actual_position = self.next_position_from_state(
            self.x, self.y, self.theta, v, w, SAMPLE_PERIOD)
        
    def next_position_from_state(self, x, y, theta, vt, wt, change_t):
        x_next = x + (-vt/wt)*sin(theta) + (vt/wt)*sin(theta + wt*change_t)
        y_next = y + (vt/wt)*cos(theta) + (-vt/wt)*cos(theta + wt*change_t)
        theta_next = wt*change_t
        self.x = x_next
        self.y = y_next
        self.theta = theta_next
        return np.vstack((x_next, y_next, theta_next))
        