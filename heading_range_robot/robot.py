from math import sin, cos, pi, sqrt
from os import path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from numpy.random import normal

class Robot():
    def __init__(self, x, y, theta, change_t, alpha1, alpha2, alpha3, alpha4, truth_file=None):
        self._use_truth_data = False
        if truth_file is not None:
            truth_file = path.abspath(path.join(path.dirname(
            __file__), truth_file))
            truth = loadmat(truth_file)
            self.all_x = truth['x'][0]
            self.all_y = truth['y'][0]
            self.all_th = truth['th'][0]
            self.all_v = truth['v'][0]
            self.all_w = truth['om'][0]
            self.all_t = truth['t'][0]
            self._use_truth_data = True
            self._initialize_position_and_heading(self.all_x, self.all_y, self.all_th, change_t)
        else:
            self._initialize_position_and_heading(x, y, theta, change_t)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

    def commanded_translational_velocity(self, t):
        return 2 + 0.5*sin(2*pi*(0.2)*t)

    def commanded_rotational_velocity(self, t):
        return -0.5 + 0.2*cos(2*pi*0.6*t)

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
        vc = self.commanded_translational_velocity(t+self.change_t)
        wc = self.commanded_rotational_velocity(t+self.change_t)
        self.vc = vc
        self.wc = wc
        if self._use_truth_data:
            t = int(t/self.change_t)+1
            self.v = self.all_v[t]
            self.w = self.all_w[t]
            self.x = self.all_x[t]
            self.y = self.all_y[t]
            self.theta = self.all_th[t]
            self.actual_position = np.vstack((self.x, self.y, self.theta))
        else:
            v = vc + self.translational_noise(vc, wc)
            w = wc + self.rotational_noise(vc, wc) 
            self.v = v
            self.w = w
            self.actual_position = self.next_position_from_state(
                self.x, self.y, self.theta, v, w, self.change_t)
            self.x = self.actual_position[0]
            self.y = self.actual_position[1]
            self.theta = self.actual_position[2]

    def next_position_from_state(self, x, y, theta, vt, wt, change_t):
        x_next = x + (-vt/wt)*sin(theta) + (vt/wt)*sin(theta + wt*change_t)
        y_next = y + (vt/wt)*cos(theta) - (vt/wt)*cos(theta + wt*change_t)
        theta_next = theta + wt*change_t
        return np.vstack((x_next, y_next, theta_next))
