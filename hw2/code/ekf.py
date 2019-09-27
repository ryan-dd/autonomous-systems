from math import cos, sin, atan2, exp

import numpy as np

from hw2.code.parameters import *


class EKF:
    def __init__(self):
        self._change_t = 0.1
        self.mean_belief = np.vstack((INITIAL_X, INITIAL_Y, INITIAL_THETA))
        self.covariance_belief = np.eye(3)
        self.Qt = np.eye(2)*np.vstack((STD_DEV_LOCATION_RANGE**2, STD_DEV_LOCATION_BEARING**2))
        self.all_features = np.vstack((LANDMARK_1_LOCATION, LANDMARK_2_LOCATION, LANDMARK_3_LOCATION))

    def prediction_step(self, theta_prev, vc, wc):
        change_t = self._change_t
        theta = theta_prev
        # Jacobian of ut at xt-1
        Gt = np.array([
            [1, 0, -vc/wc*cos(theta) + vc/wc*cos(theta + wc*change_t)],
            [0, 1, -vc/wc*sin(theta) + vc/wc*sin(theta + wc*change_t)],
            [0, 0, 1]])
        # Jacobian to map noise in control space to state space
        Vt = np.array([
        [(-sin(theta) + sin(theta + wc*change_t))/wc, vc*(sin(theta)-sin(theta + wc*change_t))/(wc**2) + (vc*cos(theta + wc*change_t)*change_t)/wc],
        [(-cos(theta) + cos(theta + wc*change_t))/wc, vc*(cos(theta)-cos(theta + wc*change_t))/(wc**2) + (vc*sin(theta + wc*change_t)*change_t)/wc],
        [0, change_t]])

        Mt = np.array([
        [ALPHA1*vc**2 + ALPHA2*wc**2, 0],
        [0, ALPHA3*vc**2 + ALPHA4*wc**2]
        ])

        self.mean_belief = self.mean_belief + np.array([
        [-vc/wc*sin(theta) + vc/wc*sin(theta + wc*change_t)],
        [vc/wc*cos(theta) - vc/wc*cos(theta + wc*change_t)],
        [wc*change_t]
        ])

        self.covariance_belief = Gt @ self.covariance_belief @ Gt.T + Vt @ Mt @ Vt.T

    def measurement_step(self, true_state):
        Qt = self.Qt
        for feature in self.all_features:
            f_x = feature[0]
            f_y = feature[1]
            mean_x = self.mean_belief[0]
            mean_y = self.mean_belief[1]
            mean_theta = self.mean_belief[2]
            # Range and bearing from mean belief
            q = (f_x - mean_x)**2 + (f_y - mean_y)**2
            zti = np.array([
                [np.sqrt(q)],
                [np.arctan2((f_y - mean_y), (f_x - mean_x)) - mean_theta]]).reshape((2,1))
             
            measurement = simulate_measurement(true_state, f_x, f_y)

            Ht = np.array([
                [-(f_x - mean_x)/np.sqrt(q), -(f_y - mean_y)/np.sqrt(q), np.array([0])],
                [(f_y - mean_y)/q, -(f_x - mean_x)/q, np.array([-1])]]).reshape((2,3))
            covariance_belief = self.covariance_belief
            mean_belief = self.mean_belief
            St = Ht @ covariance_belief @ Ht.T + Qt
            Kt = covariance_belief @ Ht.T @ np.linalg.inv(St)
            self.mean_belief = mean_belief + Kt @ (measurement - zti)
            self.covariance_belief = (np.eye(len(Kt)) - Kt @ Ht) @ covariance_belief
        self.kt = Kt
            #pzt = np.linalg.det(2*pi*St)**(-1/2) @ exp(-1/2*(zti - measurement[index]).T @ np.linalg.inv(St) @ (zti - measurement[index]))

def simulate_measurement(true_state, f_x, f_y):
    true_x = true_state[0]
    true_y = true_state[1]
    true_theta = true_state[2]
    q = (f_x - true_x)**2 + (f_y - true_y)**2
    zt = np.array([
                [np.sqrt(q)],
                [np.arctan2((f_y - true_y), (f_x - true_x)) - true_theta]]).reshape((2,1))
    return zt + np.vstack((range_measurement_noise(), bearing_measurement_noise()))

def range_measurement_noise():
    return np.random.normal(0, STD_DEV_LOCATION_RANGE)

def bearing_measurement_noise():
    return np.random.normal(0, STD_DEV_LOCATION_BEARING)