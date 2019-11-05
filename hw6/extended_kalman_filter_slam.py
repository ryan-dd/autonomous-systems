from math import cos, sin, atan2, exp

import numpy as np
from scipy.linalg import block_diag

from heading_range_robot.parameters import *


class EKF:
    def __init__(self, sample_period):
        self._change_t = sample_period
        
        self.Qt = np.diag((STD_DEV_LOCATION_RANGE**2, STD_DEV_LOCATION_BEARING**2))
        self.all_features = np.vstack((LANDMARK_1_LOCATION, LANDMARK_2_LOCATION, LANDMARK_3_LOCATION))
        self.n_features = self.all_features.shape[0]

        robot_state_mean_belief = np.vstack((INITIAL_X, INITIAL_Y, INITIAL_THETA))
        landmark_state_mean_belief = np.zeros((self.n_features*2,1))
        self.mean_belief = np.append(robot_state_mean_belief, landmark_state_mean_belief, axis=0)

        robot_state_covariances = np.eye(3)
        landmark_covariances = np.ones((self.n_features*2, self.n_features*2))*100000
        self.covariance_belief = block_diag(robot_state_covariances, landmark_covariances)
        
        self.initialized = [False]*self.n_features

    def prediction_step(self, vc, wc):       
        Fx = np.append(np.eye(3), np.zeros((3, self.n_features*2)), axis=1)
        change_t = self._change_t
        theta = self.mean_belief[2]

        self.mean_belief = self.mean_belief + Fx.T @ np.array([
        [-vc/wc*sin(theta) + vc/wc*sin(theta + wc*change_t)],
        [vc/wc*cos(theta) - vc/wc*cos(theta + wc*change_t)],
        [wc*change_t]
        ])

        # Jacobian of ut at xt-1
        Gt = np.eye(9) + Fx.T @ np.array([
            [0, 0, -vc/wc*cos(theta) + vc/wc*cos(theta + wc*change_t)],
            [0, 0, -vc/wc*sin(theta) + vc/wc*sin(theta + wc*change_t)],
            [0, 0, 0]]) @ Fx
        # Jacobian to map noise in control space to state space
        Vt = np.array([
        [(-sin(theta) + sin(theta + wc*change_t))/wc, vc*(sin(theta)-sin(theta + wc*change_t))/(wc**2) + (vc*cos(theta + wc*change_t)*change_t)/wc],
        [(-cos(theta) + cos(theta + wc*change_t))/wc, vc*(cos(theta)-cos(theta + wc*change_t))/(wc**2) + (vc*sin(theta + wc*change_t)*change_t)/wc],
        [0, change_t]])

        Mt = np.array([
        [ALPHA1*vc**2 + ALPHA2*wc**2, 0],
        [0, ALPHA3*vc**2 + ALPHA4*wc**2]
        ])
        Rt = Vt @ Mt @ Vt.T

        self.covariance_belief = Gt @ self.covariance_belief @ Gt.T + Fx.T @ Rt @ Fx

    def measurement_step(self, true_state):
        Qt = self.Qt
        for index, feature in enumerate(self.all_features):
            f_x = feature[0]
            f_y = feature[1]
            mean_x = self.mean_belief[0]
            mean_y = self.mean_belief[1]
            mean_theta = self.mean_belief[2]
            measurement = simulate_measurement(true_state, f_x, f_y)
            if not self.initialized[index]:
                r = measurement[0]
                phi = measurement[1]
                self.mean_belief[2*index + 3] = r*cos(phi + mean_theta) + mean_x
                self.mean_belief[2*index + 4] = r*sin(phi + mean_theta) + mean_y
                self.initialized[index] = True
            # Range and bearing from mean belief
            delta_x = f_x - mean_x
            delta_y = f_y - mean_y
            delta = np.vstack((delta_x, delta_y))
            q = (delta.T @ delta).flatten()
            zti = np.array([
                [np.sqrt(q)],
                [np.arctan2((f_y - mean_y), (f_x - mean_x)) - mean_theta]]).reshape((2,1))
            left = np.append(np.eye(3), np.zeros((2, 3)), axis=0)
            middle1 = np.zeros((5, 2*(index+1)-2))
            middle2 = np.append(np.zeros((3,2)), np.eye(2), axis=0)
            right = np.zeros((5, 2*self.n_features-2*(index+1)))
            Fxj = np.concatenate((left,middle1,middle2,right), axis=1)
            sqrtq = np.sqrt(q)
            Ht = 1/q * np.array(
                [[-sqrtq*delta_x, -sqrtq*delta_y, np.array([0]), sqrtq*delta_x, sqrtq*delta_y],
                [delta_y, -delta_x, -q, -delta_y, delta_x]]).reshape((2, 5)) @ Fxj
           
            covariance_belief = self.covariance_belief
            mean_belief = self.mean_belief
            St = Ht @ covariance_belief @ Ht.T + Qt
            Kt = covariance_belief @ Ht.T @ np.linalg.inv(St)
            self.mean_belief = mean_belief + Kt @ (measurement - zti)
            self.covariance_belief = (np.eye(len(Kt)) - Kt @ Ht) @ covariance_belief
        self.kt = Kt

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