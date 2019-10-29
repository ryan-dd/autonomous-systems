from math import cos, sin, atan2, exp

import numpy as np

from heading_range_robot.parameters import *


class EIF:
    def __init__(self, sample_period):
        self._change_t = sample_period
        self.mean_belief = np.vstack((INITIAL_X, INITIAL_Y, INITIAL_THETA))
        self.covariance_belief = np.eye(3)
        self.info_matrix = np.linalg.inv(self.covariance_belief)
        self.info_vector =  self.info_matrix @ self.mean_belief
        self.Qt = np.diag((0.2, 0.1))
        self.all_features = np.vstack((LANDMARK_1_LOCATION, LANDMARK_2_LOCATION, LANDMARK_3_LOCATION))

    def prediction_step(self, vc, wc):
        change_t = self._change_t
        prev_mean_belief = np.linalg.inv(self.info_matrix) @ self.info_vector
        theta = prev_mean_belief[2]
        # Jacobian of ut at xt-1
        Gt = np.array([
            [1, 0, -vc*sin(theta)*change_t],
            [0, 1, vc*cos(theta)*change_t],
            [0, 0, 1]])
        # Jacobian to map noise in control space to state space
        Vt = np.array([
        [cos(theta)*change_t, 0],
        [sin(theta)*change_t, 0],
        [0, change_t]])

        Mt = np.array([
        [0.15, 0],
        [0, 0.1]
        ])
        
        mean_belief = prev_mean_belief + np.array([
        [vc*cos(theta)*change_t],
        [vc*sin(theta)*change_t],
        [wc*change_t]
        ])

        self.info_matrix = np.linalg.inv(Gt @ np.linalg.inv(self.info_matrix) @ Gt.T + Vt @ Mt @ Vt.T)
        self.info_vector = self.info_matrix @ mean_belief
        
    def measurement_step(self, true_state):
        Qt = self.Qt
        for feature in self.all_features:
            mean_belief = np.linalg.inv(self.info_matrix) @ self.info_vector
            f_x = feature[0]
            f_y = feature[1]
            mean_x = mean_belief[0]
            mean_y = mean_belief[1]
            mean_theta = mean_belief[2]
            # Range and bearing from mean belief
            q = (f_x - mean_x)**2 + (f_y - mean_y)**2
            h = np.array([
                [np.sqrt(q)],
                [np.arctan2((f_y - mean_y), (f_x - mean_x)) - mean_theta]]).reshape((2,1))
            
            measurement = simulate_measurement(true_state, f_x, f_y)

            Ht = np.array([
                [-(f_x - mean_x)/np.sqrt(q), -(f_y - mean_y)/np.sqrt(q), np.array([0])],
                [(f_y - mean_y)/q, -(f_x - mean_x)/q, np.array([-1])]]).reshape((2,3))
            self.info_matrix = self.info_matrix @ Ht.T @ np.linalg.inv(Qt) @ Ht
            self.info_vector = self.info_vector + Ht.T @ np.linalg.inv(Qt) @ (measurement - h + Ht @ mean_belief)
        self.covariance_belief = np.linalg.inv(self.info_matrix)
        self.mean_belief = self.covariance_belief @ self.info_vector

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