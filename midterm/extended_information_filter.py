from math import cos, sin, atan2, exp, radians

import numpy as np
from tools.wrap import wrap

class EIF:
    def __init__(self, all_features):
        self.mean_belief = np.vstack((-5, 0, radians(90)))
        self.covariance_belief = np.eye(3)
        self.info_matrix = np.linalg.inv(self.covariance_belief)
        self.info_vector =  self.info_matrix @ self.mean_belief
        self.Qt = np.diag((0.2**2, 0.1**2))
        self.Mt = np.diag((0.15**2, 0.1**2))
        self.all_features = all_features

    def prediction_step(self, vc, wc, change_t):
        prev_mean_belief = np.linalg.inv(self.info_matrix) @ self.info_vector
        prev_mean_belief = wrap(prev_mean_belief, index=2)
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

        mean_belief = prev_mean_belief + np.array([
        [vc*cos(theta)*change_t],
        [vc*sin(theta)*change_t],
        [wc*change_t]
        ])
        mean_belief = wrap(mean_belief, index=2)

        Mt = self.Mt 
        self.info_matrix = np.linalg.inv(
            Gt @ np.linalg.inv(self.info_matrix) @ Gt.T + Vt @ Mt @ Vt.T
            )
        self.info_vector = self.info_matrix @ mean_belief
        
    def measurement_step(self, feature_measurements):
        Qt = self.Qt
        for index, feature in enumerate(self.all_features):
            mean_belief = np.linalg.inv(self.info_matrix) @ self.info_vector
            mean_belief = wrap(mean_belief, index=2)
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
            h = wrap(h, index=1)
            
            measurement = feature_measurements[:, index].reshape((2,1))
            measurement = wrap(measurement, index=1)
            
            Ht = np.array([
                [-(f_x - mean_x)/np.sqrt(q), -(f_y - mean_y)/np.sqrt(q), np.array([0])],
                [(f_y - mean_y)/q, -(f_x - mean_x)/q, np.array([-1])]]).reshape((2,3))
                
            self.info_matrix = self.info_matrix + Ht.T @ np.linalg.inv(Qt) @ Ht
            self.info_vector = (
                self.info_vector + 
                Ht.T @ np.linalg.inv(Qt) @ 
                (measurement - h + (Ht @ mean_belief))
            )
        self.covariance_belief = np.linalg.inv(self.info_matrix)
        self.mean_belief = self.covariance_belief @ self.info_vector
        self.mean_belief = wrap(self.mean_belief, index=2)
