from math import cos, sin, atan2, exp, sqrt

import numpy as np

from hw2.code.parameters import ALPHA1, ALPHA2, ALPHA3, ALPHA4, INITIAL_X, INITIAL_Y, INITIAL_THETA, STD_DEV_LOCATION_RANGE, STD_DEV_LOCATION_BEARING, LANDMARKS


class UKF:
    def __init__(self, sample_period):
        self._change_t = sample_period
        self.mean_belief = np.vstack((INITIAL_X, INITIAL_Y, INITIAL_THETA))
        self.covariance_belief = np.eye(3)
        self.Qt = np.eye(2) * np.vstack(
            (STD_DEV_LOCATION_RANGE**2, STD_DEV_LOCATION_BEARING**2))
        self.all_features = LANDMARKS
        self.n = 7
        kappa = 3
        self.alpha = 0.25
        self.beta = 2
        self.lambda_ = self.alpha**2*(self.n+kappa)-self.n
        self.gamma = sqrt(self.n + self.lambda_)

    def prediction_step(self, theta_prev, vc, wc, robot):
        change_t = self._change_t
        n = self.n
        gamma = self.gamma
        alpha = self.alpha
        beta = self.beta

        chi_a = self.get_sigma_points(vc, wc, gamma)
        chi_a_bar = []
        for col in chi_a.T:
            new_col = robot.next_position_from_state(col[0], col[1], col[2], vc, wc, change_t)
            chi_a_bar.append(new_col)
        chi_a_bar = np.array(chi_a_bar) # Note: each row represents a point instead of a column
        mean_belief = np.zeros((3, 1))
        covariance_belief = np.zeros((3, 3))
        for index, sigma_point in enumerate(chi_a_bar):
            if index == 0:
                wm = gamma/(n+gamma)
            else:
                wm = 1/(2*(n+gamma))
            mean_belief = np.add(mean_belief, wm*sigma_point)

        for index, sigma_point in enumerate(chi_a_bar):
            if index == 0: 
                wc = gamma/(n+gamma) + (1-alpha**2+beta)
            else:
                wc = 1/(2*(n+gamma))
            covariance_belief = np.add(covariance_belief, wc*((sigma_point - mean_belief) @ (sigma_point - mean_belief).T))
        self.chi_a = chi_a_bar
        self.mean_belief = mean_belief
        self.covariance_belief = covariance_belief

    def measurement_step(self, true_state, robot):
        for index, feature in enumerate(self.all_features):
            if index == 0:
                chi_a = self.chi_a
            else:
                chi_a = self.get_sigma_points(robot.vc, robot.wc, self.gamma)
            Zt = []
            for sigma_point in chi_a:
                f_x = feature[0]
                f_y = feature[1]
                mean_x = sigma_point[0]
                mean_y = sigma_point[1]
                mean_theta = sigma_point[2]
                # Predicted range and bearing from sigma point
                q = (f_x - mean_x)**2 + (f_y - mean_y)**2
                Zti = np.array([
                    [np.sqrt(q)],
                    [np.arctan2((f_y - mean_y), (f_x - mean_x)) - mean_theta]]).reshape((2, 1))
                Zt.append(Zti)
                
            zt_hat = np.zeros((2,1))
            for index, Zti in enumerate(Zt):
                wm = self.get_wm(index)
                zt_hat = np.add(zt_hat, wm*Zti)

            St = np.array((2,2))
            for index, Zti in enumerate(Zt):
                wc = self.get_wc(index)
                St = np.add(St, wc*((Zti - zt_hat) @ (Zti - zt_hat).T))

            covariance_belief = np.zeros((7,2))
            for index, sigma_point in enumerate(chi_a):
                wc = self.get_wc(index)
                covariance_belief = np.add(covariance_belief, wc*((sigma_point - self.mean_belief) @ (Zt[index] - zt_hat).T))
            Kt = covariance_belief @ np.linalg.inv(St)
            measurement = simulate_measurement(true_state, f_x, f_y)

            mean_belief = np.add(self.mean_belief, Kt @ (measurement - zt_hat))
            covariance_belief = self.covariance_belief - Kt @ St @ Kt.T
            self.mean_belief = mean_belief
            self.covariance_belief = covariance_belief
            self.kt = Kt

    def get_sigma_points(self, vc, wc, gamma):
        Mt = np.array([
            [ALPHA1*vc**2 + ALPHA2*wc**2, 0],
            [0, ALPHA3*vc**2 + ALPHA4*wc**2]
        ])
        extended_mean = np.concatenate([self.mean_belief.T, np.zeros((1,4))], axis=1).T
        first_row = np.concatenate([self.covariance_belief, np.zeros((3,4))], axis=1)
        second_row = np.concatenate([np.zeros((2,3)), Mt, np.zeros((2,2))], axis=1)
        third_row = np.concatenate([np.zeros((2,5)), self.Qt], axis=1)
        extended_covariance = np.concatenate([first_row, second_row, third_row], axis=0)
        
        Chi_a = np.concatenate(
            [extended_mean, 
            np.add(extended_mean, gamma*np.linalg.cholesky(extended_covariance)),
            np.add(extended_mean, gamma*np.linalg.cholesky(extended_covariance))
            ], axis=1)
        return Chi_a

    def get_wc(self, index):
        gamma = self.gamma
        alpha = self.alpha
        beta = self.beta
        n = self.n 
        if index == 0: 
            wc = gamma/(n+gamma) + (1-alpha**2+beta)
        else:
            wc = 1/(2*(n+gamma))
        return wc

    def get_wm(self, index):
        gamma = self.gamma
        n = self.n 
        if index == 0:
            wm = gamma/(n+gamma)
        else:
            wm = 1/(2*(n+gamma))
        return wm   

def simulate_measurement(true_state, f_x, f_y):
    true_x = true_state[0]
    true_y = true_state[1]
    true_theta = true_state[2]
    q = (f_x - true_x)**2 + (f_y - true_y)**2
    zt = np.array([
        [np.sqrt(q)],
        [np.arctan2((f_y - true_y), (f_x - true_x)) - true_theta]]).reshape((2, 1))
    return zt + np.vstack((range_measurement_noise(), bearing_measurement_noise()))


def range_measurement_noise():
    return np.random.normal(0, STD_DEV_LOCATION_RANGE)


def bearing_measurement_noise():
    return np.random.normal(0, STD_DEV_LOCATION_BEARING)
