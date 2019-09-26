from math import cos, sin, atan2, sqrt, pi, exp

import numpy as np

from hw2.code.parameters import *


def main():
    x = X_INITIAL
    y = Y_INITIAL
    theta = theta_initial
    mean_belief = np.vstack(x, y, theta)
    time_steps = list(range(TOTAL_TIME/SAMPLE_PERIOD))
    for time_step in time_steps:
        pass
    
def ekf_localization_with_known_correspondences(mean_belief_prev, covariance_belief_prev, control, measurement, correspondences, map):
    all_features = np.vstack(LANDMARK_1_LOCATION, LANDMARK_2_LOCATION, LANDMARK_3_LOCATION)
    theta = mean_belief_prev[2]
    vt = 0
    wt = 0
    change_t = SAMPLE_PERIOD
    Gt = np.array([
        [1, 0, -vt/wt*cos(theta) + vt/wt*cos(theta + wt*change_t)],
        [0, 1, -vt/wt*sin(theta) + vt/wt*sin(theta + wt*change_t)],
        [0, 0, 1]])
    Vt = np.array([
        [(-sin(theta) + sin(theta + wt*change_t))/wt, vt*(sin(theta)-sin(theta + wt*change_t))/(wt**2) + (vt*cos(theta + wt*change_t)*change_t)/wt],
        [(-cos(theta) + cos(theta + wt*change_t))/wt, vt*(cos(theta)-cos(theta + wt*change_t))/(wt**2) + (vt*sin(theta + wt*change_t)*change_t)/wt]
        ])
    Mt = np.array([
        [ALPHA1*vt**2 + ALPHA2*wt**2, 0],
        [0, ALPHA3*vt**2 + ALPHA4*wt**2]
        ])
    mean_belief = mean_belief_prev + np.array([
        [-vt/wt*sin(theta) + vt/wt*sin(theta + wt*change_t)],
        [vt/wt*cos(theta) - vt/wt*cos(theta + wt*change_t)],
        [wt*change_t]
        ])
    covariance_belief = Gt @ covariance_belief_prev @ Gt.T + Vt @ Mt @ Vt.T
    Qt = np.array([
        [sigma_r**2, 0],
        [0, sigma_theta**2]])
    for index, feature in enumerate(all_features):
        j = feature
        mjs = 0
        q = (feature[0]- mean_belief[0])**2 + (feature[1]- mean_belief[1])**2
        zti = np.array([
            [sqrt(q)],
            [atan2((feature[1]- mean_belief[1]), feature[0]- mean_belief[0])-mean_belief[2])],
            [mjs]])
        Ht = np.array([
            [(-mjx - mean_belief[0])/sqrt(q), (-mjy - mean_belief[1])/sqrt(q), 0],
            [(-mjy - mean_belief[1])/q, (-mjx - mean_belief[0])/q, -1],
            [0, 0, 0]])
        St = Ht @ covariance_belief @ Ht.T + Qt
        Kt = covariance_belief @ Ht.T @ np.linalg.inv(St)
        mean_belief = mean_belief + Kt*(zti - measurement[index])
        covariance_belief = (np.eye(len(Kt))-Kt @ Ht) @ covariance_belief
        pzt = np.linalg.det(2*pi*Sti)**(-1/2) @ exp(-1/2*(zti - measurement[index]).T @ np.linalg.inv(Sti) @ (zti - measurement[index]))
        return mean_belief, covariance_belief, pzt

if __name__ == "__main__":
    main()