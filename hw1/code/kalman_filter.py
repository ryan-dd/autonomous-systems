import numpy as np
from numpy import matmul, identity
from numpy.linalg import multi_dot, inv
from control import StateSpace, matlab

from hw1.code.parameters import *


m = VEHICLE_MASS
b = LINEAR_DRAG_COEFFICIENT

A = np.array([[0, 1], [0, -b/m]])
Atranspose = np.transpose(A)
B = np.array([[0], [1/m]])
C = np.array([[1, 0]])
Ctranspose = np.transpose(C)
D = np.array([[0]])
R = np.array([[MEASUREMENT_NOISE_COVARIANCE]])
Q = np.array([[PROCESS_NOISE_COVARIANCE_POSITION],[PROCESS_NOISE_COVARIANCE_VELOCITY]])
state_space = StateSpace(A, B, C, D)
discrete_time = matlab.c2d(state_space, SAMPLE_PERIOD)

def kalman_filter(mean_prev, variance_prev, ut, zt, A, B, C, Atranspose, Ctranspose, R, Q):
    mean_belief = A*mean_prev + B*ut
    variance_belief = A*variance_prev*Atranspose + R
    
    Kt = find_Kt(variance_belief, C, Ctranspose, Q)
    
    corrected_mean_belief = mean_belief + matmul(Kt, (zt - matmul(C, mean_belief)))
    
    first = matmul(Kt, C)
    identity_matrix = identity(first.size())
    corrected_var_belief = matmul((identity_matrix-first), variance_belief)
    return corrected_mean_belief, corrected_var_belief

def find_Kt(variance_belief, C, Ctranspose, Q):
    first = matmul(variance_belief, Ctranspose)
    second = inv(multi_dot([C, variance_belief, Ctranspose]) + Q)
    return matmul(first, second)
