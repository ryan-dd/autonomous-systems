import numpy as np
from numpy import matmul, identity
from numpy.linalg import multi_dot, inv
from scipy.signal import cont2discrete

from hw1.code.parameters import *

def main():
    # Define state space 
    m = VEHICLE_MASS
    b = LINEAR_DRAG_COEFFICIENT

    A = np.array([[0, 1], [0, -b/m]])
    B = np.array([[0], [1/m]])
    C = np.array([[1, 0]])
    D = np.array([[0]])
    Q = np.array([[MEASUREMENT_NOISE_COVARIANCE]])
    R = np.array([[PROCESS_NOISE_COVARIANCE_POSITION, 0],
                  [0, PROCESS_NOISE_COVARIANCE_VELOCITY]])
    state_space = cont2discrete((A, B, C, D), dt=SAMPLE_PERIOD)
    A = state_space[0]
    B = state_space[1]
    C = state_space[2]
    D = state_space[3]
    Atranspose = A.transpose()
    Ctranspose = C.transpose()
    
    # Initial beliefs
    mean_belief = np.array([[0],[0]])
    variance_belief = np.array([[1, 0],[0, 0.1]])
    # Initial conditions
    xt_current = 0
    vt_current = 0
    all_xt = [xt_current]
    all_vt = [vt_current]

    # Execute kalman filter for 50 seconds
    time_steps = int(50/SAMPLE_PERIOD)
    for index in range(time_steps):
        # Calculate input
        ut = thrust(index)
        # Sensor measurement of state 
        zt = xt_current
        # Calculate beliefs
        mean_belief, variance_belief = kalman_filter(mean_belief, variance_belief, ut, zt, A, B, C, Atranspose, Ctranspose, R, Q )
        
        # Update true state
        state = np.array([[xt_current],[vt_current]])
        state_plus_one = np.dot(A, state)+np.dot(B, ut)
        xt_current = state_plus_one[0,0]
        vt_current = state_plus_one[1,0]

        all_xt.append(xt_current)
        all_vt.append(vt_current)

def kalman_filter(mean_prev, variance_prev, ut, zt, A, B, C, Atranspose, Ctranspose, R, Q):
    # Prediction step (from controls)
    mean_belief = A*mean_prev + B*ut
    variance_belief = A*variance_prev*Atranspose + R

    # Correction step (from measurements)
    Kt = find_Kt(variance_belief, C, Ctranspose, Q)

    corrected_mean_belief = mean_belief + \
        matmul(Kt, (zt - matmul(C, mean_belief)))

    first = matmul(Kt, C)
    identity_matrix = identity(len(first))
    corrected_var_belief = matmul((identity_matrix - first), variance_belief)
    return corrected_mean_belief, corrected_var_belief


def find_Kt(variance_belief, C, Ctranspose, Q):
    first = matmul(variance_belief, Ctranspose)
    second = inv(multi_dot([C, variance_belief, Ctranspose]) + Q)
    return matmul(first, second)


if __name__ == "__main__":
    main()
