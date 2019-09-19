import numpy as np
from numpy import identity, dot
from numpy.linalg import multi_dot, inv
from scipy.signal import cont2discrete
from matplotlib import pyplot as plt

from hw1.code.parameters import *
from tools.plot_helper import create_plotting_parameters_for_normal_distribution


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
    mean_belief = np.array([[0], [0]])
    variance_belief = np.array([[1, 0], [0, 0.1]])
    # Initial conditions
    xt_current = 0
    vt_current = 0

    # Initialize histories for plotting
    all_xt = [xt_current]
    all_vt = [vt_current]
    all_mean_belief = [mean_belief]
    all_variance_belief = [variance_belief]
    all_kt = [0]

    # Execute kalman filter for 50 seconds
    time_steps = int(50/SAMPLE_PERIOD)
    for time_step in range(time_steps):
        # Calculate input
        ut = thrust((time_step+1)*SAMPLE_PERIOD)
        # Update true state
        state = np.array([[xt_current], [vt_current]])
        state_plus_one = dot(A, state)+dot(B, ut)
        xt_current = state_plus_one[0][0]
        vt_current = state_plus_one[1][0]

        # Update sensor measurement of state
        zt = xt_current
        # Calculate beliefs
        mean_belief, variance_belief, kt = kalman_filter(
            mean_belief, variance_belief, ut, zt, A, B, C, Atranspose, Ctranspose, R, Q)

        # Record outputs for plotting
        all_xt.append(xt_current)
        all_vt.append(vt_current)
        all_mean_belief.append(mean_belief)
        all_variance_belief.append(variance_belief)
        all_kt.append(kt)
    plot_everything(all_xt, all_vt, all_mean_belief, all_variance_belief, all_kt)


def kalman_filter(mean_prev, variance_prev, ut, zt, A, B, C, Atranspose, Ctranspose, R, Q):
    # Prediction step (from controls)
    mean_belief = dot(A, mean_prev) + dot(B, ut)
    variance_belief = multi_dot([A, variance_prev, Atranspose]) + R

    # Correction step (from measurements)
    Kt = find_Kt(variance_belief, C, Ctranspose, Q)

    corrected_mean_belief = mean_belief + dot(Kt, (zt - dot(C, mean_belief)))

    first = dot(Kt, C)
    identity_matrix = identity(len(first))
    corrected_var_belief = dot((identity_matrix - first), variance_belief)
    return corrected_mean_belief, corrected_var_belief, Kt


def find_Kt(variance_belief, C, Ctranspose, Q):
    return multi_dot([
        variance_belief,
        Ctranspose,
        inv(multi_dot([C, variance_belief, Ctranspose]) + Q)
    ])


def plot_everything(all_xt, all_vt, all_mean_belief, all_variance_belief, all_kt):
    time_steps = list(range(len(all_xt)))
    time_steps_in_seconds = [t*SAMPLE_PERIOD for t in time_steps]

    all_mean_belief = np.array(all_mean_belief)
    all_variance_belief = np.array(all_variance_belief)

    mean_beliefs_about_position = all_mean_belief[:, 0]
    mean_beliefs_about_velocity = all_mean_belief[:, 1]

    # Add static plots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.6, wspace=0.6)

    ax1.plot(time_steps_in_seconds, all_xt)
    ax1.plot(time_steps_in_seconds, mean_beliefs_about_position)
    ax1.set_title("Position vs Mean belief about Position")
    ax1.legend(["Actual Position", "Mean Position Belief"])

    position_error = [abs(xt-mean_beliefs_about_position[i]) for i, xt in enumerate(all_xt)]
    ax2.plot(time_steps_in_seconds, position_error)
    ax2.set_title("Error from position and mean belief")

    ax3.plot(time_steps_in_seconds, all_vt)
    ax3.plot(time_steps_in_seconds, mean_beliefs_about_velocity)
    ax3.set_title("Velocity vs Mean Belief about Velocity")
    ax3.legend(["Actual Velocity", "Mean Velocity Belief"])

    velocity_error = [abs(vt-mean_beliefs_about_velocity[i]) for i, vt in enumerate(all_vt)]
    ax4.plot(time_steps_in_seconds, velocity_error)
    ax4.set_title("Error from velocity and mean belief")

    plt.show()
    for index in range(time_steps):
        pass


if __name__ == "__main__":
    main()
