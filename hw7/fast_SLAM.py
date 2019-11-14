from random import random as rand
from math import pi, sqrt, exp, sin, cos

import numpy as np
from heading_range_robot.parameters import ALPHA1, ALPHA2, ALPHA3, ALPHA4, INITIAL_X, INITIAL_Y, INITIAL_THETA, STD_DEV_LOCATION_RANGE, STD_DEV_LOCATION_BEARING, LANDMARKS
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag

from tools.wrap import wrap


class FastSLAM:
    def __init__(self, sample_period, number_of_particles, landmarks, vision_angle=2*pi):
        self._change_t = sample_period
        self.Qt = np.eye(2) * np.vstack(
            (STD_DEV_LOCATION_RANGE**2, STD_DEV_LOCATION_BEARING**2))
        self.all_features = landmarks
        self.n_features = len(landmarks)
        self.n = number_of_particles
        self.particles = self.create_random_particles(number_of_particles)
        self.vision_angle = vision_angle

    def create_random_particles(self, n, initial_cov=1000000):
        all_particles = []
        landmark_state_mean_belief = np.zeros((self.n_features*2,1))
        robot_state_covariances = np.diag((0, 0, 0))
        landmark_covariances = np.diag(([1e10]*self.n_features*2))
        
        self.initialized = [False]*self.n_features
        self.R = np.diag((0.2**2, 0.2**2, 0.05**2))
        for _ in range(n):
            particle_x = rand()*40-20
            particle_y = rand()*40-20
            particle_theta = rand()*2*pi
            particle_mean_belief = [particle_x, particle_y, particle_theta]
            mean_belief = np.append(particle_mean_belief, landmark_state_mean_belief, axis=0)
            covariance_belief = block_diag(robot_state_covariances, landmark_covariances)
            # Initialize covariance matrices for landmarks
            all_particles.append((mean_belief, covariance_belief))
        return np.array(all_particles)

    def particle_filter(self, robot):
        vc = robot.vc
        wc = robot.wc
        true_state = robot.actual_position
        
        updated_particles = {}
        measurements_from_robot = []
        for index, feature in enumerate(self.all_features):
            f_x = feature[0]
            f_y = feature[1]
            if self.is_not_valid_measurement(f_x, f_y, true_state):
                continue
            measurement = simulate_measurement(robot.actual_position, f_x, f_y)
            measurements_from_robot.append((index, measurement))
        all_weights = []
        for particle in self.particles:
            v_perturbed = vc + robot.translational_noise(vc, wc)
            w_perturbed = wc + robot.rotational_noise(vc, wc)
            new_particle = robot.next_position_from_state(particle[0], particle[1], particle[2], v_perturbed, w_perturbed, self._change_t)
            # DO ekf_slam for the particle as if it was the actual robot

            weight = self.probability_of_measurement(new_particle, measurements_from_robot)
            all_weights.append(weight)
            updated_particles.append(new_particle)    
        all_weights = all_weights / np.sum(all_weights)
        self.resample_particles(updated_particles, all_weights)
        self.mean_belief = np.mean(self.particles, axis=0)
        self.covariance_belief = np.var(self.particles, axis=0)

    def ekf_slam_prediction_step(self, vc, wc, mean_belief, covariance_belief):       
        Fx = np.append(np.eye(3), np.zeros((3, self.n_features*2)), axis=1)
        change_t = self._change_t
        theta = np.copy(self.mean_belief[2])

        mean_belief = mean_belief + Fx.T @ np.array([
        [-vc/wc*sin(theta) + vc/wc*sin(theta + wc*change_t)],
        [vc/wc*cos(theta) - vc/wc*cos(theta + wc*change_t)],
        [wc*change_t]
        ])

        # Jacobian of ut at xt-1
        Gt = np.eye(3+2*self.n_features) + Fx.T @ np.array([
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
        self.R = Vt @ Mt @ Vt.T
        covariance_belief = Gt @ self.covariance_belief @ Gt.T + Fx.T @ self.R @ Fx
        return (mean_belief, covariance_belief)

    def ekf_slam_measurement_step(self, true_state):
        Qt = self.Qt
        for index, feature in enumerate(self.all_features):
            f_x = feature[0]
            f_y = feature[1]
            mean_x = self.mean_belief[0]
            mean_y = self.mean_belief[1]
            mean_theta = self.mean_belief[2]
            angle_to_check = wrap(np.arctan2((f_y-true_state[1]), (f_x-true_state[0])) - wrap(true_state[2]))
            if abs(angle_to_check) > self.vision_angle:
                continue

            measurement = simulate_measurement(true_state, f_x, f_y)
            if not self.initialized[index]:
                r = measurement[0]
                phi = measurement[1]
                self.mean_belief[2*index + 3] = r*cos(phi + mean_theta) + mean_x
                self.mean_belief[2*index + 4] = r*sin(phi + mean_theta) + mean_y
                self.initialized[index] = True
            # Range and bearing from mean belief
            delta_x = self.mean_belief[2*index + 3] - mean_x
            delta_y = self.mean_belief[2*index + 4] - mean_y
            delta = np.vstack((delta_x, delta_y))
            q = (delta_x)**2 + (delta_y)**2
            zti = np.array([
                [np.sqrt(q)],
                [np.arctan2((delta_y), (delta_x)) - mean_theta]]).reshape((2,1))
            left = np.append(np.eye(3), np.zeros((2, 3)), axis=0)
            middle1 = np.zeros((5, 2*(index+1)-2))
            middle2 = np.append(np.zeros((3,2)), np.eye(2), axis=0)
            right = np.zeros((5, 2*self.n_features-2*(index+1)))
            Fxj = np.concatenate((left, middle1, middle2, right), axis=1)
            sqrtq = np.sqrt(q)
            Ht = 1/q * np.array(
                [[-sqrtq*delta_x, -sqrtq*delta_y, np.array([0]), sqrtq*delta_x, sqrtq*delta_y],
                [delta_y, -delta_x, -q, -delta_y, delta_x]]).reshape((2, 5)) @ Fxj
           
            covariance_belief = self.covariance_belief
            mean_belief = self.mean_belief
            St = Ht @ covariance_belief @ Ht.T + Qt
            Kt = covariance_belief @ Ht.T @ np.linalg.inv(St)
            self.mean_belief = mean_belief + Kt @ wrap((measurement - zti), index=1)
            self.covariance_belief = (np.eye(len(Kt)) - Kt @ Ht) @ covariance_belief
            self.kt = Kt

    def is_not_valid_measurement(self, f_x, f_y, true_state):
        angle_to_check = wrap(np.arctan2((f_y-true_state[1]), (f_x-true_state[0])) - wrap(true_state[2]))
        return abs(angle_to_check) > self.vision_angle
    
    def resample_particles(self, updated_particles, all_weights):
        final_particles = []
        minv = 1/len(updated_particles)
        r = rand()*minv
        c = all_weights[0]
        i = 0
        for m in range(len(updated_particles)):
            U = r + (m*minv)
            while U > c:
                i += 1
                c = c + all_weights[i]
            final_particles.append(updated_particles[i])
        self.particles = final_particles


    def probability_of_measurement(self, new_particle, measurement_from_robot):
        weight = 1.0
        for feature_tuple in enumerate(measurement_from_robot):
            index = feature_tuple[0]
            feature = feature_tuple[1]
            f_x = feature[0]
            f_y = feature[1]
            mean_x = new_particle[0]
            mean_y = new_particle[1]
            mean_theta = new_particle[2]
            # Range and bearing from mean belief
            q = (f_x - mean_x)**2 + (f_y - mean_y)**2
            particle_range_bearing = np.array([
                [np.sqrt(q)],
                [np.arctan2((f_y - mean_y), (f_x - mean_x)) - mean_theta]]).reshape((2,1))
            error = particle_range_bearing - measurement_from_robot[index]
            p_r = prob_normal_distribution(error[0], STD_DEV_LOCATION_RANGE**2)
            p_b = prob_normal_distribution(error[1], STD_DEV_LOCATION_BEARING**2)
            weight *= p_r*p_b
        return weight

def prob_normal_distribution(mean, variance):
    return 1/sqrt(2*pi*variance)*exp(-1/2*mean**2/variance)

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
