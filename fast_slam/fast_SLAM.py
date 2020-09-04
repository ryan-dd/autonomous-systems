from random import random as rand
from math import pi, sqrt, exp, sin, cos

import numpy as np
from heading_range_robot.parameters import ALPHA1, ALPHA2, ALPHA3, ALPHA4, INITIAL_X, INITIAL_Y, INITIAL_THETA, STD_DEV_LOCATION_RANGE, STD_DEV_LOCATION_BEARING, LANDMARKS
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag, inv
import matplotlib.pyplot as plt
from copy import deepcopy

from tools.wrap import wrap


class FastSLAM:
    def __init__(self, sample_period, number_of_particles, landmarks, vision_angle=2*pi):
        self._change_t = sample_period
        self.Qt = np.diag(
            (STD_DEV_LOCATION_RANGE**2, STD_DEV_LOCATION_BEARING**2))
        self.all_features = landmarks
        self.n_features = len(landmarks)
        self.n = number_of_particles
        self.particles = self.create_random_particles(number_of_particles)
        self.vision_angle = vision_angle
        self.R = np.diag((0.2**2, 0.2**2, 0.05**2))


    def create_random_particles(self, n, initial_cov=1000000):
        all_particles = []
        landmark_state_mean_belief = np.random.random((self.n_features,2))
        landmark_covariances = np.array([np.diag((1e10,1e10))]*self.n_features)
        initialized = [False]*self.n_features
        for _ in range(n):
            particle_x = rand()*40-20
            particle_y = rand()*40-20
            particle_theta = rand()*2*pi
            particle_x = 0
            particle_y = 5
            particle_theta = 0
            particle_position = np.array([particle_x, particle_y, particle_theta])[:, np.newaxis]
            all_particles.append([particle_position, np.copy(landmark_state_mean_belief), np.copy(landmark_covariances), deepcopy(initialized)])
        return all_particles

    def particle_filter(self, robot, first_step):
        vc = robot.vc
        wc = robot.wc
        true_state = robot.actual_position    
        updated_particles = []
        # Simulate measurements
        measurements_from_robot = []
        for index, feature in enumerate(self.all_features):
            f_x = feature[0]
            f_y = feature[1]
            if self.is_not_valid_measurement(f_x, f_y, true_state):
                continue
            measurement = simulate_measurement(robot.actual_position, f_x, f_y)
            measurements_from_robot.append((index, measurement))
        # Update covariance and calculate weights
        all_weights = []
        for particle in self.particles:
            v_perturbed = vc 
            w_perturbed = wc
            v_perturbed = vc + robot.translational_noise(vc, wc)
            w_perturbed = wc + robot.rotational_noise(vc, wc)
            # sample pose
            position = particle[0]
            particle[0] = robot.next_position_from_state(position[0], position[1], position[2], v_perturbed, w_perturbed, self._change_t)
            particle[0] = wrap(particle[0], index=2)
            new_particle, w = self.ekf_measurement_step(measurements_from_robot, particle)
            updated_particles.append(new_particle)
            all_weights.append(w)

        all_weights = all_weights / np.sum(all_weights)
        # if np.count_nonzero(all_weights) < 100:
        self.particle_to_plot = self.particles[np.argmax(all_weights)]
        if first_step:
            self.particles = updated_particles
        else:
            self.resample_particles(updated_particles, all_weights)
        particles = np.array([particle[0] for particle in self.particles])[:,:2].reshape((-1,2))
        self.mean_belief = np.mean(particles, axis=0)
        self.covariance_belief = np.var(particles, axis=0)

    def ekf_measurement_step(self, measurements_from_robot, particle):
        Qt = self.Qt
        position = particle[0]
        mean_beliefs = particle[1]
        covar_beliefs = particle[2]
        initialized = particle[3]
        w=1/1000
        for measurement_set in measurements_from_robot:
            index = measurement_set[0]
            measurement = measurement_set[1]
            x = position[0]
            y = position[1]
            theta = position[2]
            if not initialized[index]:
                r1 = measurement[0]
                phi1 = measurement[1]
                mu_x1 = r1*cos(phi1 + theta) + x
                mu_y1 = r1*sin(phi1 + theta) + y
                mean_beliefs[index][0] = mu_x1
                mean_beliefs[index][1] = mu_y1
                initialized[index] = True
                delta_x1 = mu_x1 - x
                delta_y1 = mu_y1 - y
                q1 = (delta_x1)**2 + (delta_y1)**2
                H1 = np.zeros((2,2))
                H1[0,0] = (mu_x1-x)/sqrt(q1)
                H1[0,1] = (mu_y1-y)/sqrt(q1)
                H1[1,0] = -(mu_y1-y)/q1
                H1[1,1] = (mu_x1-x)/q1
                covar_beliefs[index] = inv(H1) @ Qt @ inv(H1.T)
                w = 1/1000
                continue
            # Range and bearing from mean belief
            mean_belief = mean_beliefs[index]
            f_x = mean_belief[0]
            f_y = mean_belief[1]
            delta_x = f_x - x
            delta_y = f_y - y
            q = (delta_x)**2 + (delta_y)**2
            zti = np.array([
                [np.sqrt(q)],
                [wrap(np.arctan2((delta_y), (delta_x)) - theta)]]).reshape((2,1))
            Ht = np.array([
                [(f_x - x)/np.sqrt(q), (f_y - y)/np.sqrt(q)],
                [-(f_y - y)/q, (f_x - x)/q]]).reshape((2,2))
            residual = wrap((measurement - zti), index=1)

            covariance_belief = covar_beliefs[index]
            Q = Ht @ covariance_belief @ Ht.T + Qt
            Kt = covariance_belief @ Ht.T @ np.linalg.inv(Q)
            mean_beliefs[index] = (mean_belief[:,np.newaxis] + Kt @ residual).reshape(2,)
            covar_beliefs[index] = (np.eye(len(Kt)) - Kt @ Ht) @ covariance_belief
            w = np.linalg.det(2*pi*Q)**(-1/2) * exp(-1/2*residual.T @ inv(Q) @ residual)
            # Update position
            

        return [position, mean_beliefs, covar_beliefs, initialized], w

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
