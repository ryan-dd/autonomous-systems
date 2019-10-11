from random import random as rand
from math import pi

import numpy as np
from heading_range_robot.parameters import ALPHA1, ALPHA2, ALPHA3, ALPHA4, INITIAL_X, INITIAL_Y, INITIAL_THETA, STD_DEV_LOCATION_RANGE, STD_DEV_LOCATION_BEARING, LANDMARKS


class ParticleFilter:
    def __init__(self, sample_period, number_of_particles):
        self._change_t = sample_period
        self.Qt = np.eye(2) * np.vstack(
            (STD_DEV_LOCATION_RANGE**2, STD_DEV_LOCATION_BEARING**2))
        self.all_features = LANDMARKS
        self.n = number_of_particles
        self.particles = self.create_random_particles(number_of_particles)

    def create_random_particles(self, n):
        all_particles = []
        for _ in range(n):
            particle_x = rand()*10-5
            particle_y = rand()*10-5
            particle_theta = rand()*2*pi
            weight = 1/n
            all_particles.append([particle_x, particle_y, particle_theta, weight])
        return np.array(all_particles)

    def particle_filter(self, robot):
        vc = robot.vc
        wc = robot.wc
        updated_particles = []
        measurements_from_robot = [] 
        for feature in self.all_features:
                f_x = feature[0]
                f_y = feature[1]
                measurement = simulate_measurement(robot.true_state, f_x, f_y)
                measurements_from_robot.append(measurement)
        all_weights = []
        for particle in self.particles:
            new_particle = robot.next_position_from_state(particle[0], particle[1], particle[2], vc, wc, self._change_t)
            weight = self.probability_of_measurement(new_particle, measurements_from_robot)
            all_weights.append(weight)
            updated_particles.append(new_particle)
        self.resample_particles(updated_particles, all_weights)

    def resample_particles(self, updated_particles, all_weights):
        final_particles = []
        minv = 1/len(updated_particles)
        r = rand()*minv
        c = all_weights[0]
        for m, particle in enumerate(updated_particles):
            U = r + (m*minv)
            while U > c:
                i += 1
                c = c + all_weights[i]


    def probability_of_measurement(self, new_particle, measurement_from_robot):
        weight = 1.0
        for index, feature in enumerate(self.all_features):
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
            p_r = sample_normal_distribution(error[0], STD_DEV_LOCATION_RANGE)
            p_b = sample_normal_distribution(error[1], STD_DEV_LOCATION_BEARING)
            weight *= p_r*p_b
        return weight

def sample_normal_distribution(mean, std_dev):
    return np.random.normal(mean, std_dev)

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
