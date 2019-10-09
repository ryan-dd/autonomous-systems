from random import rand
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

    def prediction_step(self, robot):
        vc = robot.vc
        wc = robot.wc
        updated_particles = []
        for particle in self.particles:
            new_particle = robot.next_position_from_state(particle[0], particle[1], particle[2], vc, wc, self._change_t)
            updated_particles.append(new_particle)

    def measurement_step(self, robot):
        pass

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
