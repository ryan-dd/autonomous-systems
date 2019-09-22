from math import cos, pi, sqrt
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt


INITIAL_X = -5 # meters
INITIAL_Y = -3 # meters
INITIAL_THETA = 90 # degrees

LANDMARK_1_LOCATION = [6,4]
LANDMARK_2_LOCATION = [-7, 8]
LANDMARK_3_LOCATION = [6, -4]
STD_DEV_LOCATION_RANGE = 0.1
STD_DEV_LOCATION_BEARING = 0.05
SAMPLE_PERIOD = 0.1 # seconds
TOTAL_TIME = 20 # seconds

ALPHA1 = 0.1
ALPHA2 = 0.01
ALPHA3 = 0.01
ALPHA4 = 0.1

def commanded_translational_velocity(t):
    return 1 + 0.5*cos(2*pi*0.2*t)

def commanded_rotational_velocity(t):
    return -0.2 + 2*cos(2*pi*0.6*t)

def translational_noise(v, w):
    translational_error_variance = ALPHA1*v**2+ALPHA2*w**2
    return normal(0, sqrt(translational_error_variance))

def rotational_noise(v, w):
    rotational_error_variance = ALPHA3*v**2+ALPHA4*w**2
    return normal(0, sqrt(rotational_error_variance))
