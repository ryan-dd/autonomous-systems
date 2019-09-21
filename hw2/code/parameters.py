from math import cos, pi
import numpy


INITIAL_X = -5 # meters
INITIAL_Y = -3 # meters
INITIAL_THETA = 90 # degrees

def commanded_translational_velocity(t):
    return 1 + 0.5*cos(2*pi*0.2*t)

def commanded_rotational_velocity(t):
    return -0.2 + 2*cos(2*pi*0.6*t)

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

