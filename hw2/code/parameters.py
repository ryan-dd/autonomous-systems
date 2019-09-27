from math import cos, pi, sqrt
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt


INITIAL_X = -5 # meters
INITIAL_Y = -3 # meters
INITIAL_THETA = pi # degrees

LANDMARK_1_LOCATION = [6, 4]
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

