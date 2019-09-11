import numpy as np
from control import StateSpace, matlab

from hw1.code.parameters import *


m = VEHICLE_MASS
b = LINEAR_DRAG_COEFFICIENT

A = np.array([[0, 1], [0, -b/m]])
B = np.array([[0], [1/m]])
C = np.array([[1, 0]])
D = np.array([[0]])
state_space = StateSpace(A, B, C, D)
discrete_time = matlab.c2d(state_space, SAMPLE_PERIOD)

