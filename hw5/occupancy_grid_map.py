from scipy.io import loadmat
import numpy as np
from math import radians, log

OFFSET_DISTANCE = np.sqrt(1+1)
ALPHA = 1
BETA = radians(5)
Z_MAX = 150
L0 = log(1.0)
L_OCC = log(0.6/0.4)
L_FREE = log(0.4/0.6)

data = loadmat("hw5/state_meas_data.mat")
states = data['X']
ranges = data['z'][0]
bearings = data['z'][1]
pointing_angles = data['thk'][0]

# Make grid map 100 x 100
grid_map = []
for i in range(100):
    grid_map.append([])
    for j in range(100):
        # Insert log odds related to p = 0.5
        grid_map[i].append(L0)

for time_step in range(len(ranges)):
    for i in range(100):
        for j in range(100):
            grid_cell_probability = grid_map[i][j]
            cell_occupied = check_grid_occupancy(i, j)
            if cell_occupied:
                grid_map[i][j] = grid_cell_probability + inverse_range_sensor_model([i, j], states, ranges, bearings, pointing_angles) - L0


def check_grid_occupancy(i, j):
    return True

def inverse_range_sensor_model(cell, xt, z_ranges, z_bearings, z_pointing_angles):
    xi = cell[0] - OFFSET_DISTANCE
    yi = cell[1] - OFFSET_DISTANCE
    x = xt[0]
    y = xt[1]
    theta = xt[2]
    r = np.sqrt((xi-x)**2 + (yi-y)**2)
    phi = np.arctan2(yi - y, xi - x) - theta
    k = np.argmin(np.abs(np.subtract(phi, z_pointing_angles)))
    if r > np.min(Z_MAX, z_ranges[k] + ALPHA/2) or phi - z_pointing_angles[k] > BETA/2:
        return L0
    elif z_ranges[k] < Z_MAX and np.abs(r-z_ranges[k]) < ALPHA/2:
        return L_OCC
    else:
        return L_FREE