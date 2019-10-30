from scipy.io import loadmat
import numpy as np
from math import radians, log
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


OFFSET_DISTANCE = 0.5
ALPHA = 1
BETA = radians(1)
Z_MAX = 150
L0 = log(1.0)
L_OCC = log(0.7/0.3)
L_FREE = log(0.3/0.7)

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
        # Insert log odds related to p = 0.5, which correlates to no information about whether
        # the cell is occupied
        grid_map[i].append(L0)

# Define sensor model
def inverse_range_sensor_model(cell, state_t, z_ranges, z_bearings, z_pointing_angles):
    xi = cell[0]
    yi = cell[1]
    x = state_t[0]
    y = state_t[1]
    theta = state_t[2]
    r = np.sqrt((xi-x)**2 + (yi-y)**2)
    phi = np.arctan2(yi - y, xi - x) - theta
    k = np.argmin(np.abs(np.subtract(phi, z_pointing_angles)))
    if r > np.min([Z_MAX, z_ranges[k] + ALPHA/2]) or np.abs(phi - z_pointing_angles[k]) > BETA/2:
        return L0
    elif z_ranges[k] < Z_MAX and np.abs(r - z_ranges[k]) < ALPHA/2:
        return L_OCC
    else:
        return L_FREE

# Update the map at every time step
for time_step in range(len(ranges[0])):
    state_t = states[:, time_step]
    ranges_t = ranges[:, time_step]
    bearings_t = bearings[:, time_step]
    for i in range(100):
        for j in range(100):
            grid_cell_log_odds = grid_map[i][j]
            grid_map[i][j] = grid_cell_log_odds + inverse_range_sensor_model([i, j], state_t, ranges_t, bearings_t, pointing_angles) - L0

# Get original probabilities as log
log_odds = np.array(grid_map)
probabilities = 1 - 1/(1+np.exp(log_odds))

# Plot the probabilities in a map
plt.imshow(probabilities.T, cmap='Blues', interpolation='nearest', origin='lower')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# _x = list(range(100))
# _y = _x
# _xx, _yy = np.meshgrid(_x, _y)
# x, y = _xx.ravel(), _yy.ravel()

# top = probabilities.ravel()
# bottom = np.zeros_like(top)
# width = depth = 1

# ax.bar3d(x, y, bottom, width, depth, top, shade=True)
# ax.set_title('Shaded')

# plt.show()