import numpy as np
# State 0 is facing towards the lava, State 1 is facing away from the lava.
# Control actions are moving forward, moving backward (which are absorbing states) and turning around

T = 1
gamma = 1

control_action_rewards = np.array(
   [[-100,100,-1],
    [100,-50,-1]]).T

measurement_probabilities = np.array(
    [[0.7, 0.3],
    [0.3, 0.7]]
)
px1_u_x2 = np.array(
    [[[0, 0], [0, 0]],
    [[0, 0], [0, 0]],
    [[0.8, 0.2],[0.2, 0.8]]]
)

# Line set for each control action
line_set = {0:[],1:[],2:[]}
for tau in range(T):
    new_line_set = {0:[],1:[],2:[]}
    # Cycle through each line
    for u_index, lines in line_set.items():
        for line in lines:
            # Cycle through each control action
            for u in range(3):
                # Cycle through each measurement
                for z in range(2):
                    # Cycle through each starting state
                    for j in range(2):
                        vkuzj = 0
                        for i in range(2):
                            vik = line[i]
                            pz_xi = measurement_probabilities[i][z]
                            pxi_u_xj = px1_u_x2[u][i][j]
                            vkujz += vik*pz_xi*pxi_u_xj