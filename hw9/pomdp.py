import numpy as np
import matplotlib.pyplot as plt
# State 0 is facing towards the lava, State 1 is facing away from the lava.
# Control actions are moving forward, moving backward (which are absorbing states) and turning around

T = 3
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

# Initial line set
line_set = [[0,0]]
for tau in range(T):
    all_new_lines = []
    policy = [[],[],[]]
    v_kuzj = np.zeros((len(line_set),3,2,2))
    # Cycle through each line
    for k, line in enumerate(line_set):
        # Cycle through each control action
        for u in range(3):
            # Cycle through each measurement
            for z in range(2):
                # Cycle through each state
                for j in range(2):
                    for i in range(2):
                        vik = line[i]
                        pz_xi = measurement_probabilities[i][z]
                        pxi_u_xj = px1_u_x2[u][i][j]
                        v_kuzj[k][u][z][j] += vik*pz_xi*pxi_u_xj
    for u in range(3):
        for k1, line1 in enumerate(line_set):
            for k2, line2 in enumerate(line_set):
                v = [0,0]
                for i in range(2):
                    v[i] = gamma*(control_action_rewards[u][i] + v_kuzj[k1][u][0][i] + v_kuzj[k2][u][1][i])
                policy[u].append(v)
                all_new_lines.append(v)
    line_set = np.copy(all_new_lines)
    if tau == T-1:
        for line in all_new_lines:
            plt.plot([0,1], line)
    # Prune the lines
    # Check for duplicates
    line_dict = {}
    next_lines = []
    for line in line_set:
        key = (line[0], line[1])
        if key not in line_dict:
            line_dict[key] = line
            next_lines.append(line)
    line_set = np.copy(next_lines)
plt.show()

