import numpy as np
# State 0 is facing towards the lava, State 1 is facing away from the lava.
# Control actions are moving forward, moving backward (which are absorbing states) and turning around

T = 2
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
line_set = [[[-100, 100]],[[-50, 100]],[[-1, -1]]]
for tau in range(T-1):
    new_line_set = [[],[],[]]
    v_ukzj = [[],[],[]]
    # Cycle through each line
    for u_associated, lines in enumerate(line_set):
        n_lines_next = len(lines)
        # For each original line there are 4 lines created
        v_ukzj[u_associated] = np.zeros((n_lines_next, 3, 2, 2))
        for k, line in enumerate(lines):
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
                            v_ukzj[u_associated][k][u][z][j] += vik*pz_xi*pxi_u_xj
    for u_associated, lines in enumerate(line_set):
        for k, line in enumerate(lines):
            for u in range(3):
                v = [0,0]
                for i in range(2):
                    v[i] = gamma*(control_action_rewards[u][i] + v_ukzj[u_associated][k][u][0][i] + v_ukzj[u_associated][k][u][1][i])
                new_line_set[u].append(v)
hi = 0