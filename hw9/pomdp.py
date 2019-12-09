import numpy as np
import matplotlib.pyplot as plt
# State 0 is facing towards the lava, State 1 is facing away from the lava.
# Control actions are moving forward, moving backward (which are absorbing states) and turning around

T = 20
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
    print(tau)
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
    # Prune the lines
    # Check for duplicates
    line_dict = {}
    next_lines = []
    for line in line_set:
        key = (line[0], line[1])
        if key not in line_dict:
            line_dict[key] = line
            next_lines.append(line)
    if tau==0:
        pruned_lines = next_lines
    else:
        # Keep any line that is not strictly dominated
        to_examine = next_lines[1]
        pruned_lines = np.array([to_examine])
        start_x = 0
        finished = False
        lines_to_iterate = np.delete(next_lines, 1, axis=0)
        while not finished:
            # Check minimum intersecting lines
            m_0 = to_examine[1]-to_examine[0]
            b_0 = to_examine[0]
            all_x = []
            lines_to_examine = []
            for line in lines_to_iterate:
                m = line[1] - line[0]
                b = line[0]
                if (m_0 - m) == 0:
                    continue
                x = (b-b_0)/(m_0-m)
                if x < start_x or x > 1:
                    continue
                lines_to_examine.append(line)
                all_x.append(x)
            if len(lines_to_examine) == 0:
                break

            mins = np.argmin(all_x)
            candidates = lines_to_examine[mins].reshape(-1,2)
            pruned_lines = np.concatenate((pruned_lines, candidates), axis=0)
            lines_to_iterate = np.delete(lines_to_iterate, mins, axis=0)
            to_examine = np.copy(candidates)[0]
            start_x = min(all_x)   
    line_set = np.copy(pruned_lines)
for line in line_set:
    plt.plot([0,1], line)
plt.show()

