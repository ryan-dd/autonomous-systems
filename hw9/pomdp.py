import numpy as np
import matplotlib.pyplot as plt
# State 0 is facing towards the lava, State 1 is facing away from the lava.
# Control actions are moving forward, moving backward (which are absorbing states) and turning around

T = 4
gamma = 1.0

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
    policy = {}
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
                policy[(v[0], v[1])] = u
                all_new_lines.append(v)
    line_set = np.copy(all_new_lines)
    if tau==0:
        pruned_lines = line_set
    else:
        # Prune the lines
        # Check for duplicates
        line_dict = {}
        next_lines = []
        for line_first in line_set:
            skip_line = False
            for check_line in next_lines:
                if np.allclose(line_first, check_line):
                    skip_line = True
                    break
            if skip_line:
                continue
            next_lines.append(line_first)
        # Keep any line that is not strictly dominated
        to_examine = next_lines[np.argmax(np.array(next_lines)[:,0])]
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



def take_measurement(actual_state):
    prob = np.random.random()
    if prob > 0.7:
        return int(not actual_state)
    else:
        return actual_state

def update_belief_after_measurement(p1, measured):
    if measured == 1:
        return 0.4*p1 + 0.3
    else:
        return -0.4*p1 + 0.7

def update_belief_after_state_change(p1, destination_state):
    if destination_state == 1:
        return 0.6*p1 + 0.2
    else:
        return -0.6*p1 + 0.8

def take_step_u3(actual_state):
    prob = np.random.random()
    if prob > 0.8:
        return actual_state
    else:
        return int(not actual_state)


steps = 20
p1 = 0.6
actual_state = 0
reward = 0
m = line_set[:,1] - line_set[:,0]
b = line_set[:,0]
for i in range(steps):
    measurement = take_measurement(actual_state)
    p1 = update_belief_after_measurement(p1, measurement)
    policy_line = line_set[np.argmax(m*p1 + b)]
    action_to_take = policy[(policy_line[0], policy_line[1])]
    reward += control_action_rewards[action_to_take][actual_state]
    if action_to_take == 0 or action_to_take == 1:
        break
    else:
        p0 = p1
        prev_state = actual_state
        actual_state = take_step_u3(actual_state)
        p1 = update_belief_after_state_change(p1, prev_state)
        print("Step: {}, p1: {}, x1: {}, x2: {}, z: {}, p2: {}".format(i, p0, prev_state, actual_state, measurement, p1))
print("Step: {}, Final Reward: {}".format(i, reward))

plt.show()
