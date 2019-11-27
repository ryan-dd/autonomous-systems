import numpy as np
import matplotlib.pyplot as plt

# Create map
N = 99
Np = 100 + 1
gamma = 0.995

full_map = np.ones((Np,Np))*-2
wall_cost = -100
goal_rewards = 100000
obstacles = -5000
# Initialize walls and obstacle maps as empty
walls = np.zeros((Np,Np))
obs1 = np.zeros((Np,Np))
obs2 = np.zeros((Np,Np))
obs3 = np.zeros((Np,Np))
goal = np.zeros((Np,Np))


# Create exterior walls
walls[1,1:N] = 1
walls[1:N,1] = 1
walls[N,1:N+1] = 1
walls[1:N,N] = 1


# Create single obstacle
obs1[19:39,29:79] = 1
obs1[9:19,59:64] = 1

# Another obstacle
obs2[44:64,9:44] = 1

# Another obstacle
obs3[42:91,74:84] = 1
obs3[69:79,49:74] = 1

# The goal states
goal[74:79,95:97] = 1

# Put walls and obstacles into full_map
r = full_map + 100000*goal - 5000*(obs1 + obs2 + obs3) - 100*walls
V = np.zeros((Np, Np)) + 1000*goal

# Plot full_map
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(V.T, origin="lower")
cb = fig.colorbar(im)
plt.draw()
plt.pause(0.0001)
# Plot it until it shows up
for i in range(3):
    im = ax.imshow(V.T, origin="lower")
    fig.canvas.draw_idle()
    plt.pause(0.0001)


V = full_map
original_V = np.copy(V)
VN_original = np.roll(V, 1, axis=0)
VS_original = np.roll(V, -1, axis=0)
VW_original = np.roll(V, 1, axis=1)
VE_original = np.roll(V, -1, axis=1)
converged=False
index=0

while not converged:
    V_prev = np.copy(V)
    # Value interation
    VN = np.roll(V, 1, axis=0)
    VS = np.roll(V, -1, axis=0)
    VW = np.roll(V, 1, axis=1)
    VE = np.roll(V, -1, axis=1)

    V_all_N = (0.8*VN + 0.1*VE + 0.1*VW)[:,:,np.newaxis]
    V_all_E = (0.8*VE + 0.1*VN + 0.1*VS)[:,:,np.newaxis]
    V_all_S = (0.8*VS + 0.1*VE + 0.1*VW)[:,:,np.newaxis]
    V_all_W = (0.8*VW + 0.1*VN + 0.1*VS)[:,:,np.newaxis]
    all_rewards = np.concatenate((V_all_N, V_all_E, V_all_S, V_all_W), axis=2)
    V = gamma * (np.amax(all_rewards, axis=2) + r)
    V[74:79,95:97] = 1000

    diff = np.abs(V-V_prev)
    index += 1
    if np.all(np.isclose(V, V_prev)):
        converged=True
cb.remove()
ax.imshow(V, origin="lower")
cb = fig.colorbar(im)
fig.canvas.draw_idle()
plt.pause(5)
action_vectors = np.array([[1,0],[0,1],[-1,0],[0,-1]]) # N, E, S, W

X = np.repeat(list(range(Np)),Np)
Y = np.tile(list(range(Np)),Np)
direction = np.argmax(all_rewards, axis=2).flatten()
U = action_vectors[:,0][direction]
V = action_vectors[:,1][direction]
q = ax.quiver(X, Y, U, V, units='xy', scale=1, color='red', label="Quiver Plot")
plt.pause(20)


