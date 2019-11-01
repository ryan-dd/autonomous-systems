from math import fmod, pi
import numpy as np

def wrap_angle(angle, l_wrap, u_wrap):
    angle_range = abs(l_wrap)+ abs(u_wrap)
    angle1 = fmod((angle + u_wrap), pi)
    if angle1 < 0:
        angle1 += angle_range
    return angle1

def wrap(angle, index=None):
    if index:
        angle[index] -= 2*np.pi * np.floor((angle[index] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle

if __name__ == "__main__":
    val1 = wrap_angle(6.7, -pi, pi)
    print(val1)
