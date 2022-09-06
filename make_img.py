import numpy as np
import math
import matplotlib.pyplot as plt

def spiral(X, Y):
    x = y = 0
    dx = 0
    dy = -1
    out = []

    rx = math.remainder(X,2)
    ry = math.remainder(Y,2)

    if rx == 0:
        x0 = math.floor(X/2) - 1
    else:
        x0 = math.ceil(X/2) - 1

    if ry == 0:
        y0 = math.floor(Y/2) - 1
    else:
        y0 = math.ceil(Y/2) - 1

    for i in range(max(X, Y)**2):
        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
            out.append([x+x0,y+y0])
            # print([x+x0,y+y0])
            # DO STUFF...
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    return out

x,y = 25,25

idx = spiral(x,y)

A = np.zeros(shape = (x,y))

for count, this_idx in enumerate(idx):
    base_number = len(idx) - count
    A[this_idx[0],this_idx[1]] = base_number - np.random.uniform(0,base_number,1)

plt.imshow(A,cmap='gray')
plt.show()

print('eof')