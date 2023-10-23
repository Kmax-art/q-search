# Quantum Walk-Based Search Algorithm
# Author: Himanshu Sahu
# Date: 23-10-2023


# Copyright (c) 2023 Himanshu Sahu. All rights reserved.

# -------- Import libraries ------------

import numpy as np
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import functools as ft
from scipy.linalg import block_diag
from rich.progress import track
from Essential_Function import *


# Plot specification

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.color'] = 'k'
plt.rcParams['axes.labelcolor'] = 'k'


N = 2**4 # lattice sites
pos = np.eye(N) # position space
coin = np.eye(4)  # coin space

# Coin Switch

UD = np.outer(coin[3],coin[0]) # Takes from U to D
DU = np.outer(coin[0],coin[3]) 
LR = np.outer(coin[2],coin[1]) 
RL = np.outer(coin[1],coin[2])  

UU = np.outer(coin[0],coin[0])
DD = np.outer(coin[3],coin[3])
LL = np.outer(coin[1],coin[1])
RR = np.outer(coin[2],coin[2])

# Constructing Shift Operator -- Periodic boundary condition 

shift_down = np.kron(np.eye(N),np.roll(np.eye(N), 1, axis=0))
shift_up = np.kron(np.eye(N),np.roll(np.eye(N), -1, axis=0))
shift_left = np.kron(np.roll(np.eye(N), 1, axis=0),np.eye(N))
shift_right = np.kron(np.roll(np.eye(N), -1, axis=0),np.eye(N))

shift = np.kron(UD,shift_up) + np.kron(DU,shift_down) + np.kron(LR,shift_left) + np.kron(RL,shift_right)

print('Is shift operator unitary?', np.allclose(shift@conjT(shift),np.eye(N**2*4)))

# Grover's Coin Operator 

Qdim = 4
Grover_OP = 2*np.ones((Qdim,Qdim))/Qdim - np.eye(Qdim)

# Evolution operator for the walk

Evolution_OP = shift@np.kron(Grover_OP,np.eye(N**2))

def ORACLE(marked_points):
    # basis = []
    mm = np.zeros((N**2,N**2))
    if len(marked_points) == 0:
        return np.eye(Qdim*N**2)
    else:
        for i in range(len(marked_points)):
            xx = np.zeros((N,1)); yy = np.zeros((N,1))
            coordinate = marked_points
            xx[coordinate[0],0] = 1; yy[coordinate[1],0] = 1
            basis = np.kron(xx,yy)
            mm = mm + basis@conjT(basis)

        R = np.eye(Qdim*N**2) - np.kron((2*np.ones((Qdim,Qdim))/Qdim),mm)
    return R


T = 8  # Amplification Time 
τ = 2 # Particle's velocity 

m = int(T/τ) # number of layers

Tf = 20 # final time 
ti = 0 # real time 


trajectory = []

position = [[np.random.randint(0,N),np.random.randint(0,N)] for i in range(int(Tf/τ))]

for i in range(Tf):
    position_at_ti = []

    for j in range(m):
        if i >= τ*j:
            position_at_ti.append(position[j + m*int((i-j*τ)/T)])
        else:
            position_at_ti.append([])
            
    trajectory.append(position_at_ti)
    
    
evolution_op_list = []

for i in range(m):
    evolution_op_list.append(Evolution_OP)

evolution_op = ft.reduce(block_diag,evolution_op_list)
  
Ψ0 = np.ones((Qdim*N**2*m,1))
Ψ0 = Ψ0/vec_norm(Ψ0)
Ψt = np.copy(Ψ0)
P = []

for i in tqdm(range(Tf)):
    position_at_ti = trajectory[i]
    
    oracle_list = []
    
    for j in range(m):
        oracle_list.append(ORACLE(position_at_ti[j]))
    
    oracle = ft.reduce(block_diag,oracle_list)
    
    U = evolution_op@oracle 
    
    Pt = []
    for k in range(m):
        ψt = Ψt[k*Qdim*N**2:k*Qdim*N**2 + Qdim*N**2]
        P_layer = np.zeros((N,N))
        for xx in range(N):
            xket = np.zeros((N,1))
            xket[xx,0] = 1
            for yy in range(N):
                yket = np.zeros((N,1))
                yket[yy,0] = 1
                basis = np.kron(xket,yket)
                pt = 0
                for ii in range(4):
                    ϕ = np.kron(coin[:,[ii]],basis)
                    pt = pt + np.absolute(conjT(ϕ)@ψt)**2
                P_layer[xx,yy] = pt
        Pt.append(P_layer)
    P.append(Pt)

    Ψt = U@Ψt
    
    

import matplotlib.pyplot as plt
import matplotlib.animation as animation



# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
fig, axs = plt.subplots(1,m,figsize=(16, 4))

ims = []
for i in track(range(10)):
    
    for ii in range(m):
        ax = axs[ii]
        ax.imshow(P[i][ii], animated=True)

    if i == 0:
        for ii in range(m):
            ax = axs[ii]
            im = ax.imshow(P[i][0])  # show an initial one first
    
    ims.append([im])

    
ani = animation.ArtistAnimation(fig, ims, interval=20, blit=False,
                                repeat_delay=1000)

# To save the animation, use e.g.
#
ani.save("movie.gif")
#
# or
#
writer = animation.FFMpegWriter(
    fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("movie.mp4", writer=writer)

plt.show()