# Quantum Walk-Based Search Algorithm
# Author: Himanshu Sahu
# Date: 23-10-2023

# Description:
# This code implements a discrete-time quantum walk-based search algorithm. It uses a quantum state
# to explore a state space and find a specific target state. The Grover coin operator and conditional
# shift operator are used in each step of the quantum walk.

# Usage:
# - Adjust the 'n' and 'num_steps' variables to match the size of your state space and the number of steps.
# - Customize the 'grover_coin' and 'conditional_shift' functions to suit your specific problem.
# - Run the code to find the solution, which is printed at the end.

# Note:
# This is a basic framework and should be adapted for your specific use case. Implement additional
# features and logic as needed, such as stopping conditions and error correction.

# For more information on quantum search algorithms and quantum walks, refer to relevant literature
# and documentation.

# Copyright (c) 2023 Himanshu Sahu. All rights reserved.


# -------- Import libraries ------------

import numpy as np
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import functools as ft
from scipy.linalg import block_diag
from sympy import *
from Essential_Function import *

# -----------

# Plot settings 

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.color'] = 'k'
plt.rcParams['axes.labelcolor'] = 'k'

# ----------------------- Ordinary Search algorithm ----------

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


# ----------- Open boundary condition -----------

# Construction of interior shift Operator 

shift_down = np.kron(np.eye(N),np.diag(np.ones((1,N-1))[0],1))
shift_up = np.kron(np.eye(N),np.diag(np.ones((1,N-1))[0],-1))
shift_left = np.kron(np.diag(np.ones((1,N-1))[0],1),np.eye(N))
shift_right = np.kron(np.diag(np.ones((1,N-1))[0],-1),np.eye(N))

shift_in = np.kron(UD,shift_up) + np.kron(DU,shift_down) + np.kron(LR,shift_left) + np.kron(RL,shift_right)

# Construction of boundary shift operator 

NN = np.zeros((N,N)); NN[N-1,N-1] = 1 
OO = np.zeros((N,N)); OO[0,0] = 1

shift_yN = np.kron(np.eye(N),NN); shift_y0 = np.kron(np.eye(N),OO)
shift_xN = np.kron(NN,np.eye(N)); shift_x0 = np.kron(OO,np.eye(N))

shift_boundary = np.kron(UU,shift_yN) + np.kron(DD,shift_y0) + np.kron(LL,shift_x0) + np.kron(RR,shift_xN)

# Complete shift operator 

shift = shift_in + shift_boundary


# ------------------------

# Grover's Coin Operator 

Qdim = 4
Grover_OP = 2*np.ones((Qdim,Qdim))/Qdim - np.eye(Qdim)

# Evolution operator for the walk

Evolution_OP = shift@np.kron(Grover_OP,np.eye(N**2))

# Oracle -  

list_ = [[6, 8] , [8, 9], [12, 5], [15, 5]]

def ORACLE(marked_points):
    mm = np.zeros((N**2,N**2))
    for i in range(len(marked_points)):
        xx = np.zeros((N,1)); yy = np.zeros((N,1))
        coordinate = marked_points[i]
        xx[coordinate[0],0] = 1; yy[coordinate[1],0] = 1
        basis = np.kron(xx,yy)
        mm = mm + (np.cos(θm)*np.exp(1j*θm))*basis@conjT(basis)
    
    R = np.eye(Qdim*N**2) - np.kron((2*np.ones((Qdim,Qdim))/Qdim),mm)
    
    return R

oracle = ORACLE(list_)


U = Evolution_OP@oracle 

# Initial State -----

Ψ0 = np.ones((Qdim*N**2,1))
Ψ0 = Ψ0/vec_norm(Ψ0)


# Evolution ------

Ψt = np.copy(Ψ0)
steps = 60
prob = np.zeros((steps,len(list_)))

xx = np.zeros((N,1)); yy = np.zeros((N,1))


for i in range(steps):
    for j in range(len(list_)):
        xx = np.zeros((N,1)); yy = np.zeros((N,1))
        coordinate = list_[j]
        xx[coordinate[0],0] = 1; yy[coordinate[1],0] = 1
        basis = np.kron(xx,yy)
        ϕ1 = np.kron(coin[:,[0]],basis); ϕ2 = np.kron(coin[:,[1]],basis); ϕ3 = np.kron(coin[:,[2]],basis); ϕ4 = np.kron(coin[:,[3]],basis)
        
        prob[i,j] = np.absolute(conjT(ϕ1)@Ψt)**2 + np.absolute(conjT(ϕ2)@Ψt)**2 + np.absolute(conjT(ϕ3)@Ψt)**2 + np.absolute(conjT(ϕ4)@Ψt)**2 
    
    Ψt = U@Ψt

# --------------------------------------
    
xaxis = np.linspace(0,steps-1,steps)
ax = plt.subplot(111)

for i in range(len(list_)):
    ax.scatter(xaxis, prob[:,i], color='white', s=25, zorder=2)
    ax.scatter(xaxis, prob[:,i], color='black', s=10, zorder=3)
    ax.plot(xaxis,prob[:,i], linestyle='-', linewidth=1, zorder=1,label = str(list_[i]))
    
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='both', labelsize=15)
font2 = {'family':'serif','color':'black','size':15}
plt.xlabel('Number of Steps',fontdict = font2)
plt.ylabel('Probability $P$',fontdict = font2)
plt.legend(title = 'Target Points',loc='best', frameon = False,fontsize = 12,title_fontsize = 12)
plt.show()


# -------------- Periodic boundary condition ---------


# Constructing Shift Operator -- Periodic boundary condition 

shift_down = np.kron(np.eye(N),np.roll(np.eye(N), 1, axis=0))
shift_up = np.kron(np.eye(N),np.roll(np.eye(N), -1, axis=0))
shift_left = np.kron(np.roll(np.eye(N), 1, axis=0),np.eye(N))
shift_right = np.kron(np.roll(np.eye(N), -1, axis=0),np.eye(N))

shift = np.kron(UD,shift_up) + np.kron(DU,shift_down) + np.kron(LR,shift_left) + np.kron(RL,shift_right)

# Grover's Coin Operator 

Qdim = 4
Grover_OP = 2*np.ones((Qdim,Qdim))/Qdim - np.eye(Qdim)

# Evolution operator for the walk

Evolution_OP = shift@np.kron(Grover_OP,np.eye(N**2))

# Oracle -  

# linear-set of points 
list_ =  [[6, 8] , [8, 9], [12, 5], [15, 5]] # [[2,2],[2,4]] # ,[2,6],[2,8]]



def ORACLE(marked_points,ϕm):
    mm = np.zeros((N**2,N**2))
    for i in range(len(marked_points)):
        xx = np.zeros((N,1)); yy = np.zeros((N,1))
        coordinate = marked_points[i]
        xx[coordinate[0],0] = 1; yy[coordinate[1],0] = 1
        basis = np.kron(xx,yy)
        mm = mm + (np.cos(θm)*np.exp(1j*θm))*basis@conjT(basis)
    
    R = np.eye(Qdim*N**2) - np.kron((2*np.ones((Qdim,Qdim))/Qdim),mm)
    return R

oracle = ORACLE(list_,ϕm)


U = Evolution_OP@oracle 

# Initial State -----

Ψ0 = np.ones((Qdim*N**2,1))
Ψ0 = Ψ0/vec_norm(Ψ0)


# Evolution ------

Ψt = np.copy(Ψ0)
steps = 50
prob = np.zeros((steps,len(list_)))

xx = np.zeros((N,1)); yy = np.zeros((N,1))


for i in range(steps):
    for j in range(len(list_)):
        xx = np.zeros((N,1)); yy = np.zeros((N,1))
        coordinate = list_[j]
        xx[coordinate[0],0] = 1; yy[coordinate[1],0] = 1
        basis = np.kron(xx,yy)
        ϕ1 = np.kron(coin[:,[0]],basis); ϕ2 = np.kron(coin[:,[1]],basis); ϕ3 = np.kron(coin[:,[2]],basis); ϕ4 = np.kron(coin[:,[3]],basis)
        
        prob[i,j] = np.absolute(conjT(ϕ1)@Ψt)**2 + np.absolute(conjT(ϕ2)@Ψt)**2 + np.absolute(conjT(ϕ3)@Ψt)**2 + np.absolute(conjT(ϕ4)@Ψt)**2 
    
    Ψt = U@Ψt

                
xaxis = np.linspace(0,steps-1,steps)
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['font.family'] = 'Calibri'


ax = plt.subplot(111)

for i in range(len(list_)):
    ax.scatter(xaxis, prob[:,i], color='white', s=25, zorder=2)
    ax.scatter(xaxis, prob[:,i], color='black', s=10, zorder=3)
    ax.plot(xaxis,prob[:,i], linestyle='-', linewidth=1, zorder=1,label = str(list_[i]))
    
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='both', labelsize=15)
font2 = {'family':'serif','color':'black','size':15}
plt.xlabel('Number of Steps',fontdict = font2)
plt.ylabel('Probability $P$',fontdict = font2)
plt.legend(title = 'Target Points',loc='best', frameon = False,fontsize = 12,title_fontsize = 12)
plt.show()