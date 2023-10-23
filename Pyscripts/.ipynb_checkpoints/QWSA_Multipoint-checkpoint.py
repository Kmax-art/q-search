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


# ----------- Static Case ---------------

# Periodic boundary condition 

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


list_ = [[[6, 8]], [[8, 9]], [[12, 5]], [[15, 5]]] # 


def ORACLE(marked_points):

    mm = np.zeros((N**2,N**2))
    for i in range(len(marked_points)):
        xx = np.zeros((N,1)); yy = np.zeros((N,1))
        coordinate = marked_points[i]
        xx[coordinate[0],0] = 1; yy[coordinate[1],0] = 1
        basis = np.kron(xx,yy)
        mm = mm + basis@conjT(basis)
    
    R = np.eye(Qdim*N**2) - np.kron((2*np.ones((Qdim,Qdim))/Qdim),mm)
    return R


oracle_list = []
evolution_op_list = []

for i in range(len(list_)):
    oracle_list.append(ORACLE(list_[i]))
    evolution_op_list.append(Evolution_OP)


oracle = ft.reduce(block_diag,oracle_list)
evolution_op = ft.reduce(block_diag,evolution_op_list)

print('Is oracle operator unitary?',np.allclose( oracle_list[0]@conjT(oracle_list[0]),np.eye(Qdim*N**2)))

U = evolution_op@oracle 

# Initial State -----

layers = len(list_)
Ψ0 = np.ones((Qdim*N**2*layers,1))
Ψ0 = Ψ0/vec_norm(Ψ0)


# Evolution ------

Ψt = np.copy(Ψ0)
steps = 50
P = []

for i in range(steps):
    Pt = []
    for k in range(layers):
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

    
    U = evolution_op@oracle 

# Initial State -----


layers = len(list_)
Ψ0 = np.ones((Qdim*N**2*layers,1))
Ψ0 = Ψ0/vec_norm(Ψ0)


# Evolution ------

Ψt = np.copy(Ψ0)
steps = 50
prob = [np.zeros((steps,len(list_[i]))) for i in range(layers)] # np.zeros((steps,layers))

for i in range(steps):    
    for k in range(layers):
        ψt = Ψt[k*Qdim*N**2:k*Qdim*N**2 + Qdim*N**2]
        list__ = list_[k]
        tot_marked = len(list__)
        for j in range(tot_marked):
            xx = np.zeros((N,1)); yy = np.zeros((N,1))
            coordinate = list__[j]
            xx[coordinate[0],0] = 1; yy[coordinate[1],0] = 1
            basis = np.kron(xx,yy)
            ϕ1 = np.kron(coin[:,[0]],basis); ϕ2 = np.kron(coin[:,[1]],basis); ϕ3 = np.kron(coin[:,[2]],basis); ϕ4 = np.kron(coin[:,[3]],basis)

            prob[k][i,j] = np.absolute(conjT(ϕ1)@ψt)**2 + np.absolute(conjT(ϕ2)@ψt)**2 + np.absolute(conjT(ϕ3)@ψt)**2 + np.absolute(conjT(ϕ4)@ψt)**2 
    Ψt = U@Ψt


xaxis = np.linspace(0,steps-1,steps)

fig, ax = plt.subplots(figsize = (6,4))

for j in range(layers):
    p = prob[j]
    list__ = list_[j]
    tot_marked = len(list__)
    for i in range(tot_marked):
        ax.scatter(xaxis, p[:,i], color='white', s=50, zorder=2)
        ax.scatter(xaxis, p[:,i], color='black', s=10, zorder=3)
        ax.plot(xaxis,p[:,i], linestyle='-', linewidth=1, zorder=1,label = str(list__[i]) +' \& ' + str(j))

font2 = {'family':'serif','color':'black','size':15}
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='both', labelsize=15)
plt.legend(title = r'Target Points \& Layer',loc='best', frameon = False)
# plt.xlim(0,20)
plt.xlabel(r'number of steps',fontdict = font2)
plt.ylabel(r'Probability',fontdict = font2)
plt.title('Periodic boundary condition',loc='left')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define the Gaussian function
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) / std_dev)**2 / 2)

# Generate sample data
x = np.linspace(0,steps-1,steps)
fig, ax = plt.subplots(figsize = (6,4))

for j in range(layers):
    p = prob[j]
    list__ = list_[j]
    tot_marked = len(list__)
    for i in range(tot_marked):
        x = np.linspace(0,steps-1,steps)
        y = p[:,i]
        initial_guess = [1.0, 0.0, 1.0]  # Initial guess for amplitude, mean, and std_dev
        params, covariance = curve_fit(gaussian, x, y, p0=initial_guess)
        fitted_amplitude, fitted_mean, fitted_std_dev = params
        x = np.linspace(0,steps,4*steps)
        ax.plot(x, gaussian(x, fitted_amplitude, fitted_mean, fitted_std_dev),linestyle='-', linewidth=1, zorder=1,label = str(list__[i]) +' \& ' + str(j))
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='major', labelsize=15)
font2 = {'family':'serif','color':'black','size':15}
plt.xlabel(r'$t$',fontdict = font2)
plt.ylabel(r'$P(t)$',fontdict = font2)
plt.legend(title = r'Target Points \& Layer',loc='center left', bbox_to_anchor=(1, 0.5),frameon = False)

plt.title('Search algorithm on Torus',fontdict = font2)
plt.show()


fig, axs = plt.subplots(1, 4,figsize = (12,2))

for i in range(4):
    ax = axs[i]
    pcm = ax.imshow(P[mx][i],cmap = 'viridis')
    fig.colorbar(pcm, ax=ax,fraction=0.046, pad=0.04)
    plt.setp(ax.spines.values(), alpha = 0)
    # ax.tick_params(which = 'both', size = 0, labelsize = 0,color = 'w')

plt.show()



# Open boundary condition 


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

print('Is shift operator unitary?', np.allclose(shift@conjT(shift),np.eye(N**2*4)))

# Grover's Coin Operator 

Qdim = 4
Grover_OP = 2*np.ones((Qdim,Qdim))/Qdim - np.eye(Qdim)

# Evolution operator for the walk

Evolution_OP = shift@np.kron(Grover_OP,np.eye(N**2))


list_ = [[[6, 8]], [[8, 9]], [[12, 5]], [[15, 5]]] 


def ORACLE(marked_points):
    mm = np.zeros((N**2,N**2))
    for i in range(len(marked_points)):

        xx = np.zeros((N,1)); yy = np.zeros((N,1))
        coordinate = marked_points[i]
        xx[coordinate[0],0] = 1; yy[coordinate[1],0] = 1
        basis = np.kron(xx,yy)
        mm = mm + basis@conjT(basis)
    
    R = np.eye(Qdim*N**2) - np.kron((2*np.ones((Qdim,Qdim))/Qdim),mm)
    return R

oracle_list = []
evolution_op_list = []

for i in range(len(list_)):
    oracle_list.append(ORACLE(list_[i]))
    evolution_op_list.append(Evolution_OP)


oracle = ft.reduce(block_diag,oracle_list)
evolution_op = ft.reduce(block_diag,evolution_op_list)

print('Is oracle operator unitary?', np.allclose(oracle_list[0]@conjT(oracle_list[0]),np.eye(Qdim*N**2)))

U = evolution_op@oracle 

# Initial State -----

layers = len(list_)
Ψ0 = np.ones((Qdim*N**2*layers,1))
Ψ0 = Ψ0/vec_norm(Ψ0)


# Evolution ------

Ψt = np.copy(Ψ0)
steps = 50
P = []

for i in range(steps):
    Pt = []
    for k in range(layers):
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

    
U = evolution_op@oracle 

# Initial State -----


layers = len(list_)
Ψ0 = np.ones((Qdim*N**2*layers,1))
Ψ0 = Ψ0/vec_norm(Ψ0)


# Evolution ------

Ψt = np.copy(Ψ0)
steps = 80
prob = [np.zeros((steps,len(list_[i]))) for i in range(layers)] # np.zeros((steps,layers))

for i in range(steps):    
    for k in range(layers):
        ψt = Ψt[k*Qdim*N**2:k*Qdim*N**2 + Qdim*N**2]
        list__ = list_[k]
        tot_marked = len(list__)
        for j in range(tot_marked):
            xx = np.zeros((N,1)); yy = np.zeros((N,1))
            coordinate = list__[j]
            xx[coordinate[0],0] = 1; yy[coordinate[1],0] = 1
            basis = np.kron(xx,yy)
            ϕ1 = np.kron(coin[:,[0]],basis); ϕ2 = np.kron(coin[:,[1]],basis); ϕ3 = np.kron(coin[:,[2]],basis); ϕ4 = np.kron(coin[:,[3]],basis)

            prob[k][i,j] = np.absolute(conjT(ϕ1)@ψt)**2 + np.absolute(conjT(ϕ2)@ψt)**2 + np.absolute(conjT(ϕ3)@ψt)**2 + np.absolute(conjT(ϕ4)@ψt)**2 
    Ψt = U@Ψt

    
    
xaxis = np.linspace(0,steps-1,steps)

fig, ax = plt.subplots(figsize = (6,4))
mx = []
for j in range(layers):
    p = prob[j]
    list__ = list_[j]
    tot_marked = len(list__)
    for i in range(tot_marked):
        ax.scatter(xaxis, p[:,i], color='white', s=50, zorder=2)
        ax.scatter(xaxis, p[:,i], color='black', s=10, zorder=3)
        ax.plot(xaxis,p[:,i], linestyle='-', linewidth=1, zorder=1,label = str(list__[i]) +' \& ' + str(j))
    mx.append(np.argmax(p[0:50,i]))
        
font2 = {'family':'serif','color':'black','size':15}
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='both', labelsize=15)
plt.legend(title = r'Target Points \& Layer',loc=(0.55,0.6),frameon = False) #,bbox_to_anchor=(1,0.5))
plt.xlabel(r'number of steps',fontdict = font2)
plt.ylabel(r'Probability',fontdict = font2)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define the Gaussian function
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) / std_dev)**2 / 2)

# Generate sample data
x = np.linspace(0,steps-1,steps)
fig, ax = plt.subplots(figsize = (6,4))

for j in range(layers):
    p = prob[j]
    list__ = list_[j]
    tot_marked = len(list__)
    for i in range(tot_marked):
        x = np.linspace(0,steps-1,steps)
        y = p[:,i]
        initial_guess = [1.0, 0.0, 1.0]  # Initial guess for amplitude, mean, and std_dev
        params, covariance = curve_fit(gaussian, x, y, p0=initial_guess)
        fitted_amplitude, fitted_mean, fitted_std_dev = params
        x = np.linspace(0,steps,4*steps)
        ax.plot(x, gaussian(x, fitted_amplitude, fitted_mean, fitted_std_dev),linestyle='-', linewidth=1, zorder=1,label = str(list__[i]) +' \& ' + str(j))
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='major', labelsize=15)
font2 = {'family':'serif','color':'black','size':15}
plt.xlabel(r'$t$',fontdict = font2)
plt.ylabel(r'$P(t)$',fontdict = font2)
plt.legend(title = r'Target Points \& Layer',loc='center left', bbox_to_anchor=(1, 0.5),frameon = False)

plt.title('Search algorithm on Torus',fontdict = font2)

plt.show()



fig, axs = plt.subplots(1, 4,figsize = (12,2))

for i in range(4):
    ax = axs[i]
    pcm = ax.imshow(P[mx[i]][i])
    fig.colorbar(pcm, ax=ax,fraction=0.046, pad=0.04)

plt.show()


# -------------- Dynamical Search algorithm ------------

_0 = np.array([[1],[0]]); _1 = np.array([[0],[1]])

_00 = _0@conjT(_0)
_01 = _0@conjT(_1)
_10 = _1@conjT(_0)
_11 = _1@conjT(_1)


# Define labels :

list_ = [[[6, 8]], [[8, 9]], [[12, 5]], [[15, 5]]]

N = 2**4 # lattice sites
n = len(list_) # number of layers

TPlus = np.roll(np.eye(N), 1, axis=0)
TMinus = np.roll(np.eye(N), -1, axis=0)
I = np.eye(N)

tplus = np.roll(np.eye(n), 1, axis=0)
tminus = np.roll(np.eye(n), -1, axis=0)
iden = np.eye(n)

# Constructing shift operator :

S = np.kron(ft.reduce(np.kron,[_10,_10,_10]), ft.reduce(np.kron,[I,TPlus,tplus])) + np.kron(ft.reduce(np.kron,[_10,_01,_10]), ft.reduce(np.kron,[TPlus,I,tplus])) + np.kron(ft.reduce(np.kron,[_10,_10,_01]), ft.reduce(np.kron,[I,TPlus,tminus])) + np.kron(ft.reduce(np.kron,[_10,_01,_01]), ft.reduce(np.kron,[TPlus,I,tminus])) + np.kron(ft.reduce(np.kron,[_01,_10,_10]), ft.reduce(np.kron,[TMinus,I,tplus])) + np.kron(ft.reduce(np.kron,[_01,_10,_01]), ft.reduce(np.kron,[TMinus,I,tminus])) + np.kron(ft.reduce(np.kron,[_01,_01,_10]), ft.reduce(np.kron,[I,TMinus,tplus])) + np.kron(ft.reduce(np.kron,[_01,_01,_01]), ft.reduce(np.kron,[I,TMinus,tminus])) 


# Is this unitary? :

print('Is shift operator unitary?' , np.allclose(S@conjT(S),np.eye(N**2*n*8)) )


def grover_coin_operator(n):
    N = 2 ** n
    H = np.ones((N, N)) / N  # Hadamard matrix (normalized)

    # Construct the Grover's coin operator
    coin_operator = 2 * H - np.identity(N)

    return coin_operator


# Grover's Coin Operator 

Qdim = 2**3
Sdim = N*N*n
dim = Qdim*Sdim

Grover_OP = grover_coin_operator(3) # 2*np.ones((Qdim,Qdim))/Qdim - np.eye(Qdim)

G = np.kron(Grover_OP,np.eye(Sdim))

# Evolution Operator : 

EVOLUTION = S@G


# Oracle -  
Ψc = np.ones((Qdim,1))
Ψc = Ψc/vec_norm(Ψc)

ΨcΨc = Ψc@conjT(Ψc)

mm = np.zeros((Sdim,Sdim))

for i in range(n):
    xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))
    coordinate = list_[i]
    for j in range(len(coordinate)):
        marked_point = coordinate[j]
        xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))
        xx[marked_point[0],0] = 1; yy[marked_point[1],0] = 1; zz[i,0] = 1
        basis = ft.reduce(np.kron,[xx,yy,zz]) # np.kron(xx,yy)
        mm = mm + basis@conjT(basis)

R = np.eye(dim) - 2*np.kron(ΨcΨc,mm) # np.kron((2*np.ones((Qdim,Qdim))/Qdim),mm)


# Total evolution operator :

U = EVOLUTION@R

# Initial State -----

Ψ0 = np.ones((dim,1))
Ψ0 = Ψ0/vec_norm(Ψ0)
coin = np.eye(Qdim)

# Evolution ------

Ψt = np.copy(Ψ0)
steps = 50
prob = [np.zeros((steps,len(list_[i]))) for i in range(n)] # np.zeros((steps,layers))

xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))

for i in tqdm(range(steps)):
    for k in range(n):
        coordinate = list_[k]
        for j in range(len(coordinate)):
            marked_point = coordinate[j]
            xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))
            xx[marked_point[0],0] = 1; yy[marked_point[1],0] = 1; zz[k,0] = 1
            basis = ft.reduce(np.kron,[xx,yy,zz])
            for ii in range(Qdim):
                ϕ = np.kron(coin[:,[ii]],basis)
                prob[k][i,j] = prob[k][i,j] + np.absolute(conjT(ϕ)@Ψt)**2
        
    Ψt = U@Ψt

    
xaxis = np.linspace(0,steps-1,steps)

fig, ax = plt.subplots(figsize = (6,4))
layers = len(list_)
for j in range(layers):
    p = prob[j]
    list__ = list_[j]
    tot_marked = len(list__)
    for i in range(tot_marked):
        ax.scatter(xaxis, p[:,i], color='white', s=50, zorder=2)
        ax.scatter(xaxis, p[:,i], color='black', s=10, zorder=3)
        ax.plot(xaxis,p[:,i], linestyle='-', linewidth=1, zorder=1,label = str(list__[i]) +' \& ' + str(j))

font2 = {'family':'serif','color':'black','size':15}
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='both', labelsize=15)
plt.legend(title = r'Target Points \& Layer',loc='best',frameon = False)
plt.xlabel(r'number of steps',fontdict = font2)
plt.ylabel(r'Probability',fontdict = font2)
plt.title('Periodic boundary condition',loc='left')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define the Gaussian function
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) / std_dev)**2 / 2)
mx = []
# Generate sample data
x = np.linspace(0,steps-1,steps)
fig, ax = plt.subplots(figsize = (6,4))

for j in range(layers):
    p = prob[j]
    list__ = list_[j]
    tot_marked = len(list__)
    for i in range(tot_marked):
        x = np.linspace(0,steps-1,steps)
        y = p[:,i]
        initial_guess = [1.0, 0.0, 1.0]  # Initial guess for amplitude, mean, and std_dev
        params, covariance = curve_fit(gaussian, x, y, p0=initial_guess)
        fitted_amplitude, fitted_mean, fitted_std_dev = params
        x = np.linspace(0,steps,4*steps)
        ax.plot(x, gaussian(x, fitted_amplitude, fitted_mean, fitted_std_dev),linestyle='-', linewidth=1, zorder=1,label = str(list__[i]) +' \& ' + str(j))
    mx.append(int(fitted_mean))
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='major', labelsize=15)
font2 = {'family':'serif','color':'black','size':15}
plt.xlabel(r'$t$',fontdict = font2)
plt.ylabel(r'$P(t)$',fontdict = font2)
plt.legend(title = r'Target Points \& Layer',loc='center left', bbox_to_anchor=(1, 0.5),frameon = False)

plt.title('Search algorithm on Torus',fontdict = font2)
plt.show()


# Initial State -----

Ψ0 = np.ones((dim,1))
Ψ0 = Ψ0/vec_norm(Ψ0)
coin = np.eye(Qdim)

# Evolution ------

Ψt = np.copy(Ψ0)
steps = 50
P = []

for i in range(steps):
    Pt = np.zeros((N,N,n),dtype = float)
    for kk in range(n):
        zz = np.zeros((n,1))
        zz[kk,0] = 1
        for ii in range(N):
            xx = np.zeros((N,1))
            xx[ii,0] = 1
            for jj in range(N):
                yy = np.zeros((N,1))
                yy[jj,0] = 1
                basis = ft.reduce(np.kron,[xx,yy,zz])
                for cc in range(Qdim):
                    ϕ = np.kron(coin[:,[cc]],basis)
                    Pt[ii,jj,kk] = Pt[ii,jj,kk] + np.absolute(conjT(ϕ)@Ψt)**2
    P.append(Pt)
    Ψt = U@Ψt

    
    fig, axs = plt.subplots(1, 4,figsize = (12,2))


for i in range(4):
    ax = axs[i]
    pcm = ax.imshow(P[mx[i]][:,:,i])
    fig.colorbar(pcm, ax=ax,fraction=0.046, pad=0.04)
plt.show()


# Open boundary condition 


N = 2**4 # lattice sites
n = len(list_) # number of layers

TPlus = np.diag(np.ones((1,N-1))[0],1)
TMinus = np.diag(np.ones((1,N-1))[0],-1)
# TPlus = np.roll(np.eye(N), 1, axis=0)
# TMinus = np.roll(np.eye(N), -1, axis=0)
I = np.eye(N)

tplus = np.roll(np.eye(n), 1, axis=0)
tminus = np.roll(np.eye(n), -1, axis=0)
iden = np.eye(n)

NN = np.zeros((N,N)); NN[N-1,N-1] = 1 
OO = np.zeros((N,N)); OO[0,0] = 1

Qdim = 2**3
Sdim = N*N*n
dim = Qdim*Sdim


def kron_delta(a,b):
    if a == b:
        return 1
    else:
        return 0

def coin_tr(i,j):
    if i == 0 and j == 0:
        return _00
    elif i == 0 and j == 1:
        return _01
    elif i == 1 and j == 0:
        return _10
    else:
        return _11
    
def space_tr(i):
    if i == 0:
        return I
    elif i == 1:
        return TPlus
    else:
        return TMinus

def layer_tr(i):
    if i == 0:
        return iden
    elif i == 1:
        return tplus
    else:
        return tminus


Sint = np.zeros((dim,dim))

for i in range(2):
    for j in range(2):
        for k in range(2):
            list1 = [coin_tr(1-i,i),coin_tr(1-j,j),coin_tr(1-k,k)]
            list2 = [space_tr(((-1)**i)*(1-kron_delta(i,j))),space_tr( ((-1)**i)*kron_delta(i,j)),layer_tr((-1)**k)]
            
            Sint = Sint + np.kron(ft.reduce(np.kron,list1),ft.reduce(np.kron,list2))
            

            
Sbdr = np.zeros((dim,dim))

for i in range(2):
    for j in range(2):
        for k in range(2):
            list1 = [coin_tr(i,i),coin_tr(j,j),coin_tr(1-k,k)]
            list2 = []
            
            if ((-1)**i)*(1-kron_delta(i,j)) == 0:
                list2.append(I)
            elif ((-1)**i)*(1-kron_delta(i,j)) == 1:
                list2.append(OO)
            elif ((-1)**i)*(1-kron_delta(i,j)) == -1:
                list2.append(NN)
                
            if ((-1)**i)*kron_delta(i,j) == 0:
                list2.append(I)
            elif ((-1)**i)*kron_delta(i,j) == 1:
                list2.append(OO)
            elif ((-1)**i)*kron_delta(i,j) == -1:
                list2.append(NN)
                
            list2.append(layer_tr((-1)**k))
            # if (-1)**k == 0:
            #     list2.append(iden)
            # elif (-1)**k == 1:
            #     list2.append(tplus)
            # elif (-1)**k == -1:
            #     list2.append(tminus)
            
        
            Sbdr = Sbdr + np.kron(ft.reduce(np.kron,list1),ft.reduce(np.kron,list2))

            
            
S = Sint + Sbdr

# Is this unitary? :

print('Is shift operator unitary?' , np.allclose(S@conjT(S),np.eye(dim)))


# Grover's Coin Operator 

Qdim = 2**3
Sdim = N*N*n
dim = Qdim*Sdim

Grover_OP = grover_coin_operator(3) # 2*np.ones((Qdim,Qdim))/Qdim - np.eye(Qdim)

G = np.kron(Grover_OP,np.eye(Sdim))

# Evolution Operator : 

EVOLUTION = S@G


# Oracle -  
Ψc = np.ones((Qdim,1))
Ψc = Ψc/vec_norm(Ψc)

ΨcΨc = Ψc@conjT(Ψc)

mm = np.zeros((Sdim,Sdim))

for i in range(n):
    xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))
    coordinate = list_[i]
    for j in range(len(coordinate)):
        marked_point = coordinate[j]
        xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))
        xx[marked_point[0],0] = 1; yy[marked_point[1],0] = 1; zz[i,0] = 1
        basis = ft.reduce(np.kron,[xx,yy,zz]) # np.kron(xx,yy)
        mm = mm + basis@conjT(basis)

R = np.eye(dim) - 2*np.kron(ΨcΨc,mm) # np.kron((2*np.ones((Qdim,Qdim))/Qdim),mm)

# Total evolution operator :

U = EVOLUTION@R

# Is this unitary? :

print('Is evolution operator unitary?' , np.allclose(U@conjT(U),np.eye(dim)) )


# Initial State -----

Ψ0 = np.ones((dim,1))
Ψ0 = Ψ0/vec_norm(Ψ0)
coin = np.eye(Qdim)

# Evolution ------

Ψt = np.copy(Ψ0)
steps = 50
prob = [np.zeros((steps,len(list_[i]))) for i in range(n)] # np.zeros((steps,layers))

xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))

for i in tqdm(range(steps)):
    for k in range(n):
        coordinate = list_[k]
        for j in range(len(coordinate)):
            marked_point = coordinate[j]
            xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))
            xx[marked_point[0],0] = 1; yy[marked_point[1],0] = 1; zz[k,0] = 1
            basis = ft.reduce(np.kron,[xx,yy,zz])
            for ii in range(Qdim):
                ϕ = np.kron(coin[:,[ii]],basis)
                prob[k][i,j] = prob[k][i,j] + np.absolute(conjT(ϕ)@Ψt)**2
        
    Ψt = U@Ψt

    
xaxis = np.linspace(0,steps-1,steps)

fig, ax = plt.subplots(figsize = (6,4))

for j in range(layers):
    p = prob[j]
    list__ = list_[j]
    tot_marked = len(list__)
    for i in range(tot_marked):
        ax.scatter(xaxis, p[:,i], color='white', s=50, zorder=2)
        ax.scatter(xaxis, p[:,i], color='black', s=10, zorder=3)
        ax.plot(xaxis,p[:,i], linestyle='-', linewidth=1, zorder=1,label = str(list__[i]) +' \& ' + str(j))

font2 = {'family':'serif','color':'black','size':15}
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='both', labelsize=15)
plt.legend(title = r'Target Points \& Layer',loc='best',frameon = False)
plt.xlabel(r'number of steps',fontdict = font2)
plt.ylabel(r'Probability',fontdict = font2)
plt.title('Open boundary condition',loc='left')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define the Gaussian function
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) / std_dev)**2 / 2)
mx = []
# Generate sample data
x = np.linspace(0,steps-1,steps)
fig, ax = plt.subplots(figsize = (6,4))

for j in range(layers):
    p = prob[j]
    list__ = list_[j]
    tot_marked = len(list__)
    for i in range(tot_marked):
        x = np.linspace(0,steps-1,steps)
        y = p[:,i]
        initial_guess = [1.0, 0.0, 1.0]  # Initial guess for amplitude, mean, and std_dev
        params, covariance = curve_fit(gaussian, x, y, p0=initial_guess)
        fitted_amplitude, fitted_mean, fitted_std_dev = params
        x = np.linspace(0,steps,4*steps)
        ax.plot(x, gaussian(x, fitted_amplitude, fitted_mean, fitted_std_dev),linestyle='-', linewidth=1, zorder=1,label = str(list__[i]) +' \& ' + str(j))
    mx.append(int(fitted_mean))
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='major', labelsize=15)
font2 = {'family':'serif','color':'black','size':15}
plt.xlabel(r'$t$',fontdict = font2)
plt.ylabel(r'$P(t)$',fontdict = font2)
plt.legend(title = r'Target Points \& Layer',loc='center left', bbox_to_anchor=(1, 0.5),frameon = False)

plt.title('Search algorithm on Torus',fontdict = font2)
plt.show()



# Initial State -----

Ψ0 = np.ones((dim,1))
Ψ0 = Ψ0/vec_norm(Ψ0)
coin = np.eye(Qdim)

# Evolution ------

Ψt = np.copy(Ψ0)
steps = 50
P = []

for i in tqdm(range(steps)):
    Pt = np.zeros((N,N,n),dtype = float)
    for kk in range(n):
        zz = np.zeros((n,1))
        zz[kk,0] = 1
        for ii in range(N):
            xx = np.zeros((N,1))
            xx[ii,0] = 1
            for jj in range(N):
                yy = np.zeros((N,1))
                yy[jj,0] = 1
                basis = ft.reduce(np.kron,[xx,yy,zz])
                for cc in range(Qdim):
                    ϕ = np.kron(coin[:,[cc]],basis)
                    Pt[ii,jj,kk] = Pt[ii,jj,kk] + np.absolute(conjT(ϕ)@Ψt)**2
    P.append(Pt)
    Ψt = U@Ψt


    
    fig, axs = plt.subplots(1, 4,figsize = (12,2))


for i in range(4):
    ax = axs[i]
    pcm = ax.imshow(P[mx[i]][:,:,i])
    fig.colorbar(pcm, ax=ax,fraction=0.046, pad=0.04)

plt.show()


# Periodic boundary conditon - Another approach 


N = 2**4 # lattice sites
n = len(list_) # number of layers

TPlus = np.roll(np.eye(N), 1, axis=0)
TMinus = np.roll(np.eye(N), -1, axis=0)
I = np.eye(N)

tplus = np.roll(np.eye(n), 1, axis=0)
tminus = np.roll(np.eye(n), -1, axis=0)
iden = np.eye(n)


def kron_delta(a,b):
    if a == b:
        return 1
    else:
        return 0

def coin_tr(i,j):
    if i == 0 and j == 0:
        return _00
    elif i == 0 and j == 1:
        return _01
    elif i == 1 and j == 0:
        return _10
    else:
        return _11
    
def space_tr(i):
    if i == 0:
        return I
    elif i == 1:
        return TPlus
    else:
        return TMinus

def layer_tr(i):
    if i == 0:
        return iden
    elif i == 1:
        return tplus
    else:
        return tminus


    
S = np.zeros((dim,dim),dtype = complex)

for i in range(2):
    for j in range(2):
        for k in range(2):
            list1 = [coin_tr(1-i,i),coin_tr(1-j,j),coin_tr(1-k,k)]
            list2 = [space_tr(((-1)**i)*(1-kron_delta(i,j))),space_tr( ((-1)**i)*kron_delta(i,j)),layer_tr((-1)**k)]
            S = S + np.kron(ft.reduce(np.kron,list1),ft.reduce(np.kron,list2))

            
            # Is this unitary? :

print('Is shift operator unitary?' , np.allclose(S@conjT(S),np.eye(dim)))


# Grover's Coin Operator 

Qdim = 2**3
Sdim = N*N*n
dim = Qdim*Sdim

Grover_OP = grover_coin_operator(3) # 2*np.ones((Qdim,Qdim))/Qdim - np.eye(Qdim)

G = np.kron(Grover_OP,np.eye(Sdim))

# Evolution Operator : 

EVOLUTION = S@G


# Oracle -  
Ψc = np.ones((Qdim,1))
Ψc = Ψc/vec_norm(Ψc)

ΨcΨc = Ψc@conjT(Ψc)

mm = np.zeros((Sdim,Sdim))

for i in range(n):
    xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))
    coordinate = list_[i]
    for j in range(len(coordinate)):
        marked_point = coordinate[j]
        xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))
        xx[marked_point[0],0] = 1; yy[marked_point[1],0] = 1; zz[i,0] = 1
        basis = ft.reduce(np.kron,[xx,yy,zz]) # np.kron(xx,yy)
        mm = mm + basis@conjT(basis)

R = np.eye(dim) - 2*np.kron(ΨcΨc,mm) # np.kron((2*np.ones((Qdim,Qdim))/Qdim),mm)


# Total evolution operator :

U = EVOLUTION@R


# Is this unitary? :

print('Is evolution operator unitary?' , np.allclose(U@conjT(U),np.eye(dim)) )


# Initial State -----

Ψ0 = np.ones((dim,1))
Ψ0 = Ψ0/vec_norm(Ψ0)
coin = np.eye(Qdim)

# Evolution ------

Ψt = np.copy(Ψ0)
steps = 50
prob = [np.zeros((steps,len(list_[i]))) for i in range(n)] # np.zeros((steps,layers))

xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))

for i in range(steps):
    for k in range(n):
        coordinate = list_[k]
        for j in range(len(coordinate)):
            marked_point = coordinate[j]
            xx = np.zeros((N,1)); yy = np.zeros((N,1)); zz = np.zeros((n,1))
            xx[marked_point[0],0] = 1; yy[marked_point[1],0] = 1; zz[k,0] = 1
            basis = ft.reduce(np.kron,[xx,yy,zz])
            for ii in range(Qdim):
                ϕ = np.kron(coin[:,[ii]],basis)
                prob[k][i,j] = prob[k][i,j] + np.absolute(conjT(ϕ)@Ψt)**2
        
    Ψt = U@Ψt

    
    xaxis = np.linspace(0,steps-1,steps)

fig, ax = plt.subplots(figsize = (6,4))

for j in range(layers):
    p = prob[j]
    list__ = list_[j]
    tot_marked = len(list__)
    for i in range(tot_marked):
        ax.scatter(xaxis, p[:,i], color='white', s=100, zorder=2)
        ax.scatter(xaxis, p[:,i], color='black', s=20, zorder=3)
        ax.plot(xaxis,p[:,i], linestyle='-', linewidth=1, zorder=1,label = str(list__[i]) +' \& ' + str(j))

font2 = {'family':'serif','color':'black','size':15}
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='both', labelsize=15)
plt.legend(title = r'Target Points \& Layer',loc='center left', bbox_to_anchor=(1, 0.5),frameon = False)
# plt.xlim(0,20)
plt.xlabel(r'$N$',fontdict = font2)
plt.ylabel(r'$P(t)$',fontdict = font2)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define the Gaussian function
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) / std_dev)**2 / 2)
mx = []
# Generate sample data
x = np.linspace(0,steps-1,steps)
fig, ax = plt.subplots(figsize = (6,4))

for j in range(layers):
    p = prob[j]
    list__ = list_[j]
    tot_marked = len(list__)
    for i in range(tot_marked):
        x = np.linspace(0,steps-1,steps)
        y = p[:,i]
        initial_guess = [1.0, 0.0, 1.0]  # Initial guess for amplitude, mean, and std_dev
        params, covariance = curve_fit(gaussian, x, y, p0=initial_guess)
        fitted_amplitude, fitted_mean, fitted_std_dev = params
        x = np.linspace(0,steps,4*steps)
        ax.plot(x, gaussian(x, fitted_amplitude, fitted_mean, fitted_std_dev),linestyle='-', linewidth=1, zorder=1,label = str(list__[i]) +' \& ' + str(j))
    mx.append(int(fitted_mean))
plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='major', labelsize=15)
font2 = {'family':'serif','color':'black','size':15}
plt.xlabel(r'$t$',fontdict = font2)
plt.ylabel(r'$P(t)$',fontdict = font2)
plt.legend(title = r'Target Points \& Layer',loc='center left', bbox_to_anchor=(1, 0.5),frameon = False)

plt.title('Search algorithm on Torus',fontdict = font2)
plt.show()



# Initial State -----

Ψ0 = np.ones((dim,1))
Ψ0 = Ψ0/vec_norm(Ψ0)
coin = np.eye(Qdim)

# Evolution ------

Ψt = np.copy(Ψ0)
steps = 50
P = []

for i in range(steps):
    Pt = np.zeros((N,N,n),dtype = float)
    for kk in range(n):
        zz = np.zeros((n,1))
        zz[kk,0] = 1
        for ii in range(N):
            xx = np.zeros((N,1))
            xx[ii,0] = 1
            for jj in range(N):
                yy = np.zeros((N,1))
                yy[jj,0] = 1
                basis = ft.reduce(np.kron,[xx,yy,zz])
                for cc in range(Qdim):
                    ϕ = np.kron(coin[:,[cc]],basis)
                    Pt[ii,jj,kk] = Pt[ii,jj,kk] + np.absolute(conjT(ϕ)@Ψt)**2
    P.append(Pt)
    Ψt = U@Ψt

    
    fig, axs = plt.subplots(1, 4,figsize = (12,2))


for i in range(4):
    ax = axs[i]
    pcm = ax.imshow(P[mx[i]][:,:,i])
    fig.colorbar(pcm, ax=ax,fraction=0.046, pad=0.04)

plt.show()