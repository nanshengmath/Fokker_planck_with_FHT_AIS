import numpy as np
import os
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Given constants
n = 4
d = n**3
h = 1 / (n + 1)
λ = 0.01
β0 = 6  # High temperature warm-up 

β = 12  # Low temperature
N_AIS = 100 
dβ = (β - β0) / N_AIS
V_grid = np.arange(β0,β+dβ,dβ) / β  # grid points used to compute effective potential

T = 1
scale = 200
dt = 0.0005 * scale / β0
N = int(T*scale/dt)  # Interested in T = 1
dt = T*scale/N  # Correct for small deviation

l_MALA = True  # MALA option
l_AIS = True  # AIS option
N_AIS_MALA = 1  # Number of MALA steps per AIS step

BATCH_SIZE = 100  # Set your desired batch size

stretch = 10

data_path = f"./Saved_Data/1D_GZ_data_d_{d}.npz"

# Hyperparameters
hyperparameters = {
    "d": d,
    "h": h,
    "λ": λ,
    "β": β,
    "dt": dt,
    "N": N,
    "BATCH_SIZE": BATCH_SIZE
}

# Create directory if it doesn't exist
if rank == 0:
    if not os.path.exists("./Saved_Data"):
        os.makedirs("./Saved_Data")
    
    if os.path.exists(data_path):
        os.remove(data_path)

local_samples_β0 = np.empty((0, n + 2, n + 2, n + 2))
if l_AIS:
    local_samples = np.empty((0, n + 2, n + 2, n + 2))

def padding(U):
    if np.size(U.shape) == 1:
        U = U[np.newaxis,:]
    dim0, dim1 = U.shape
    dim2 = round(np.power(dim1,1/3))

    U_tmp = np.zeros((dim0, dim2+2, dim2+2, dim2+2))
    U_tmp[:,1:-1,1:-1,1:-1] = U.reshape((dim0, dim2, dim2, dim2))
    return U_tmp

# 2D Ginzburg Landau
def V(U, AIS_step=0):
    
    U = padding(U)
    
    return V_grid[AIS_step] * β / d * ( λ/h**2/2 * (
        np.sum((U - np.roll(U, 1, axis=1))**2, axis = (1,2,3)) + np.sum((U - np.roll(U, -1, axis=1))**2, axis = (1,2,3))
        + np.sum((U - np.roll(U, 1, axis=2))**2, axis = (1,2,3)) + np.sum((U - np.roll(U, -1, axis=2))**2, axis = (1,2,3))
        + np.sum((U - np.roll(U, 1, axis=3))**2, axis = (1,2,3)) + np.sum((U - np.roll(U, -1, axis=3))**2, axis = (1,2,3))
    ) + np.sum(((1 - U**2)**2)[:,1:-1,1:-1,1:-1], axis = (1,2,3)) /4/λ )

def V_grad(U, AIS_step=0):
    
    U = padding(U)

    return V_grid[AIS_step] * β / d * ( λ/h**2 * (
        6*U - np.roll(U, 1, axis=1) - np.roll(U, -1, axis=1) 
        - np.roll(U, -1, axis=2) - np.roll(U, 1, axis=2)
        - np.roll(U, -1, axis=3) - np.roll(U, 1, axis=3)
    ) - U * (1 - U**2) /λ )[:,1:-1,1:-1,1:-1].reshape((U.shape[0], -1))

def Langevin_step(U,V,V_grad,l_MALA=True):

    if np.size(U.shape) == 1:
        U = U[np.newaxis,:]
    
    dW = np.sqrt(dt) * np.random.randn(U.shape[0], d)
    
    U_tmp = U.copy()
    U_tmp += (-V_grad(U_tmp) * dt + np.sqrt(2) * dW)

    count = np.sum(U_tmp > 2.5) + np.sum(U_tmp <-2.5)
    U_tmp[U_tmp > 2.5] = 2.5
    U_tmp[U_tmp < -2.5] = -2.5

    if l_MALA:
        log_acceptance_ratio = np.minimum(0.0, - (V(U_tmp) - V(U)) \
                                      - 1/4/dt * (np.sum(((U - U_tmp + dt * V_grad(U_tmp)))**2,axis=1) \
                                                      - np.sum(((U_tmp - U + dt * V_grad(U)))**2,axis=1)))
        # Accept or reject
        accept = np.log(np.random.uniform(size=U.shape[0])) < log_acceptance_ratio
        
        # print(np.sum(accept))
    
        return np.where(accept[:, None], U_tmp, U), count
    else:
        return U_tmp, count

def Snooker(U,V_AIS_tmp,i,candid):

    u = np.random.uniform(0, 1)
    z = (u * np.sqrt(stretch) + (1 / np.sqrt(stretch)) * (1 - u))**2

    rand_ind = np.random.choice(candid)
    U_tmp = (1 - z) * U[rand_ind,:] + z * U[i,:]

    log_acceptance_ratio = np.minimum(0.0, (d-1) * np.log(z) - (V_AIS_tmp(U_tmp) - V_AIS_tmp(U[i,:]))) 
    # Accept or reject
    accept = np.log(np.random.uniform(size=1)) < log_acceptance_ratio

    if accept:
        U[i,:] = U_tmp
    
    return U

def BD(U,i,candid):
    rate = V(U,-1) - V(U,0)
    rate_i = rate[i] - np.mean(rate)

    if rate_i > 0:
        if np.log(np.random.uniform(size=1)) > - rate_i / N_AIS:
            rand_ind = np.random.choice(candid)
            U[i,:] = U[rand_ind,:]
    else:
        if np.log(np.random.uniform(size=1)) > rate_i / N_AIS:
            rand_ind = np.random.choice(candid)
            U[rand_ind,:] = U[i,:]
            
    return U

count = 0

for iteration in range(30//size + 1):
    
    # U = np.zeros((BATCH_SIZE, d))  # Initialize at each iteration
    U = np.ones((BATCH_SIZE, d))  # Initialize at each iteration
    
    # Creating samples for 1D Ginzburg Landau
    for _ in range(N):
        U, count_ = Langevin_step(U,V,V_grad,l_MALA)
        count += count_

    print(f'Saved data at rank {rank} and iteration {iteration}, total violation is {count}')

    U_tmp = np.zeros((BATCH_SIZE, n + 2, n + 2, n + 2))
    U_tmp[:,1:-1,1:-1,1:-1] = U.reshape((BATCH_SIZE, n, n, n))    
    local_samples_β0 = np.concatenate((local_samples_β0, U_tmp), axis=0)

    if l_AIS:
        
        for AIS_step in range(1,N_AIS+1):

            V_AIS_tmp = lambda U : V(U, AIS_step)
            V_grad_AIS_tmp = lambda U : V_grad(U, AIS_step)
                
            for i in range(BATCH_SIZE):

                candid = np.delete(np.arange(BATCH_SIZE), i)

                # MALA and Snooker
                for _ in range(N_AIS_MALA):
                    U_, _ = Langevin_step(U[i-1,:],V_AIS_tmp,V_grad_AIS_tmp,l_MALA) 
                    U_tmp = U.copy()
                    U_tmp[i-1,:] = U_
        
                    U_tmp = Snooker(U_tmp,V_AIS_tmp,i,candid)

                # Birth death
                U = BD(U_tmp,i,candid)

        U_tmp = np.zeros((BATCH_SIZE, n + 2, n + 2, n + 2))
        U_tmp[:,1:-1,1:-1,1:-1] = U.reshape((BATCH_SIZE, n, n, n))  
        local_samples = np.concatenate((local_samples, U_tmp), axis=0)

all_samples = comm.gather(local_samples, root=0)
all_samples_β0 = comm.gather(local_samples_β0, root=0)

if rank == 0:

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples_β0 = np.concatenate(all_samples_β0, axis=0)
    
    # Save the accumulated data
    if l_AIS:
        np.savez(data_path, samples=all_samples, hyperparameters=hyperparameters)
    else:
        np.savez(data_path, samples=all_samples_β0, hyperparameters=hyperparameters)

    print(f'total number of violation is {count}')
    
    # Visualization of the (n/2,n/2)-th marginal distribution
    samples_1_mar = all_samples_β0[:, round(n/2), round(n/2), round(n/2)]  # Extract the d/2-th bit from each sample
    plt.hist(samples_1_mar, bins=50, density=True)
    plt.title(f"{round(n/2),round(n/2),round(n/2)}-th Marginal Distribution")
    plt.savefig('test1.png',bbox_inches='tight')
    plt.clf()
    
    if l_AIS:
        
        # Visualization of the (d/2)-th marginal distribution
        samples_1_mar = all_samples[:, round(n/2), round(n/2), round(n/2)]  # Extract the (n/2,n/2)-th bit from each sample
        plt.hist(samples_1_mar, bins=50, density=True)
        plt.title(f"{round(n/2),round(n/2),round(n/2)}-th Marginal Distribution")
        plt.savefig('test2.png',bbox_inches='tight')
        plt.clf()
        