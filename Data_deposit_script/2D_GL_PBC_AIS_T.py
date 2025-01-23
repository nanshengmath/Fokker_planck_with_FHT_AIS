import numpy as np
import os
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Given constants
n = 16
d = n**2
h = 1 / n
λ = h / 8
β0 = 20

T = 12
scale = 150 / β0 
dt = 0.0005 * scale
N_step = int(T*scale/dt)  
dt = T*scale/N_step  # Correct for small deviation

β = 36  # Low temperature
N_AIS = 10 
V_grid = np.geomspace(β0,β,N_AIS+1) / β  # grid points used to compute effective potential

l_MALA = True  # MALA option
N_AIS_MALA = int(np.ceil(1/N_AIS/dt))  # Number of MALA steps per AIS step
N_AIS_MALA_LAST = 700
  
stretch = 10

BATCH_SIZE = 100  # Set your desired batch size
N_BATCH = 6000 // BATCH_SIZE

data_path = f"./Saved_Data/2D_GZ_data_d_{d}"

# Hyperparameters
hyperparameters = {
    "d": d,
    "h": h,
    "λ": λ,
    "β0": β0,
    "β": β,
    "dt": dt,
    "N_step": N_step,
    "BATCH_SIZE": BATCH_SIZE
}

# Create directory if it doesn't exist
if rank == 0:
    if not os.path.exists("./Saved_Data"):
        os.makedirs("./Saved_Data")

    with np.load(data_path+f'_MALA_β_{β0}_T_{T}.npz') as data:
        all_samples_β0 = data['samples_β0']
    all_samples_β0 = all_samples_β0.reshape(all_samples_β0.shape[0],-1)
else:
    all_samples_β0 = None

N_BATCH_NODE = N_BATCH // size
if rank < N_BATCH % size:
    N_BATCH_NODE += 1

local_samples_β0 = np.empty((N_BATCH_NODE*BATCH_SIZE,d))

comm.Scatter(all_samples_β0, local_samples_β0, root=0)  # Scatter the data

if rank == 0:
    all_samples_β0 = all_samples_β0.reshape(all_samples_β0.shape[0],n,n)

def padding(U):
    if np.size(U.shape) == 1:
        U = U[np.newaxis,:]

    dim0, dim1 = U.shape
    dim2 = round(np.sqrt(dim1))

    return U.reshape((dim0, dim2, dim2))

# 2D Ginzburg Landau
def V(U, AIS_step=0):
    U = padding(U)
    return V_grid[AIS_step] * β * ( λ/2 * (
        np.sum((U - np.roll(U, 1, axis=1))**2, axis = (1,2)) + np.sum((U - np.roll(U, -1, axis=1))**2, axis = (1,2))
        + np.sum((U - np.roll(U, 1, axis=2))**2, axis = (1,2)) + np.sum((U - np.roll(U, -1, axis=2))**2, axis = (1,2))
    ) + h**2 * np.sum(((1 - U**2)**2), axis = (1,2)) /4/λ )

def V_grad(U, AIS_step=0):
    U = padding(U)
    return V_grid[AIS_step] * β * ( λ * (
        4*U - np.roll(U, 1, axis=1) - np.roll(U, -1, axis=1) - np.roll(U, 1, axis=2) - np.roll(U, -1, axis=2)
    ) - h**2 * U * (1 - U**2) /λ ).reshape((U.shape[0], -1))

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

def BD(U,i,candid,AIS_step):
    rate = V(U,-1) - V(U,0)
    rate_i = rate[i] - np.mean(rate)

    grid_coeff = β0/(β-β0) * (β/β0)**(AIS_step/N_AIS) * np.log(β/β0) 
    if rate_i > 0:
        if np.log(np.random.uniform(size=1)) > - grid_coeff * rate_i / N_AIS:
            rand_ind = np.random.choice(candid)
            U[i,:] = U[rand_ind,:]
    else:
        if np.log(np.random.uniform(size=1)) > grid_coeff * rate_i / N_AIS:
            rand_ind = np.random.choice(candid)
            U[rand_ind,:] = U[i,:]
            
    return U

def transit(y):
    dim0 = y.shape[0]
    y = y.reshape(dim0,-1)
    g1 = np.mean( np.exp(-2/d * np.sum((y-1)**2,axis=1) ))
    g2 = np.mean( np.exp(-2/d * np.sum((y+1)**2,axis=1) ))
    return(g1/(g1+g2))

count = 0

vars_to_save = {}

if rank == 0:
    data = [transit(all_samples_β0)]

local_samples = np.empty((0, n, n))

for AIS_step in range(1,N_AIS+1):

    for iteration in range(N_BATCH_NODE):
    
        if AIS_step == 1:
            U = local_samples_β0[iteration*BATCH_SIZE:(iteration+1)*BATCH_SIZE, :]
        else:
            U = local_samples_save[iteration*BATCH_SIZE:(iteration+1)*BATCH_SIZE, :]

        V_AIS_tmp = lambda U : V(U, AIS_step)
        V_grad_AIS_tmp = lambda U : V_grad(U, AIS_step)
            
        for i in range(BATCH_SIZE):

            candid = np.delete(np.arange(BATCH_SIZE), i)

            # MALA and Snooker
            for _ in range(N_AIS_MALA):
                U_, _ = Langevin_step(U[i,:],V_AIS_tmp,V_grad_AIS_tmp,False) 
                U_tmp = U.copy()
                U_tmp[i,:] = U_
    
            U_tmp = Snooker(U_tmp,V_AIS_tmp,i,candid)

            # Birth death
            U = BD(U_tmp,i,candid,AIS_step)

        for _ in range(N_AIS_MALA_LAST):
            U, _ = Langevin_step(U,V_AIS_tmp,V_grad_AIS_tmp,l_MALA) 

        local_samples = np.concatenate((local_samples, padding(U)), axis=0)

    all_samples = comm.gather(local_samples, root=0)
    local_samples_save = local_samples.copy()
    local_samples_save = local_samples_save.reshape(-1, d)
    local_samples = np.empty((0, n, n))

    if rank == 0:
        
        # Save the accumulated data
        all_samples = np.concatenate(all_samples, axis=0)
        data.append(transit(all_samples))
    
if rank == 0:

    # Save the accumulated data
    vars_to_save[f'samples_β0'] = all_samples_β0
    vars_to_save[f'samples'] = all_samples
    np.savez(data_path+f'_β0_{β0}_β_{β}_T_{T}.npz', **vars_to_save, hyperparameters=hyperparameters)

    # Visualization of the (n/2,n/2)-th marginal distribution
    samples_1_mar = all_samples_β0[:, round(n/2), round(n/2)]  # Extract the d/2-th bit from each sample
    plt.hist(samples_1_mar, bins=50, density=True)
    plt.title(f"{round(n/2),round(n/2)}-th Marginal Distribution")
    plt.savefig(f'test1_T_{T}.png',bbox_inches='tight')
    plt.clf()
        
    # Visualization of the (n/2,n/2)-th marginal distribution
    samples_1_mar = all_samples[:, round(n/2), round(n/2)]  # Extract the d/2-th bit from each sample
    plt.hist(samples_1_mar, bins=50, density=True)
    plt.title(f"{round(n/2),round(n/2)}-th Marginal Distribution")
    plt.savefig(f'test2_T_{T}.png',bbox_inches='tight')
    plt.clf()

    print(data)
        