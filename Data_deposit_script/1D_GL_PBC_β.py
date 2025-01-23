import numpy as np
import os
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Given constants
d = 256
h = 1 / d
λ = 0.5 * h
β0 = 3  # High temperature warm-up  # 3 for d = 256

l_MALA = True  # MALA option
l_AIS = False  # AIS option
N_AIS_MALA = 1  # Number of MALA steps per AIS step
N_AIS_MALA_LAST = 0

β_list = [4,5,7,8]

for β in β_list:

    if rank == 0:
        print(f'β = {β}')

    vars_to_save = {}

    local_samples_β0 = np.empty((0, d))
    if l_AIS:
        local_samples = np.empty((0, d))

    if l_AIS:
        N_AIS = 100 
        V_grid = np.linspace(β0,β,N_AIS+1) / β  # grid points used to compute effective potential
    else:
        β0 = β
        N_AIS = 0
        V_grid = np.array([1.])
    
    T = 20
    scale = 60 / β0 # 80 for d = 16 & 120 for d = 64 & 50 for d = 256
    dt = 0.0005 * scale
    N_step = int(T*scale/dt)  
    dt = T*scale/N_step  # Correct for small deviation
    
    BATCH_SIZE = 100  # Set your desired batch size
    N_BATCH = 6000 // BATCH_SIZE
    
    stretch = 10
    
    if l_AIS:
        data_path = f"./Saved_Data/1D_GZ_data_d_{d}_β_{β}.npz"
    else:
        data_path = f"./Saved_Data/1D_GZ_data_d_{d}_MALA_β_{β}.npz"
    
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
        
        if os.path.exists(data_path):
            os.remove(data_path)
    
    def padding(U):
        if np.size(U.shape) == 1:
            U = U[np.newaxis,:]
        return U
    
    def V(U, AIS_step=0):
        U = padding(U)
        return V_grid[AIS_step] * β * ( λ/h/2 * np.sum((U - np.roll(U, 1, axis = 1))**2, axis = 1) + h * np.sum(((1 - U**2)**2), axis = 1) /4/λ )
            
    def V_grad(U, AIS_step=0):
        U = padding(U)
        return V_grid[AIS_step] * β * ( λ/h * (2*U - np.roll(U, 1, axis = 1) - np.roll(U, -1, axis = 1)) - h * U * (1 - U**2) /λ )
    
    def Langevin_step(U,V,V_grad,l_MALA=True):
        U = padding(U)
        
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

    def transit(y):
        g1 = np.mean( np.exp(-2/d * np.sum((y-1)**2,axis=1) ))
        g2 = np.mean( np.exp(-2/d * np.sum((y+1)**2,axis=1) ))
        return(g1/(g1+g2))
    
    count = 0
    
    N_BATCH_NODE = N_BATCH // size
    if rank < N_BATCH % size:
        N_BATCH_NODE += 1
    
    for iteration in range(N_BATCH_NODE):
        
        U = np.ones((BATCH_SIZE, d))  # Initialize at each iteration
    
        # Creating samples for 1D Ginzburg Landau
        for _ in range(N_step):
            U, count_ = Langevin_step(U,V,V_grad,l_MALA)
            count += count_
    
        print(f'Saved data at rank {rank} and iteration {iteration}, total violation is {count}')
    
        local_samples_β0 = np.concatenate((local_samples_β0, U), axis=0)
    
        # Save the accumulated data
        if l_AIS:
                
            for AIS_step in range(1,N_AIS+1):
    
                V_AIS_tmp = lambda U : V(U, AIS_step)
                V_grad_AIS_tmp = lambda U : V_grad(U, AIS_step)
                    
                for i in range(BATCH_SIZE):
    
                    candid = np.delete(np.arange(BATCH_SIZE), i)
    
                    # MALA and Snooker
                    for _ in range(N_AIS_MALA):
                        U_, _ = Langevin_step(U[i,:],V_AIS_tmp,V_grad_AIS_tmp,l_MALA) 
                        U_tmp = U.copy()
                        U_tmp[i,:] = U_
            
                        U_tmp = Snooker(U_tmp,V_AIS_tmp,i,candid)
    
                    # Birth death
                    U = BD(U_tmp,i,candid)

                for _ in range(N_AIS_MALA_LAST):
                    U, _ = Langevin_step(U,V_AIS_tmp,V_grad_AIS_tmp,l_MALA) 
          
            local_samples = np.concatenate((local_samples, U), axis=0)  
    
    all_samples_β0 = comm.gather(local_samples_β0, root=0)
    if l_AIS:
        all_samples = comm.gather(local_samples, root=0)
    
    if rank == 0:
    
        all_samples_β0 = np.concatenate(all_samples_β0, axis=0)
        
        # Save the accumulated data
        if l_AIS:
            all_samples = np.concatenate(all_samples, axis=0)
            vars_to_save[f'samples_β0'] = all_samples_β0
            vars_to_save[f'samples'] = all_samples
        else:
            vars_to_save[f'samples_β0'] = all_samples_β0
            vars_to_save[f'samples'] = all_samples_β0
    
        print(f'total number of violation is {count}')

        np.savez(data_path, **vars_to_save, hyperparameters=hyperparameters)

        # Visualization of the (d/2)-th marginal distribution
        samples_1_mar = all_samples_β0[:, round(d/2)]  # Extract the d/2-th bit from each sample
        plt.hist(samples_1_mar, bins=50, density=True)
        plt.title(f"{round(d/2)}-th Marginal Distribution")
        plt.savefig(f'test1_β_{β}.png',bbox_inches='tight')
        plt.clf()
        
        if l_AIS:
            
            # Visualization of the (d/2)-th marginal distribution
            samples_1_mar = all_samples[:, round(d/2)]  # Extract the d/2-th bit from each sample
            plt.hist(samples_1_mar, bins=50, density=True)
            plt.title(f"{round(d/2)}-th Marginal Distribution")
            plt.savefig(f'test2_β_{β}.png',bbox_inches='tight')
            plt.clf()
                    