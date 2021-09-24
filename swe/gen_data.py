import numpy as np
import torch
import hydra
import os

from src.utils import swe2D_sim

import pytorch_lightning as pl

# make sure this seed different from the eval_exps_01.ipnb seed or else training data = test data!
pl.seed_everything(42) 

# read config from hydra
with hydra.initialize(config_path='configs'):
    cfg = hydra.compose(config_name='defaults.yaml')

n_static_channels = 2 # bed depth potential and boundary mask
    
# load model to get the necessary logistics for Red-Black Gauss Seidel solver
model = hydra.utils.instantiate(
        cfg.model,
        instance_dimensionality = cfg.data.instance_dimensionality,
        input_channels = 5,
        static_channels = n_static_channels,
        offset = [1],
        _recursive_ = False # necessary for model.configure_optimizers()
)
settings = model.impl_layers[0].settings
del model


# correct initializer for linear solver outside model (model wasn't written for this usage...)
settings['x_init'] = torch.Tensor([]) 

###################
# run simulations #
###################

data_all = []

# integration time steps, number of different initial states, length of simulations
dts = [300, 900, 1500] # we keep the time per simulation constant (simulations start from
Ts =  [300, 100, 60]   # initial perturbations that will essentially die down in amplitude),
Ns =  [200, 600, 1000] # and the total number of states (= dataset size).

for dt, T, N in zip(dts, Ts, Ns):

    print('simulating test data with (semi-)implicit numerical simulator.')
    print(f'-simulating {N} simulations (in parallel) for {T} time steps for integration step-size dt={dt}')
    data = swe2D_sim(N=N,
                     T=T, 
                     my=100,
                     nx=100,
                     dt=dt,     # temporal discretization [s]
                     dx=1e4,    # physical distance between grid points
                     g=9.81,    # graviation constant
                     w_imp=0.5, # weight between implicit/explicit integration step
                     cd=1e-3,   # coefficient of drag
                     ah=1000.,
                     data_scales=np.ones((1,1,5,1,1)),
                     settings=settings, 
                     verbose=True, 
                     comp_u='calculate', # whether to solve or calculate velocities u. 
                     init_vals=None) # initial values for simulation (will be drawn if init_vals=None)

    print('data.shape:', data.shape)
    
    try:
        os.stat('./data')
    except:
        os.mkdir('./data')
    
    fn = f'./data/dataset_dt{str(int(dt))}'
    print(f'storing data to {fn}.npz')
    np.savez(fn, 
             z_vals=data[:,:,0],
             u_vals=data[:,:,1],
             v_vals=data[:,:,2],
             depth_profiles=data[:,0,3])
    print('done.\n')
