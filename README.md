# Learning Implicit PDE Integration with Linear Implicit Layers

Code for reproducing the experiments of our [NeurIPS 2021 workshop submission](https://openreview.net/forum?id=veNBQ15T6N0).
To run the code, first create an environment for our implem package for implicitly integrated dynamical models.
We use anaconda to set up and activate the environment via:
```
$ cd implicit_emulators/
$ conda env create --prefix .implem_env --file environment.yml
$ conda activate .implem_env/
$ python setup.py develop
```

To (re-)train the models used for our experiments, first generate the training data via
```
$ cd swe/
$ python gen_data.py
```
This should create three datasets (`swe/data/dataset_dt*.npz`). The data will require about ~20.9GB of free disk space.

To train the networks, use the commands (sequentially or in parallel on multiple nodes):
```
$ python main.py data.name=SWE2D_dt300 data.data_fn=dataset_dt300 model=model_implicit
$ python main.py data.name=SWE2D_dt900 data.data_fn=dataset_dt900 model=model_implicit
$ python main.py data.name=SWE2D_dt1500 data.data_fn=dataset_dt1500 model=model_implicit

$ python main.py data.name=SWE2D_dt300 data.data_fn=dataset_dt300 model=model_explicit
$ python main.py data.name=SWE2D_dt900 data.data_fn=dataset_dt900 model=model_explicit
$ python main.py data.name=SWE2D_dt1500 data.data_fn=dataset_dt1500 model=model_explicit
```

The explicit models should converge in about 10h (tested on an Nvidia V100), the semi-implicit models should converge between 24h (dt=300), 34h (dt=900) and 72h (dt=1500).
Training can be monitored with tensorbard: `tensorboard --logdir swe/outputs`.
If you want to cut waiting time, you should be able to interrupt training at about 1/2 of the above-stated times without noticable loss of performance.

We included the trained model parameters from our model fits in this repository (under `swe/outputs/SWE2D_dt*/.../min_net0/`).

The analysis of results is done with the notebook `swe/eval_exps.ipynb` and by default will load our trained networks.
To change to your trained networks, change the paths in the network-loading notebook cells accordingly. 
The notebook will reproduce the results of our submission and store figure panels under `swe/figs/`.

We also included a quick tutorial notebook that loads one of our trained semi-implicit models (dt=900) and evaluates an example simulation, the learned computational stencils, and the convergence of the Red-Black Gauss-Seidel solver on an example forward- and backward pass of backpropagation.
