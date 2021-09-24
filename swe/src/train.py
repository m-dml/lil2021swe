import os
import subprocess
import hydra
from omegaconf import OmegaConf, read_write

import numpy as np
import torch
from torchsummary import summary

import pytorch_lightning as pl

from implem.utils import init_torch_device, as_tensor, device, dtype, dtype_np
from implem.data import DataModule, MultiStepMultiTrialDataset
from implem.model import LightningImplicitModel, LightningImplicitPretrainModel

from .utils import fix_bM_BCs, enforce_bcs, network_init_swe


def load_data_swe(filedir, data_fn, instance_dimensionality='2D', normalize_channels=True):

    assert instance_dimensionality in ['1D', '2D']
    data_suff = ''

    data_fn_full = os.path.join(filedir, f'{data_fn}.npz')
    data = np.load(data_fn_full)
    print('(#trials, simulation length [steps], grid width, grid height) = ', data['z_vals'].shape)

    boundary_mask = np.ones_like(data['z_vals'])
    boundary_mask[:,:,0] = boundary_mask[:,:,-1] = 0.
    if instance_dimensionality == '2D':
        boundary_mask[:,:,:,0] = boundary_mask[:,:,:,-1] = 0.

    channels = [data['z_vals'], data['u_vals']]
    if instance_dimensionality == '2D':
        channels.append(data['v_vals'])
    channels.append(np.expand_dims(data['depth_profiles'],1).repeat(data['z_vals'].shape[1], axis=1))
    channels.append(boundary_mask)

    data = np.stack(channels, axis=2).astype(dtype_np) # N x T x C x H x W

    data_scales = 1.
    if normalize_channels:
        data_scales = np.sqrt((data**2).mean(axis=tuple([i for i in range(data.ndim) if i != 2])))
        if instance_dimensionality == '1D':
            data_scales = data_scales.reshape(1,1,-1,1)
        else:
            data_scales = data_scales.reshape(1,1,-1,1,1)
        data_scales[0,0,-1] = 1. # don't change binary mask channels
        data /= data_scales      # normalize per channel
    print('data tensor shape:', data.shape)

    if instance_dimensionality == '1D':
        channel_descr = ['z_vals', 'u_vals', 'depth_profiles', 'boundary_mask']
    else:
        channel_descr = ['z_vals', 'u_vals', 'v_vals', 'depth_profiles', 'boundary_mask']

    n_static_channels = 2 # depth_profiles, boundary_mask

    return data, channel_descr, n_static_channels, data_scales


def train(cfg):
    
    pl.seed_everything(1234)

    #############
    # load data #
    #############

    data, channel_descr, n_static_channels, data_scales = load_data_swe(
        filedir=cfg.data.root_data,
        data_fn=cfg.data.data_fn,
        instance_dimensionality=cfg.data.instance_dimensionality,
        normalize_channels=True
    )
    offset = np.arange(cfg.data.offset) + 1 if np.ndim(cfg.data.offset) == 0 else cfg.data.offset
    dm = hydra.utils.instantiate(
            cfg.data,
            data=data,
            Dataset=MultiStepMultiTrialDataset,
            offset=offset
    )
    dm.setup()

    for batch in dm.train_dataloader():
        x,y = batch
        print('data shapes check: (x.shape, y.shape)', x.shape, y.shape)
        break


    ###############
    # model setup #
    ###############
    
    print(f'setting up implicit emulator.') if cfg.model.layer.implicit_mode else print(f'setting up explicit emulator.')
    
    assert cfg.model.output_channels[-1] == data.shape[2] - n_static_channels
    model = hydra.utils.instantiate(
            cfg.model,
            instance_dimensionality = cfg.data.instance_dimensionality,
            input_channels = data.shape[2],
            static_channels = n_static_channels,
            offset = offset,
            system_determ=[fix_bM_BCs, None],
            format_output = enforce_bcs,
            _recursive_ = False # necessary for model.configure_optimizers()
    )
    network_init_swe(model, x, data_scales, dt=300., dx=1e4, w_imp=0.5, g=9.81, cd=1e-3, ah=1e3)
    print('model', model)

    ##################
    # model training #
    ##################

    if cfg.trainer.gpus < 1:
        callbacks = None
    else:
        callbacks = [
            pl.callbacks.EarlyStopping(monitor='loss/val', patience=50),
            #pl.callbacks.ModelCheckpoint(monitor="loss/val", 
            #                             save_top_k=3, mode='min', 
            #                             dirpath=".", save_last=True)
        ]
    

    logger_tb = pl.loggers.TensorBoardLogger(".", "", "", 
                                             log_graph=True, 
                                             default_hp_metric=False)
    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer),
                         logger=logger_tb, 
                         callbacks=callbacks)
    trainer.fit(model, dm)
