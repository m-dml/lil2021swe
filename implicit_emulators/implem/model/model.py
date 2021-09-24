import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import logging
import hydra
from omegaconf import read_write

from implem.utils import init_torch_device, as_tensor, device, dtype, dtype_np
from implem.utils import banded2full_blockmat
from implem.utils import comp_dAdM, comp_dAdM_default, comp_dAdM_default_sep_eqs_per_field
from implem.utils import transpose_compact_blockmat, transpose_compact_blockmat_sep_eqs_per_field
from implem.utils import banded_gauss_seidel_redblack, stencil_offsets_to_indices
 
from implem.model.networks import BilinearConvLayer

from functools import partial


class LinearSolveGSRB(torch.autograd.Function):

    @staticmethod
    def forward(ctx, b, M, settings=None):

        # plug in key values that change between forward and backward pass
        settings['x_init'] = settings['x_forward_init']
        settings['max_iter'] = settings['max_iter_forward']
        settings['thresh'] = settings['thresh_forward']

        # solve A x = b
        solution, _ = banded_gauss_seidel_redblack(M, b, **settings)

        # clean up and store for backward pass
        settings.pop('x_init')
        settings.pop('max_iter')
        settings.pop('thresh')
        ctx.save_for_backward(M, solution)
        ctx.settings = settings

        return solution

    @staticmethod
    def backward(ctx, grad_output):

        M, solution = ctx.saved_tensors
        settings = ctx.settings

        # plug in key values that change between forward and backward pass
        settings['max_iter'] = settings['max_iter_backward']
        settings['thresh'] = settings['thresh_backward']
        if settings['x_backward_init'] is None:
            settings['x_init'] = 1.*grad_output.detach()
        else:
            settings['x_init'] = settings['x_backward_init']
        if 'pad_x_backward_init' in settings.keys():
            assert len(settings['pad_x_backward_init']) == 4
            settings['x_init'] = torch.nn.functional.pad(input = settings['x_init'],
                                                         pad = settings['pad_x_backward_init'])

        if settings['sep_eqs_per_field']:
            transpose_M = transpose_compact_blockmat_sep_eqs_per_field
            start_flatten_dim_M = 3 # M.shape = (N, L, K, *spatial_dims)
        else:
            transpose_M = transpose_compact_blockmat
            start_flatten_dim_M = 4 # M.shape = (N, L, L, K, *spatial_dims)
        MT = transpose_M(M.flatten(start_dim=start_flatten_dim_M),
                         offdiagonals=settings['offdiagonals']).reshape(M.shape)

        # masking out equations for backwards pass, e.g. because they will be zeroed out
        # by the next steps of the backwards pass anyways and just make the solve harder:
        if 'backwards_mask' in settings.keys() and not settings['backwards_mask'] is None:
            backwards_mask = settings['backwards_mask']
            assert np.all(grad_output.shape == backwards_mask.shape)
            grad_output = backwards_mask * grad_output + (1-backwards_mask) * 0.
            if settings['sep_eqs_per_field']:
                repeats = (1,*M.shape[1:3],1,1)
            else:
                repeats = (1,1,*M.shape[2:4],1,1)
                backwards_mask = backwards_mask.unsqueeze(2)
            backwards_mask = backwards_mask.unsqueeze(-3).repeat(repeats)
            # replace equations with identity
            MT = backwards_mask * MT + (1-backwards_mask) * 0. # replace equations with identity
            K2, i = M.shape[-3]//2, range(M.shape[1]) # central stencil element, blockdiagonal
            MT[:,i,i,K2,:,:] = backwards_mask[:,i,i,K2,:,:] * MT[:,i,i,K2,:,:] \
                               + (1-backwards_mask)[:,i,i,K2,:,:] * 1.

        # solve A.T x = grad_output
        z, _ = banded_gauss_seidel_redblack(MT, grad_output,
                                            **settings)

        # clean up
        settings.pop('x_init')
        settings.pop('max_iter')
        settings.pop('thresh')

        # prepare final results
        dAdM, mask = settings['dAdM'], settings['dAdM_mask']
        idxz, idxy = dAdM[0], dAdM[1]
        newshape = (z.shape[0], *mask.shape)

        # appending 'None' to account for '**settings' in forward:
        grad_input = (z,
                      - mask*(z.flatten(start_dim=1)[:,idxz] * solution.flatten(start_dim=1)[:,idxy]).reshape(newshape),
                      None)

        return grad_input

def default_system_determ(settings, x, b, M):

    return b, M, None


class ImplicitLayer(pl.LightningModule):

    def __init__(self,
                 network,
                 input_channels,
                 output_channels,
                 instance_dimensionality='2D',
                 static_channels=0,
                 linear_solver=LinearSolveGSRB,
                 stencil_shape='full',
                 K=1,
                 num_fields=1,
                 sep_eqs_per_field=False,
                 implicit_mode=True,
                 explicit_channels=0,
                 bilin_out_conv=False,
                 out_conv_kernel_size=1,
                 system_determ=None,
                 init_from_current_x=False,
                 **kwargs):

        super().__init__()

        assert instance_dimensionality in ['1D', '2D']
        if instance_dimensionality == '2D' and stencil_shape=='full':
            assert int(np.sqrt(K))**2 == K    # only supporting square filters
            assert np.mod(int(np.sqrt(K)),2) == 1 # only supporting odd stencil sizes
        elif instance_dimensionality == '2D' and stencil_shape=='cross':
            assert np.mod(K-1,4) == 0         # only supporting fully symmetric cross-shaped filters
            assert np.mod((K+1)//2,2) == 1    # only supporting odd stencil sizes
        elif instance_dimensionality == '1D' and stencil_shape=='cross':
            print("Warning: for 1D models, stencil_shape='cross' makes no difference over 'full'")
        implicit_mode = False if num_fields < 1 else implicit_mode
        linear_solver = LinearSolveGSRB if linear_solver=='LinearSolveGSRB' else linear_solver
        self.system_determ = default_system_determ if system_determ is None else system_determ

        self.save_hyperparameters()  # saves everything passed to this class as a hyperperameter.
        assert self.hparams.stencil_shape in ['full', 'cross']

        self.num_channels_lin_sys = num_fields * (K+1) if sep_eqs_per_field else num_fields**2 * K + num_fields
        self.forward_model = self.init_forward_block(
            input_channels=input_channels,
            output_channels=self.num_channels_lin_sys + explicit_channels,
            instance_dimensionality=instance_dimensionality,
            cfg_network=self.hparams.network
        )
        Conv = torch.nn.Conv2d if instance_dimensionality=='2D' else torch.nn.Conv1d
        if bilin_out_conv:
            self.out_conv = BilinearConvLayer(
                     input_channels = input_channels + num_fields + explicit_channels,
                     output_channels = output_channels,
                     bilin_channels=output_channels,
                     Conv=Conv,
                     kernel_size=out_conv_kernel_size)
        else:
            self.out_conv = Conv(in_channels = input_channels + num_fields + explicit_channels,
                                 out_channels = output_channels,
                                 kernel_size = out_conv_kernel_size,
                                 padding=(out_conv_kernel_size-1)//2)
        self._set_settings()

    @staticmethod
    def init_forward_block(input_channels, 
                           output_channels, 
                           instance_dimensionality, 
                           cfg_network):

        logging.debug(f"Setting up model for {input_channels} input channels.")

        model = hydra.utils.instantiate(cfg_network,
                                        input_channels=input_channels,
                                        output_channels=output_channels,
                                        instance_dimensionality=instance_dimensionality
                                       )

        model.train() # unfreezes the model.

        return model


    def _forward(self, x):
        # Constructs system of linear equations Ax=b from input x, i.e. computes 
        # A(x) and b(x). Sparse & banded matrix A is given in compact form through tensor M.

        K, L = self.hparams.K, self.hparams.num_fields

        out = self.forward_model(x)

        if self.hparams.instance_dimensionality == '1D':
            out = out.unsqueeze(-2)

        idx_end_M = self.num_channels_lin_sys - L
        idx_end_b = self.num_channels_lin_sys
        
        M = out[:,:idx_end_M]          # batch-size x elements_of_M * K x n
        b = out[:,idx_end_M:idx_end_b] # batch-size x num_fields x n
        out_expl = out[:,idx_end_b:]   # batch-size x explicit_channels x n

        if not self.hparams.implicit_mode: # directly return b(x_t) as prediction for x_t+1
            return None, b, out_expl

        # setup for compact matrix-vector product Ax = f(M,x) = b
        self._set_mask(M)

        # poor-mans diagonal dominiance (also ensure diag(A)=1)
        r = 1.05  # margin for diagonal dominiance: diagonal at least r times larger than abs sum of off-diagonal
        if self.hparams.sep_eqs_per_field:
            Mabs = torch.abs(M.reshape(-1, L, K, *M.shape[-2:])) # seperates bands for L fields
            s = (Mabs * self.mask.unsqueeze(1)).sum(axis=2) # masking out out-of-domain stencil elements
            b = b / (r * s)
            M = M / s.repeat_interleave(K, dim=1)
            M = M.reshape(M.shape[0], L, K, *M.shape[2:])
            M[:,:,K//2] = 1. # main diagonal of A is unity
        else:
            Mabs = torch.abs(M.reshape(-1, L, K*L, *M.shape[-2:])) # seperates bands for L fields
            s = (Mabs * self.mask.unsqueeze(1)).sum(axis=2) # masking out out-of-domain stencil elements
            b = b / (r * s)
            M = M / s.repeat_interleave(K*L, dim=1)
            M = M.reshape(M.shape[0], L, L, K, *M.shape[2:])
            M[:,range(L),range(L),K//2] = 1. # main diagonal of A is unity
            
        # setup for solver layer:
        self._set_tables(ny=M.shape[-2], nx=M.shape[-1])

        b, M, self.settings['backwards_mask'] = self.system_determ(self.settings, x, b, M)

        return M, b, out_expl

    
    def _set_mask(self, M):
        # defines mask for correct computation of diagonal dominance: 
        # along the boundaries, we do not add up all stencil elements when ensuring 
        # diagonal dominance, those outside the domain get masked out!

        # upon first call to forward:
        try:

            self.mask

        except:

            # create mask indicating which elements of M will actually be read out
            L, ny, nx = self.hparams.num_fields, M.shape[-2], M.shape[-1]
            kernel_size = self.settings['kernel_size']
            pad_sizes = (int(kernel_size[0]//2), int(kernel_size[1]//2))
            padding = (pad_sizes[0],pad_sizes[0],pad_sizes[1],pad_sizes[1])
            padding_rev = (pad_sizes[1],pad_sizes[1],pad_sizes[0],pad_sizes[0])
            mask = torch.nn.functional.pad(
                torch.ones(*M.shape[-2:],dtype=M.dtype,device=M.device),
                pad=padding_rev) # torch pad inverts order or axes !
            idxby, idxbx = stencil_offsets_to_indices(
                self.settings['stencil_offsets'],
                ny=M.shape[-2], nx=M.shape[-1], padding=padding)
            self.mask = mask[idxby,idxbx].unsqueeze(0)
            if not self.settings['sep_eqs_per_field']:
                self.mask = self.mask.repeat(1,L,1,1)


    def _set_settings(self):

        try:

            self.settings

        except:

            K, L = self.hparams.K, self.hparams.num_fields
            
            if self.hparams.instance_dimensionality == '1D':

                kernel_size = (1,K)
                assert np.all(np.mod(kernel_size,2) == 1) # only odd kernel_sizes atm (3x3, 5x5 etc.)
                kernel_shape = (1,kernel_size)

                iys = torch.zeros(K, dtype=torch.int, device=device)
                ixs = torch.arange(-(K//2), K//2+1, dtype=torch.int, device=device)
                stencil_offsets = torch.stack((iys, ixs))
                colors = [(0,i) for i in range((kernel_size[1]+1)//2)]
                idx_col = list((i, *locs) for i in range(L) for locs in (colors))
                idx_col = torch.tensor(idx_col, dtype=torch.int, device=device)
                grid_type = 'grid'

            elif self.hparams.instance_dimensionality == '2D':

                if self.hparams.stencil_shape == 'full':

                    kernel_size = (int(np.sqrt(K)), int(np.sqrt(K)))
                    assert np.prod(kernel_size) == K
                
                    # e.g. offsets for full 3x3 stencil (K=9 stencil elements)
                    # iys = [-1,-1,-1, 0, 0, 0, 1, 1, 1]
                    # ixs = [-1, 0, 1,-1, 0, 1,-1, 0, 1]
                    iys = torch.arange(-(kernel_size[0]//2), kernel_size[0]//2+1, 
                                       dtype=torch.int, device=device)
                    iys = torch.repeat_interleave(input=iys, repeats=kernel_size[1])
                    ixs = torch.arange(-(kernel_size[1]//2), kernel_size[1]//2+1,
                                       dtype=torch.int, device=device).repeat(kernel_size[0])
                    # e.g. colors for full 3x3 stencil: [(0,0), (0,1), (1,0), (1,1)]
                    colors = [(i,j) for i in range((kernel_size[0]+1)//2) for j in range((kernel_size[1]+1)//2)]
                    grid_type = 'grid'

                elif self.hparams.stencil_shape == 'cross':

                    kernel_size = ((K+1)//2, (K+1)//2)
                    assert kernel_size[0] + kernel_size[1] == K + 1

                    # e.g. offsets for cross-shaped 3x3 stencil (K=5 stencil elements)
                    # iys = [-1, 0, 0, 0, 1]
                    # ixs = [ 0,-1, 0, 1, 0]
                    iys = torch.cat((torch.arange(-(kernel_size[0]//2),0), # top of cross
                                     torch.zeros(2*(kernel_size[1]//2)),   # left, center and right
                                     torch.arange((kernel_size[0]//2)+1))  # bottom of cross
                                   ).int().to(device)
                    ixs = torch.cat((torch.zeros(kernel_size[0]//2),       # top of cross
                                     torch.arange(-(kernel_size[1]//2),0), # left
                                     torch.arange((kernel_size[1]//2)+1),  # center and right
                                     torch.zeros(kernel_size[0]//2))       # bottom of cross
                                   ).int().to(device)

                    # e.g. colors for cross-shaped 3x3 stencil: [(0,0), (0,1)]
                    colors = [(0,i) for i in range((np.max(kernel_size)+1)//2)]
                    grid_type = 'checker'

                    
            idx_col = list((i, *locs) for i in range(L) for locs in (colors))
            idx_col = torch.tensor(idx_col, dtype=torch.int, device=device)
            stencil_offsets = torch.stack((iys, ixs))

            self.settings = {'sep_eqs_per_field' : self.hparams.sep_eqs_per_field,
                             'grid_type' : grid_type,
                             'kernel_size' : kernel_size,
                             'idx_col' : idx_col,
                             'stencil_offsets' : stencil_offsets,
                             'x_forward_init' : None,
                             'x_backward_init' : None,
                             'backwards_mask' : None,
                             'thresh_backward' : 1e-13,
                             'max_iter_backward' : 1000,
                             'thresh_forward' : 1e-13,
                             'max_iter_forward' : 1000}

    def _set_tables(self, ny, nx):

        # upon first call to forward:
        try:

            assert self.settings['dAdM'].shape[-1]==nx
            if self.hparams.instance_dimensionality == '2D':
                assert self.settings['dAdM'].shape[-2]==ny
            self.settings['offdiagonals']

        except:

            L, K = self.hparams.num_fields, self.hparams.K

            if self.hparams.instance_dimensionality == '1D':
                
                assert ny==1
                offdiagonals = range(-(K//2), K//2 + 1)
                constr_A_fun = partial(banded2full_blockmat,
                                       offdiagonals=offdiagonals,
                                       kernel_size=self.settings['kernel_size'])
                dAdM, dAdM_mask = comp_dAdM(L=L, K=K, nx=nx,
                                            constr_A_fun = constr_A_fun)

            elif self.hparams.instance_dimensionality == '2D':

                kernel_size = self.settings['kernel_size']
                y_range = np.arange(-(kernel_size[0]//2),(kernel_size[0]//2)+1)
                x_range = np.arange(-(kernel_size[1]//2),(kernel_size[1]//2)+1)
                if self.hparams.stencil_shape == 'full':
                    offdiagonals = (nx*y_range.reshape(-1,1) + x_range.reshape(1,-1)).flatten()
                elif self.hparams.stencil_shape == 'cross':
                    offdiagonals = np.unique(np.concatenate([nx*y_range, x_range]))

                if self.settings['sep_eqs_per_field']:
                    comp_dAdM = comp_dAdM_default_sep_eqs_per_field
                    dAdM_mask = self.mask.repeat(L, 1, 1, 1)
                else:
                    comp_dAdM = comp_dAdM_default
                    dAdM_mask = self.mask[:,:K].unsqueeze(0).repeat(L, L, 1, 1, 1)
                dAdM = comp_dAdM(mask = dAdM_mask, 
                                 stencil_offsets = self.settings['stencil_offsets'],
                                 kernel_size = kernel_size)

            self.settings['offdiagonals'] = offdiagonals
            self.settings['dAdM'], self.settings['dAdM_mask'] = dAdM, dAdM_mask


    def _set_inits(self, b, x, x_forward_init = None, x_backward_init = None, pad_input = True):

        if not x_forward_init is None:
            init = x_forward_init
        elif self.hparams.init_from_current_x:
            init = x[:,:self.hparams.num_fields]
        else:
            init = b

        if pad_input:
            stencil_offsets = self.settings['stencil_offsets'].detach().cpu().numpy()
            pad = (-np.min(np.minimum(stencil_offsets[1],0)), np.max(np.maximum(stencil_offsets[1],0)),
                   -np.min(np.minimum(stencil_offsets[0],0)), np.max(np.maximum(stencil_offsets[0],0)))
            self.settings['pad_x_backward_init'] = pad
        else:
            self.settings['pad_x_backward_init'] = None
        self.settings['x_forward_init'] = init.detach()
        self.settings['x_backward_init'] = x_backward_init


    def forward(self,
                x,
                x_forward_init = None,
                x_backward_init = None,
                *args, **kwargs):

        # immediate output of the feed-forward part of the model:
        M, b, out_expl  = self._forward(x)

        if self.hparams.implicit_mode:
            # set initialization for iterative solver (faster convergence):
            self._set_inits(b,x, x_forward_init, x_backward_init)

            # output via solver layer:
            out = self.hparams.linear_solver.apply(b, M, self.settings)

        else:
            # explicit mode: directly return b(x_t) as solution
            out = b

        # add direct explicit output channels (non-implicit!), if any
        out = self.out_conv(torch.cat((out, out_expl, x), dim=1))

        return out.squeeze(-2) if self.hparams.instance_dimensionality == '1D' else out


class LightningImplicitModel(pl.LightningModule):

    def __init__(self,
                 layer,
                 optim,
                 instance_dimensionality='2D',
                 input_channels=1,
                 static_channels=0,
                 offset=1,
                 stencil_shape=['full'],
                 stencil_size=[1],
                 num_fields=[1],
                 output_channels=[16],
                 explicit_channels=None,
                 sep_eqs_per_field=False,
                 loss_mask=None,
                 format_output=None,
                 system_determ=None,
                 add_out_block=False,
                 **kwargs):

        super().__init__()

        assert instance_dimensionality in ['1D', '2D']
        if type(sep_eqs_per_field) == bool:
            sep_eqs_per_field = [sep_eqs_per_field for i in range(len(num_fields))]
        if explicit_channels is None or explicit_channels == 'None':
            explicit_channels = [layer.network.hidden_channels[-1] for i in range(len(num_fields))]
        system_determ = [None for i in range(len(num_fields))] if system_determ is None else system_determ
        assert len(stencil_size) == len(stencil_shape) == len(num_fields) == len(explicit_channels)
        assert len(stencil_size) == len(output_channels) == len(system_determ)

        #self.save_hyperparameters()  # saves everything passed to this class as a hyperperameter.
        self.static_channels = static_channels
        self.optim = optim
        self.format_output = format_output

        self.impl_layers, layer_in, extra_statics = [], input_channels, 0
        for i in range(len(stencil_size)):
            assert stencil_shape[i] in ['full', 'cross']
            K = stencil_size[i]
            if instance_dimensionality=='2D':
                K = K**2 if stencil_shape[i] == 'full' else (K-1)*2+1
            self.impl_layers.append(
                hydra.utils.instantiate(
                        layer,
                        instance_dimensionality=instance_dimensionality,
                        input_channels=layer_in,
                        output_channels=output_channels[i],
                        static_channels=static_channels + extra_statics,
                        stencil_shape=stencil_shape[i],
                        K=K,
                        num_fields=num_fields[i],
                        sep_eqs_per_field=sep_eqs_per_field[i],
                        explicit_channels=explicit_channels[i],
                        system_determ=system_determ[i],
                        _recursive_ = False # necessary for model.configure_optimizers()
                )
            )
            layer_in, extra_statics = output_channels[i], num_fields[i]
        if add_out_block:
            self.impl_layers[-1].forward_model = torch.nn.Conv2d( 
                self.impl_layers[-2].hparams.output_channels,
                self.impl_layers[-1].hparams.output_channels-1,
                kernel_size=3,
                padding=1)
        self.impl_layers = torch.nn.ModuleList(self.impl_layers)
        self.loss_mask = loss_mask

    def forward(self, x):

        z = x
        for layer in self.impl_layers:
            x = layer(x)
            #z = torch.cat((z, x), dim=1) # torch.cat((z, statics), dim=1)

        if not self.format_output is None:
            x = self.format_output(x,z)

        return x


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optim, params=self.parameters())
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, verbose=True, patience=3, min_lr=1e-8)

        scheduler = [{
            'scheduler': scheduler,
            'monitor': "loss/val",
            'interval': 'epoch',
            'strict': True,
            },
        ]

        return [optimizer], scheduler


    def training_step(self, batch, batch_idx, *args, **kwargs):

        x,y = batch
        pred = self(x)

        y = y[:,0] # only predicting first (=only) sequence element
        if self.static_channels > 0: # do not predict last channels (= depth-profile and boundary mask)
            y = y[:,:-self.static_channels] 

        loss_function = torch.nn.MSELoss()
        if not self.loss_mask is None:
            pred = pred*self.loss_mask
            y = y*self.loss_mask
        loss = loss_function(input=pred, target=y)
        self.log("loss/train", loss)

        loss_scale = 1.
        return loss_scale * loss


    def validation_step(self, batch, batch_idx, *args, **kwargs):

        x,y = batch
        pred = self(x)

        y = y[:,0] # only predicting first (=only) sequence element
        if self.static_channels > 0: # do not predict last channels (= depth-profile and boundary mask)
            y = y[:,:-self.static_channels] 

        loss_function = torch.nn.MSELoss()
        if not self.loss_mask is None:
            pred = pred*self.loss_mask
            y = y*self.loss_mask
        loss = loss_function(input=pred, target=y)
        self.log("loss/val", loss)
        
        loss_scale = 1.
        return loss_scale * loss


class LightningImplicitPretrainModel(LightningImplicitModel):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.M_targets = None


    def _set_M_targets(self, N, domain_shape):

        if self.M_targets is None or self.M_targets[0].shape[0] != N:
            self.M_targets = []
            for layer in self.impl_layers:
                L, K = layer.hparams.num_fields, layer.hparams.K
                M_shape = (N, L, K, *domain_shape) if layer.hparams.sep_eqs_per_field else (N, L, L, K, *domain_shape)
                M_target = torch.zeros(M_shape,
                                       dtype=dtype,
                                       device=device,
                                       requires_grad=False)
                if layer.hparams.sep_eqs_per_field:
                    M_target[:,:,K//2] = 1.
                else:
                    M_target[:,range(L),range(L),K//2] = 1.
                self.M_targets.append(M_target)


    def forward(self, x):

        z = x
        Ms = []
        self._set_M_targets(N=x.shape[0], domain_shape=x.shape[-2:])
        for layer in self.impl_layers:
            M, b, out_expl = layer._forward(x) # involves no solve operation !
            x = layer.out_conv(torch.cat((b, out_expl, x), dim=1))
            Ms.append(M)

        if not self.format_output is None:
            x = self.format_output(x,z)

        return x, Ms


    def training_step(self, batch, batch_idx, *args, **kwargs):

        x,y = batch
        pred, Ms = self(x)
        self._set_M_targets(N=x.shape[0], domain_shape=x.shape[-2:])

        y = y[:,0] # only predicting first (=only) sequence element
        if self.static_channels > 0: # do not predict last channels (= depth-profile and boundary mask)
            y = y[:,:-self.static_channels] 

        loss_function = torch.nn.MSELoss()
        if not self.loss_mask is None:
            pred = pred*self.loss_mask
            y = y*self.loss_mask
        loss = loss_function(input=pred, target=y)
        for M, M_target in zip(Ms, self.M_targets):
            if not M is None: # M returned None if layer in explicit mode
                loss = loss + loss_function(input=M, target=M_target)
        self.log("loss/train", loss)

        loss_scale = 1.
        return loss_scale * loss


    def validation_step(self, batch, batch_idx, *args, **kwargs):

        x,y = batch
        pred, Ms = self(x)
        self._set_M_targets(N=x.shape[0], domain_shape=x.shape[-2:])

        y = y[:,0] # only predicting first (=only) sequence element
        if self.static_channels > 0: # do not predict last channels (= depth-profile and boundary mask)
            y = y[:,:-self.static_channels] 

        loss_function = torch.nn.MSELoss()
        if not self.loss_mask is None:
            pred = pred*self.loss_mask
            y = y*self.loss_mask
        loss = loss_function(input=pred, target=y)
        for M, M_target in zip(Ms, self.M_targets):
            if not M is None: # M returned None if layer in explicit mode
                loss = loss + loss_function(input=M, target=M_target)
        self.log("loss/val", loss)
        
        loss_scale = 1.
        return loss_scale * loss
