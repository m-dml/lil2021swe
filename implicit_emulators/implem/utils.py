import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Set, Dict, Tuple, Optional

def init_torch_device():
    if torch.cuda.is_available():
        print('using CUDA !')
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("CUDA not available")
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    return device

device, dtype, dtype_np = init_torch_device(), torch.float32, np.float32


def as_tensor(x):
    return torch.as_tensor(x, dtype=dtype, device=device)


################################################
# ML-parametrizing systems of linear equations #
################################################

def banded2full_mat(M, offdiagonals=None):
    # constructs banded matrix A from tensor M containing the (off-)diagonal bands of A in its rows.
    # M.shape[-2:]=[K,N], where K is number of non-zero bands in A.

    # Assuming symmetrically located bands:
    # M[K//2,:] the main diagonal of A, M[0,:] gives the lowest band of A and M[-1,:] the uppermost band.
    # Bands are stored in M[k,0:n-k+K//2] for k >= K//2 and  M[k,K//2-k:n] for k < K//2. 
    
    assert len(M.shape) >= 2
    
    K,n = M.shape[1:] # number of bands, dimensionality of matrix
    assert np.mod(K,2) == 1
    
    # assert M is batched for the remainder
    M = M.reshape(1,*M.shape) if len(M.shape) == 2 else M
    N = M.shape[0]

    # assuming symmetrical offsets for off-diagonal bands
    offdiagonals = np.arange(-(K//2), K//2+1) if offdiagonals is None else np.asarray(offdiagonals)
    assert np.all(offdiagonals[K//2+1:]>0) and np.all(offdiagonals[:K//2]<0) and offdiagonals[K//2]==0 
    
    # main diagonal
    A = torch.diag_embed(M[:,K//2,:]) # main diagonal

    # offdiagonals
    idx = np.arange(n)
    for j in np.arange(K): # iterating over k < K//2 and k > K//2
        o = offdiagonals[j]
        o_ = np.abs(o)
        idx_ = idx[:-o_]
        if j < K//2:
            A[:,idx_+o_, idx_] = M[:, j, idx_+o_]  # lower-triangular bands
        else:
            A[:,idx_, idx_+o_] = M[:, j, idx_]       # upper-triangular bands
        
    return A if N > 1 else A[0]


def full2banded_mat(A, Koff):
    # constructs a compact tensor from banded matrix A with K = Koff+1 unique bands (Koff off-diagonal bands
    # plus the diagonal), i.e. M.shape[-2:]=[K,N], where K is number of non-zero bands in A.

    # Assuming symmetrically located bands:
    # M[K//2,:] the main diagonal of A, M[0,:] gives the lowest band of A and M[-1,:] the uppermost band.
    # Bands are stored in M[k,0:n-k+K//2] for k >= K//2 and  M[k,K//2-k:n] for k < K//2. 
    
    assert len(A.shape) == 3

    N, K, n = A.shape[0], Koff + 1, A.shape[-1]
    
    assert np.mod(K,2) == 1 # assuming symmetrically located bands !

    M = torch.zeros((N, K, n))
    for i in range(K):
        tmp = torch.cat((torch.zeros((N,max(K//2-i,0))), 
                         torch.diagonal(A,i-K//2,dim1=-2,dim2=-1)),
                         dim=1)
        tmp = torch.cat((tmp, torch.zeros((N,max(i-K//2,0)))), dim=1)
        M[:,i,:] = tmp

    return M


def banded2full_blockmat(M, comp_mask=True, kernel_size=None, **kwargs):
    # constructs a block-matrix A with L x L blocks consisting of banded n x n matrices.
    # K bands per block are stored in input tensor M of size batch_size x L x L x K x n.

    L,K = M.shape[2:4]
    assert M.shape[1:4] == (L,L,K)

    if comp_mask:
        kernel_size = (int(np.sqrt(K)), int(np.sqrt(K))) if kernel_size is None else kernel_size
        assert np.prod(kernel_size) == K

        if kernel_size[0] == 1 and kernel_size[1] > 1:
            ny, nx = 1, M.shape[-1]
        else:
            ny = nx = int(np.sqrt(M.shape[-1]))
        mask = torch.ones((1,1,ny,nx), requires_grad=False, device=device)
        mask = hankel_unfold(mask, kernel_size=kernel_size)
        mask = mask.reshape(1,1,1,K,ny,nx).repeat(M.shape[0],L,L,1,1,1)
        mask = banded2full_blockmat(mask.flatten(start_dim=4), comp_mask=False, **kwargs)
    
    Aij = []
    for i in range(L):
        for j in range(L):
            Aij.append(banded2full_mat(M[:,i,j], **kwargs))
    # block-matrix in torch: first cat Aij to form rows of A, then cat rows of A to form A: 
    A = torch.cat(
        [torch.cat([Aij[i*L+j] for j in range(L)],dim=-1) for i in range(L)], 
        dim = -2
    )
    return A*mask if comp_mask else A # batch_size x (L * n) x (L * n)


##########################################################
# Compact-form matrix-vector product for banded matrices #
##########################################################


def transpose_compact(M, offdiagonals=None):
    # 'tranposes' a compact representation M of a banded matrix A, in the sense that the 
    # matrix compactly represented by transpose_compact(M) is A.transpose(-2,-1). 

    K = M.shape[1]
    assert np.mod(K,2) == 1 # Assuming symmetrically located bands at off-diagonals -K//2, .. 0, .., K//2

    # assuming symmetrical offsets for off-diagonal bands
    offdiagonals = np.arange(-(K//2), K//2+1) if offdiagonals is None else np.asarray(offdiagonals)

    MT = torch.flip(M, [1]) # first flip upper-triangular with lower-triangular bands
    for k in range(K):       # then for each band of A ...
        MT[:,k] = torch.roll(MT[:,k],shifts=-offdiagonals[k],dims=-1) # fix readout offset within (off-)diagonal band  

    return MT


def transpose_compact_blockmat(M, offdiagonals=None):
    # 'tranposes' a compact representation M of a banded matrix A, in the sense that the 
    # matrix compactly represented by transpose_compact(M) is A.transpose(-2,-1). 

    L, K = M.shape[2], M.shape[3]
    assert M.shape[1:4] == (L,L,K)
    assert np.mod(K,2) == 1 # Assuming symmetrically located bands at off-diagonals -K//2, .. 0, .., K//2

    # assuming symmetrical offsets for off-diagonal bands
    offdiagonals = np.arange(-(K//2), K//2+1) if offdiagonals is None else np.asarray(offdiagonals)
    assert len(offdiagonals) == K

    MT = M.transpose(1,2)     # first tranpose block-structure of A (different variable groups)
    assert np.all(offdiagonals[::-1] == - offdiagonals)
    MT = torch.flip(MT, [3])  # then flip upper-triangular with lower-triangular bands
    for k in range(K):        # then for each band of A ...
        MT[:,:,:,k] = torch.roll(MT[:,:,:,k],-offdiagonals[k],dims=-1) # fix readout offset within (off-)diagonal band  
    return MT


def transpose_compact_blockmat_sep_eqs_per_field(M, offdiagonals=None):
    # 'tranposes' a compact representation M of a banded matrix A, in the sense that the 
    # matrix compactly represented by transpose_compact(M) is A.transpose(-2,-1). 

    L, K = M.shape[1:3]
    assert np.mod(K,2) == 1 # Assuming symmetrically located bands at off-diagonals -K//2, .. 0, .., K//2

    # assuming symmetrical offsets for off-diagonal bands
    offdiagonals = np.arange(-(K//2), K//2+1) if offdiagonals is None else np.asarray(offdiagonals)
    assert len(offdiagonals) == K

    assert np.all(offdiagonals[::-1] == - offdiagonals)
    MT = torch.flip(M, [2])   # flip upper-triangular with lower-triangular bands
    for k in range(K):        # then for each band of A ...
        MT[:,:,k] = torch.roll(MT[:,:,k],-offdiagonals[k],dims=-1) # fix readout offset within (off-)diagonal band  
    return MT


def hankel_unfold(x, kernel_size):
    # creates Hankel matrix from shift-padded versions of input x for shifts corresponding to 2D convolution

    out_shape = (*x.shape[:2], np.prod(kernel_size), *x.shape[2:])
    assert len(x.shape) == 4     # unfold only implemented for 4D tensors
    assert len(kernel_size) == 2 # 
    
    out = torch.nn.functional.unfold(x, kernel_size, padding=[s//2 for s in kernel_size])

    return out.reshape(out_shape) # restoring channels


def bandeddot_blockmat(M : torch.Tensor,
                       x : torch.Tensor,
                       stencil_idx : Tuple[torch.Tensor, torch.Tensor]):
    # implements a matrix-vector product Ax where banded n x n matrix A is given as a tensor M 
    # with shapes M.shape = [N,L,L,K,ny,nx], x.shape = [N, L, ny+py, nx+px], where N is the batch-size, K the
    # number of stencil elements, L the number of fields, (ny,nx) the grid size and (px,py) the input padding.
    # stencil_idx=(iis, ijs) with stencil_idx.shape=(2,K) gives the spatial offsets of each stencil element.
    # E.g. for a full 3x3 stencil, iis = [-1,-1,-1, 0, 0, 0, 1, 1, 1]
    #                              ijs = [-1, 0, 1,-1, 0, 1,-1, 0, 1]
    #
    # Assumes block-matrix A has L x L blocks consisting of banded n x n matrices.
    # K bands per block are stored in input tensor M of size batch_size x L x L x K x n.
    #
    # Assuming symmetrically located bands:
    # M[:,:,:,K//2] the main diagonal of A, M[:,:,:,0] gives the lowest band of A and M[:,:,:,-1] the uppermost.
    # Bands are stored in M[:,k,0:N-k+K//2] for k >= K//2 and  M[:,k,K//2-k:N] for k < K//2.
    # The ordering of bands in k and the offset indices iis, ijs need to match !
    #
    # If sep_eqs_per_field==True, assumes that M.shape = [N,L,K,ny,nx] and M[:,i] gives the diagonal blocks 
    # A[i,i] for all variable groups i.

    idxby, idxbx = stencil_idx[0], stencil_idx[1]
    assert x.shape[1] == M.shape[1] # number of variable groups L
    assert M.shape[2] == M.shape[1] # L x L blocks within A

    return (M * x[:,:,idxby,idxbx].unsqueeze(1)).sum(dim=(2,3))


def bandeddot_blockmat_sep_eqs_per_field(M : torch.Tensor,
                                         x : torch.Tensor,
                                         stencil_idx : Tuple[torch.Tensor, torch.Tensor]):
    # implements a matrix-vector product Ax where banded n x n matrix A is given as a tensor M 
    # with shapes M.shape = [N,L,K,ny,nx], x.shape = [N, L, ny+py, nx+px], where N is the batch-size, K the
    # number of stencil elements, L the number of fields, (ny,nx) the grid size and (px,py) the input padding.
    # stencil_idx=(iis, ijs) with stencil_idx.shape=(2,K) gives the spatial offsets of each stencil element.
    # E.g. for a full 3x3 stencil, iis = [-1,-1,-1, 0, 0, 0, 1, 1, 1]
    #                              ijs = [-1, 0, 1,-1, 0, 1,-1, 0, 1]
    #
    # Assumes block-matrix A has L x L blocks consisting of banded n x n matrices.
    # M gives the diagonal blocks of A !
    # K bands per block are stored in input tensor M of size batch_size x L x L x K x n.
    #
    # Assuming symmetrically located bands:
    # M[:,:,:,K//2] the main diagonal of A, M[:,:,:,0] gives the lowest band of A and M[:,:,:,-1] the uppermost.
    # Bands are stored in M[:,k,0:N-k+K//2] for k >= K//2 and  M[:,k,K//2-k:N] for k < K//2.
    # The ordering of bands in k and the offset indices iis, ijs need to match !

    idxby, idxbx = stencil_idx[0], stencil_idx[1]
    assert x.shape[1] == M.shape[1] # number of variable groups L

    return (M * x[:,:,idxby,idxbx]).sum(dim=2)


def comp_dAdM(L, K, nx, constr_A_fun):
    # Brute-force functon to trace indicies from a tensor M in a matrix A = constr_A_fun(M).
    # In the case where A = constr_A_fun(M) is constructed by simply pasting elements M[ij]
    # into components A[kl], the result of this function coincides with the derivative dA/dM.

    ntot = int(L*L*K*nx)
    Mi = torch.arange(ntot).reshape(1,L,L,K,nx) + 1
    Ai = constr_A_fun(Mi).detach().cpu().numpy()
    
    dAdM = np.nan * np.zeros((2,ntot))
    for kl in range(ntot):
        i, j = np.where(Ai==kl+1)
        assert len(i) <= 1
        if len(i) == 1:
            dAdM[:,kl] = (i,j)
    dAdM = dAdM.reshape(2,L,L,K,nx)

    mask = as_tensor(1. * np.invert(np.isnan(dAdM)).prod(axis=0).reshape(dAdM.shape[1:]))
    dAdM = np.nan_to_num(dAdM).astype(np.int)

    return dAdM, mask


def comp_dAdM_default(mask : torch.Tensor,
                      stencil_offsets : torch.Tensor,
                      kernel_size):

    L, K, ny, nx = mask.shape[1:]
    assert mask.shape[0] == L

    dAdM = np.zeros((2,*mask.shape),dtype=np.int)
    idx = np.arange(nx*ny).reshape(ny,nx)
    for i in range(L): # for each field
        for j in range(L): # for each field
            for k in range(K): # for each stencil element

                dAdM[0][i][j][k] = np.arange(nx*ny).reshape(ny,nx) + i * (nx*ny)
                dAdM[0][i][j][k] = (mask[i][j][k].detach().cpu().numpy() * dAdM[0][i][j][k]).astype(np.int)

                oi, oj = stencil_offsets[:,k]
                ii = range(0, ny+oi) if oi < 0 else range(oi,ny)
                jj = range(0, nx+oj) if oj < 0 else range(oj,nx)
                I,J = np.where(mask[i][j][k].cpu().numpy())
                dAdM[1][i][j][k][I,J] = idx[ii][:,jj].flatten() + j * nx*ny

    return dAdM


def comp_dAdM_default_sep_eqs_per_field(mask : torch.Tensor,
                                        stencil_offsets : torch.Tensor,
                                        kernel_size):

    L, K, ny, nx = mask.shape
    dAdM = np.zeros((2,*mask.shape),dtype=np.int)
    idx = np.arange(nx*ny).reshape(ny,nx)
    for i in range(L): # for each field
        for k in range(K): # for each stencil element

            dAdM[0][i][k] = np.arange(nx*ny).reshape(ny,nx) + i * (nx*ny)
            dAdM[0][i][k] = (mask[i][k].detach().cpu().numpy() * dAdM[0][i][k]).astype(np.int)

            oi, oj = stencil_offsets[:,k]
            ii = range(0, ny+oi) if oi < 0 else range(oi,ny)
            jj = range(0, nx+oj) if oj < 0 else range(oj,nx)
            I,J = np.where(mask[i][k].cpu().numpy())
            dAdM[1][i][k][I,J] = idx[ii][:,jj].flatten() + i * nx*ny

    return dAdM


#####################################################
# Numerical solvers for systems of linear equations #
#####################################################

def stencil_offsets_to_indices(stencil_offsets : torch.Tensor,
                               ny : int, nx : int,
                               padding : Tuple[int, int, int, int]):

    iys, ixs = stencil_offsets[0], stencil_offsets[1]
    idxby = torch.arange(ny,device=iys.device).reshape(1,-1,1) + padding[0] + iys.reshape(-1,1,1)
    idxbx = torch.arange(nx,device=ixs.device).reshape(1,1,-1) + padding[2] + ixs.reshape(-1,1,1)
    stencil_idx = (idxby, idxbx)
    return stencil_idx

def banded_gauss_seidel_redblack(M : torch.Tensor,
                                 b : torch.Tensor,
                                 grid_type : str,
                                 kernel_size : tuple,
                                 idx_col : torch.Tensor,
                                 stencil_offsets : torch.Tensor,
                                 x_init : torch.Tensor = torch.Tensor([]),
                                 sep_eqs_per_field : bool = False,
                                 thresh : float = 1e-10,
                                 max_iter : int =1000,
                                 **kwargs):
    # Red-black Gauss-Seidel for solving Ax=b for A given by batched tensor M and batched-vector tensor b.
    # Returns solution x and a batchsize-by-N numpy array with errors, where n is the number of
    # Gauss-Seidel iterations until convergence (MSE of Ax - b across batch < thresh).

    assert grid_type in ['grid', 'checker']
    if sep_eqs_per_field:
        assert len(M.shape) == len(b.shape) + 1 # M.shape = (N,L,K,...), b.shape = (N,L,...)
        L,K = M.shape[1:3]
        bandeddot = bandeddot_blockmat_sep_eqs_per_field
        if grid_type == 'grid':
            banded_gauss_seidel_redblack_iter = banded_gs_rb_iter_grid_sep_eqs_per_field
        elif grid_type =='checker':
            banded_gauss_seidel_redblack_iter = banded_gs_rb_iter_checker_sep_eqs_per_field
    else:
        assert len(M.shape) == len(b.shape) + 2 # M.shape = (N,L,L,K,...), b.shape = (N,L,...)
        L,K = M.shape[2:4]
        bandeddot = bandeddot_blockmat
        if grid_type == 'grid':
            banded_gauss_seidel_redblack_iter = banded_gs_rb_iter_grid
        elif grid_type =='checker':
            banded_gauss_seidel_redblack_iter = banded_gs_rb_iter_checker

    pad_sizes = (int(kernel_size[0]//2), int(kernel_size[1]//2))
    padding : Tuple[int, int, int, int] = (pad_sizes[0], pad_sizes[0], pad_sizes[1], pad_sizes[1])
    bounds : Tuple[int, int, int, int] = (pad_sizes[0], - pad_sizes[0] if pad_sizes[0] > 0 else None, 
                                          pad_sizes[1], - pad_sizes[1] if pad_sizes[1] > 0 else None)
    jumps : Tuple[int, int] = ((kernel_size[0]+1)//2, (kernel_size[1]+1)//2)

    if x_init.numel()==0:
        x_shape = (*b.shape[:-2], b.shape[-2]+padding[0]+padding[1], b.shape[-1]+padding[2]+padding[3])
        x = torch.zeros(x_shape, dtype = b.dtype, device = b.device)
    else:
        x = x_init
        if np.all(x.shape[-2:] == b.shape[-2:]):
            x = torch.nn.functional.pad(input = x, pad = padding)
    assert (x.shape[-2] == b.shape[-2]+padding[0]+padding[1])
    assert (x.shape[-1] == b.shape[-1]+padding[2]+padding[3])

    assert stencil_offsets.shape[-1] == K
    stencil_idx = stencil_offsets_to_indices(stencil_offsets, 
                                             ny=M.shape[-2], nx=M.shape[-1],
                                             padding=padding)

    errs = []
    err, ic = torch.tensor(np.inf, requires_grad=False), 0
    while torch.max(err) > thresh and ic < max_iter:
        
        x = banded_gauss_seidel_redblack_iter(M,b,x,idx_col,jumps,stencil_idx,bounds)

        Ax = bandeddot(M, x, stencil_idx=stencil_idx)
        err = torch.mean((Ax-b)**2, dim=(-2,-1))
        errs.append(err)
        ic += 1

    x_valid = x[...,bounds[0]:bounds[1],bounds[2]:bounds[3]]     # get `valid` area without padding

    return x_valid, torch.stack(errs).detach().cpu()


def banded_gs_rb_iter_grid(M : torch.Tensor, b : torch.Tensor, x : torch.Tensor,
                           idx_col: torch.Tensor,
                           jumps: Tuple[int, int],
                           stencil_idx: Tuple[torch.Tensor,torch.Tensor],
                           bounds: Tuple[int, int, int, int]):
    # Single iteration for Red-black Gauss-Seidel for solving Ax=b for batched-matrix tensor A
    # and batched-vector tensor b. Returns updated solution x.
    # Assumes different colors are given simply by different array offsets.
    # This version will assign 

    L,K = M.shape[2:4]
    jy, jx = jumps[0], jumps[1]

    for c in range(idx_col.shape[0]): # in case of more than two colors, denote current color as "red" 

        idx_red = idx_col[c]
        i, locy, locx = idx_red[0], idx_red[1], idx_red[2] # i: variable group, locs: spatial locations 
        # for each current color "red", set "black" = "not red"
        x0b = 1. * x
        x0b_valid = x0b[...,bounds[0]:bounds[1],bounds[2]:bounds[3]] # get `valid` area without padding
        x0b_valid[:,i,locy::jy,locx::jx] = 0.
        
        Ax0b = bandeddot_blockmat(M, x0b, stencil_idx=stencil_idx)

        # immediately overwrite updated state locations !
        x_valid = x[...,bounds[0]:bounds[1],bounds[2]:bounds[3]]     # get `valid` area without padding

        b_red = b[:,i,locy::jy,locx::jx]
        Ax0b_red = Ax0b[:,i,locy::jy,locx::jx]
        M_red_red = M[:, i, i, K//2, locy::jy, locx::jx]
        x_valid[:,i,locy::jy,locx::jx]  = (b_red - Ax0b_red) / M_red_red

    return x


def banded_gs_rb_iter_grid_sep_eqs_per_field(M : torch.Tensor, b : torch.Tensor, x : torch.Tensor,
                                            idx_col: torch.Tensor,
                                            jumps: Tuple[int, int],
                                            stencil_idx: Tuple[torch.Tensor,torch.Tensor],
                                            bounds: Tuple[int, int, int, int]):
    # Single iteration for Red-black Gauss-Seidel for solving Ax=b for batched-matrix tensor A
    # and batched-vector tensor b. Returns updated solution x.
    # Assumes different colors are given simply by different array offsets.

    L,K = M.shape[1:3]
    jy, jx = jumps[0], jumps[1]

    for c in range(idx_col.shape[0]): # in case of more than two colors, denote current color as "red" 

        idx_red = idx_col[c]
        i, locy, locx = idx_red[0], idx_red[1], idx_red[2] # i: variable group, locs: spatial locations 
        # for each current color "red", set "black" = "not red"
        x0b = 1. * x
        x0b_valid = x0b[...,bounds[0]:bounds[1],bounds[2]:bounds[3]] # get `valid` area without padding
        x0b_valid[:,i,locy::jy,locx::jx] = 0.
        
        Ax0b = bandeddot_blockmat_sep_eqs_per_field(M, x0b, stencil_idx=stencil_idx)

        # immediately overwrite updated state locations !
        x_valid = x[...,bounds[0]:bounds[1],bounds[2]:bounds[3]]     # get `valid` area without padding

        b_red = b[:,i,locy::jy,locx::jx]
        Ax0b_red = Ax0b[:,i,locy::jy,locx::jx]
        M_red_red = M[:, i, K//2, locy::jy, locx::jx]
        x_valid[:,i,locy::jy,locx::jx]  = (b_red - Ax0b_red) / M_red_red

    return x


def banded_gs_rb_iter_checker(M : torch.Tensor, b : torch.Tensor, x : torch.Tensor,
                              idx_col: torch.Tensor,
                              jumps: Tuple[int, int],
                              stencil_idx: Tuple[torch.Tensor,torch.Tensor],
                              bounds: Tuple[int, int, int, int]):
    # Single iteration for Red-black Gauss-Seidel for solving Ax=b for batched-matrix tensor A
    # and batched-vector tensor b. Returns updated solution x.
    # Assumes different colors are given simply by different array offsets.

    L,K = M.shape[2:4]
    jy, jx = jumps[0], jumps[1]

    for c in range(idx_col.shape[0]): # in case of more than two colors, denote current color as "red" 

        idx_red = idx_col[c]
        i, locy, locx = idx_red[0], idx_red[1], idx_red[2] # i: variable group, locs: spatial locations 
        # for each current color "red", set "black" = "not red"
        x0b = 1. * x
        x0b_valid = x0b[...,bounds[0]:bounds[1],bounds[2]:bounds[3]] # get `valid` area without padding
        for d in range(min(jx,jy)):
            off_y, off_x = torch.remainder(locy+d,jy), torch.remainder(locx+d,jx)
            x0b_valid[:,i,off_y::jy,off_x::jx] = 0.
        
        Ax0b = bandeddot_blockmat(M, x0b, stencil_idx=stencil_idx)

        # immediately overwrite updated state locations !
        x_valid = x[...,bounds[0]:bounds[1],bounds[2]:bounds[3]]     # get `valid` area without padding

        for d in range(min(jx,jy)):
            off_y, off_x = torch.remainder(locy+d,jy), torch.remainder(locx+d,jx)
            b_red = b[:,i,off_y::jy,off_x::jx]
            Ax0b_red = Ax0b[:,i,off_y::jy,off_x::jx]
            M_red_red = M[:, i, i, K//2, off_y::jy, off_x::jx]
            x_valid[:,i,off_y::jy,off_x::jx]  = (b_red - Ax0b_red) / M_red_red

    return x


def banded_gs_rb_iter_checker_sep_eqs_per_field(M : torch.Tensor, b : torch.Tensor, x : torch.Tensor,
                                                idx_col: torch.Tensor,
                                                jumps: Tuple[int, int],
                                                stencil_idx: Tuple[torch.Tensor,torch.Tensor],
                                                bounds: Tuple[int, int, int, int]):
    # Single iteration for Red-black Gauss-Seidel for solving Ax=b for batched-matrix tensor A
    # and batched-vector tensor b. Returns updated solution x.
    # Assumes different colors are given simply by different array offsets.

    L,K = M.shape[1:3]
    jy, jx = jumps[0], jumps[1]

    for c in range(idx_col.shape[0]): # in case of more than two colors, denote current color as "red" 

        idx_red = idx_col[c]
        i, locy, locx = idx_red[0], idx_red[1], idx_red[2] # i: variable group, locs: spatial locations 
        # for each current color "red", set "black" = "not red"
        x0b = 1. * x
        x0b_valid = x0b[...,bounds[0]:bounds[1],bounds[2]:bounds[3]] # get `valid` area without padding
        for d in range(min(jx,jy)):
            off_y, off_x = torch.remainder(locy+d,jy), torch.remainder(locx+d,jx)
            x0b_valid[:,i,off_y::jy,off_x::jx] = 0.
        
        Ax0b = bandeddot_blockmat_sep_eqs_per_field(M, x0b, stencil_idx=stencil_idx)

        # immediately overwrite updated state locations !
        x_valid = x[...,bounds[0]:bounds[1],bounds[2]:bounds[3]]     # get `valid` area without padding

        for d in range(min(jx,jy)):
            off_y, off_x = torch.remainder(locy+d,jy), torch.remainder(locx+d,jx)
            b_red = b[:,i,off_y::jy,off_x::jx]
            Ax0b_red = Ax0b[:,i,off_y::jy,off_x::jx]
            M_red_red = M[:, i, K//2, off_y::jy, off_x::jx]
            x_valid[:,i,off_y::jy,off_x::jx]  = (b_red - Ax0b_red) / M_red_red

    return x


def biCGstab_l(M : torch.Tensor,
               b : torch.Tensor,
               kernel_size : tuple,
               stencil_offsets : torch.Tensor,
               x_init : torch.Tensor = torch.Tensor([]),
               sep_eqs_per_field : bool = False,
               thresh : float = 1e-10,
               max_iter : int = 1000,
               l : int = 4,
               **kwargs):
    # Implementation of BiCGstab(l) algorithm for solving Ax=b for A given by batched tensor M 
    # and batched-vector tensor b.
    # Returns solution x and a batchsize-by-N numpy array with errors, where n is the number of
    # BiCGstab(l) iterations until convergence (MSE of Ax - b across batch < thresh).

    # Direct implementation of algorithm 3.1 from
    # Sleijpen, G. L., & Fokkema, D. R. (1993).
    # BiCGstab (ell) for linear equations involving unsymmetric matrices with complex spectrum.
    # Electronic Transactions on Numerical Analysis., 1, 11-32.

    if sep_eqs_per_field:
        assert len(M.shape) == len(b.shape) + 1 # M.shape = (N,L,K,...), b.shape = (N,L,...)
        L,K = M.shape[1:3]
        bandeddot = bandeddot_blockmat_sep_eqs_per_field
    else:
        assert len(M.shape) == len(b.shape) + 2 # M.shape = (N,L,L,K,...), b.shape = (N,L,...)
        L,K = M.shape[2:4]
        bandeddot = bandeddot_blockmat

    pad_sizes = (int(kernel_size[0]//2), int(kernel_size[1]//2))
    padding : Tuple[int, int, int, int] = (pad_sizes[0], pad_sizes[0], pad_sizes[1], pad_sizes[1])
    bounds : Tuple[int, int, int, int] = (pad_sizes[0], - pad_sizes[0] if pad_sizes[0] > 0 else None, 
                                          pad_sizes[1], - pad_sizes[1] if pad_sizes[1] > 0 else None)

    if x_init.numel()==0:
        x_shape = (*b.shape[:-2], b.shape[-2]+padding[0]+padding[1], b.shape[-1]+padding[2]+padding[3])
        x = torch.zeros(x_shape, dtype = b.dtype, device = b.device)
    else:
        x = x_init
        if np.all(x.shape[-2:] == b.shape[-2:]):
            x = torch.nn.functional.pad(input = x, pad = padding)
    assert (x.shape[-2] == b.shape[-2]+padding[0]+padding[1])
    assert (x.shape[-1] == b.shape[-1]+padding[2]+padding[3])

    assert stencil_offsets.shape[-1] == K
    stencil_idx = stencil_offsets_to_indices(stencil_offsets, 
                                             ny=M.shape[-2], nx=M.shape[-1],
                                             padding=padding)
    def inner(x,y):
        return (x*y).sum(dim=(-3,-2,-1))

    # initialize algorithm
    Ax = bandeddot(M, x, stencil_idx=stencil_idx)
    r =  torch.nn.functional.pad(input = b - Ax, pad = padding)
    r0 = 1. * r / inner(r,r)
    rho = omega = 1.
    alpha = 0.

    u = torch.zeros_like(r)
    u_hat = torch.zeros_like(r).repeat((l, 1,1,1,1))
    r_hat = torch.zeros_like(r).repeat((l, 1,1,1,1))

    tau = torch.zeros((l,l))
    sigma = torch.ones(l)
    gamma, gamma_, gamma__ = torch.zeros(l), torch.zeros(l), torch.zeros(l)

    errs = []
    err, ic = torch.tensor(np.inf, requires_grad=False), 0
    while torch.max(err) > thresh and ic < max_iter:

        rho = - omega * rho

        # Bi-CG part
        for j in range(l):
            rho_n = inner(r0, r if j==0 else r_hat[j-1])
            beta = (rho_n/rho)*alpha
            rho = rho_n

            u = r - beta * u
            u_hat[:j] = r_hat[:j] - beta * u_hat[:j]
            valid = bandeddot(M, u_hat[j-1] if j>0 else u, stencil_idx = stencil_idx)
            u_hat[j][...,bounds[0]:bounds[1],bounds[2]:bounds[3]] = valid

            alpha = rho/inner(u_hat[j], r0)
            r = r - alpha * u_hat[0]
            r_hat[:j] = r_hat[:j] - alpha * u_hat[1:j+1]
            valid = bandeddot(M, r_hat[j-1] if j>0 else r, stencil_idx = stencil_idx)
            r_hat[j][...,bounds[0]:bounds[1],bounds[2]:bounds[3]] = valid

            x = x + alpha * u

        # mod G-S / MR part
        for j in range(l):
            for i in range(j):
                tau[i,j] = 1/sigma[i] * inner(r_hat[i], r_hat[j])
                r_hat[j] = r_hat[j] - tau[i,j] * r_hat[i]
            sigma[j] = inner(r_hat[j], r_hat[j])
            gamma_[j] = inner(r_hat[j], r) / sigma[j]
        omega = gamma[-1] = gamma_[-1]
        for j in range(l-2,-1,-1):
            gamma[j] = gamma_[j] - torch.sum(tau[j,j+1:] * gamma[j+1:])
        for j in range(l-1):
            gamma__[j] = gamma[j+1] + torch.sum(tau[j,j+1:-1] * gamma[j+2:])

        # update
        x = x + gamma[0] * r
        for j in range(l):
            u = u - gamma[j] * u_hat[j]
            r = r - gamma_[j] * r_hat[j]
        for j in range(l-1):
            x = x + gamma__[j] * r_hat[j]

        # check for convergence
        Ax = bandeddot(M, x, stencil_idx=stencil_idx)
        err = torch.mean((Ax-b)**2, dim=(-2,-1))
        errs.append(err)
        ic += 1

    x_valid = x[...,bounds[0]:bounds[1],bounds[2]:bounds[3]]     # get `valid` area without padding

    return x_valid, torch.stack(errs).detach().cpu()
