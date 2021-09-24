import numpy as np
import torch
from implem.utils import init_torch_device, as_tensor, device, dtype, dtype_np
from implem.utils import banded_gauss_seidel_redblack
import matplotlib.pyplot as plt

###########################################
# Semi-implicit numerical solvers for SWE #
###########################################

def swe1D_true_bM(x, dt, dx, g, w_imp, cd, data_scales, comp_u='solve'):
    # 1-D Shallow-Water equations semi-implicit solver
    # ported to python from Fortran code written by
    #               Kai Logemann, HZG, October 2020

    assert comp_u in ['solve', 'calculate']
    N = x.shape[0]

    ksi = x[:,0] * data_scales.flatten()[0]
    u   = x[:,1] * data_scales.flatten()[1]
    dp  = x[:,-2] * data_scales.flatten()[2]

    h = ksi + dp    
    nx = ksi.shape[-1]

    Ce, Cw = torch.zeros_like(ksi), torch.zeros_like(ksi)
    uint, us, uints = torch.zeros_like(ksi), torch.zeros_like(ksi), torch.zeros_like(ksi)

    i, ip = np.arange(1, nx-2), np.arange(2, nx-1)
    hm = (h[:,i]+h[:,ip]) / 2.
    Cw[:,ip] = Ce[:,i] = dt**2/dx**2 * w_imp**2 * g * hm
    uint[:,i] = hm * u[:,i]    
    us[:,i] = u[:,i] - dt*cd/hm*u[:,i]*torch.abs(u[:,i]) - (1.-w_imp)*dt*g*(ksi[:,ip]-ksi[:,i])/dx
    uints[:,i] = hm * us[:,i]


    if comp_u == 'solve':
        M_true = torch.zeros((N,2,2,3,1,nx))
    elif comp_u == 'calculate':
        M_true = torch.zeros((N,1,1,3,1,nx))

    # dksi_new/dksi_old
    M_true[:,0,0,0,0] = - Cw / (1. + Ce + Cw) # Cw: first two elements and last element zero
    M_true[:,0,0,1,0] = 1.
    M_true[:,0,0,2,0] = - Ce / (1. + Ce + Cw) # Ce: first element and last two elements zero


    if comp_u == 'solve':
        # will return system of linear equations to solve for both new xi and u

        # du_new/dksi_old
        M_true[:,1,0,1,0,i] = - dt/dx * w_imp * g # main diagonal and first upper diaginal: 
        M_true[:,1,0,2,0,i] = dt/dx * w_imp * g   # first element and last two elements zero

        # du_new/du_old
        M_true[:,1,1,1,0] = 1.

        b_true = torch.zeros((N,2*nx))
        i, im = np.arange(1, nx-1), np.arange(0, nx-2)

        # water height
        # old state
        b_true[:,i] = ksi[:,i] # part of b for xi: first element and last element zero 
        # div
        b_true[:,i] -= dt*(1.-w_imp)/dx * (uint[:,i] - uint[:,im])
        b_true[:,i] -= dt*w_imp/dx * (uints[:,i] - uints[:,im])
        # normalize rows of A & b:
        b_true[:,:nx] /= (1 + Ce + Cw)

        # velocities
        b_true[:,nx:] = us  # part of b for u: first element and last two elements zero  (via us)

        b_true = b_true.reshape(N,2,1,nx)

    if comp_u == 'calculate':
        # will return system of linear equations to solve only for new xi. 
        # New u will be calculated from that in an extra step.

        b_true = torch.zeros((N,nx))
        i, im = np.arange(1, nx-1), np.arange(0, nx-2)

        # water height
        # old state
        b_true[:,:nx] = ksi
        # closed boundaries
        b_true[:,0] = b_true[:,nx-1] = 0.
        # div
        b_true[:,i] -= dt*(1.-w_imp)/dx * (uint[:,i] - uint[:,im])
        b_true[:,i] -= dt*w_imp/dx * (uints[:,i] - uints[:,im])
        # normalize rows of A & b:
        b_true[:,:nx] /= (1 + Ce + Cw)

        b_true = b_true.reshape(N,1,1,nx)

    return b_true, M_true, us

def swe1D_true_solve(x, dt, dx, g, w_imp, cd, data_scales, settings, comp_u='solve'):

    assert comp_u in ['solve', 'calculate']

    b_true, M_true, us = swe1D_true_bM(x, dt, dx, g, w_imp, cd, data_scales, comp_u)

    pred_hat, diagnostics = banded_gauss_seidel_redblack(M_true, b_true, **settings)

    if comp_u == 'solve':
        pred_hat_out = pred_hat
        
    elif comp_u == 'calculate':
        pred_hat_out = torch.zeros((x.shape[0],2,1,x.shape[-1]))
        pred_hat_out[:,:1] = pred_hat
        pred_hat_out[:,1,0,1:-2] = us[:,1:-2] - dt/dx * w_imp * g * (pred_hat_out[:,0,0,2:-1] - pred_hat_out[:,0,0,1:-2])

    return pred_hat_out, diagnostics

def swe1D_get_init(N, nx):

    # set depth profile d [m]
    sigma, tau, mu = 10., 100., 15
    profile = sigma * np.exp(- (np.arange(2*nx)- nx)**2 / tau)
    mean, cov = np.ones(nx) * mu, np.empty((nx,nx))
    cov[0] = profile[nx:]
    for i in range(1,nx):
        cov[i] = profile[nx-(i):-i]
    d = np.atleast_2d(np.random.multivariate_normal(mean, cov, size=N))
    d[:,0] = d[:,-1] = - mu       
    assert np.all(d[:,1:-1] > 0) # no dry cells within domain for now

    # set initial sea surface elevation zeta [m]
    zeta = 0 * np.sqrt(0.001) * np.random.normal(size=(N,nx))
    zeta[np.arange(N),np.random.randint(nx, size=N)] = 0.05 + np.random.random(size=N) * 0.1 # initial disturbance
    # set initial velocities u [m/s]
    u = 0. * np.random.normal(size=(N,nx))

    mask = np.ones((N,nx))
    mask[:,0] = mask[:,-1] = 0.

    return zeta, u, d, mask


def swe1D_sim(N, T, nx, dt, dx, g, w_imp, cd, data_scales, settings, 
              verbose=False, comp_u='solve', init_vals=None):
    
    zeta, u, d, mask = swe1D_get_init(N=N, nx=nx)
    x0 = np.stack((zeta, u, d, mask), axis=1)
    if not init_vals is None:
        assert init_vals.shape == x0.shape 
        x0 = init_vals
        d, mask = init_vals[:,-2], init_vals[:,-1]

    statics = as_tensor(np.stack((d, mask), axis=1))

    out = [as_tensor(x0)]
    with torch.no_grad():
        for t in range(T):
            if verbose and np.mod(t,10)==0:
                print(f'simulating {t}/{T}')
            #x_est = model(x)
            x_est, diagnostics = swe1D_true_solve(out[-1], dt, dx, g, w_imp, cd, 
                                             data_scales, settings=settings, comp_u=comp_u)
            x_est = x_est.squeeze(-2)/as_tensor(data_scales[0,:,:2])
            out.append(torch.cat((x_est, statics), dim=1))

    return torch.stack(out, axis=1)


def swe2D_true_bM(x, dt, dx, g, w_imp, cd, ah, data_scales, comp_u='solve'):
    # 2-D Shallow-Water equations semi-implicit solver
    # ported to python from Fortran code written by
    #               Kai Logemann, HZG, October 2020

    N = x.shape[0]
    ny, nx = x.shape[-2:]

    ksi = x[:,0] * data_scales.flatten()[0]
    u   = x[:,1] * data_scales.flatten()[1]
    v   = x[:,2] * data_scales.flatten()[2]
    dp  = x[:,3] * data_scales.flatten()[3]

    h = ksi + dp

    # equation of motion => us,vs (interim solution)
    # vertical integrals of velocity => uint, and interim solution uints
    # coefficients for zetan system of equations: ce,cw,cn,cs

    Ce, Cw = torch.zeros_like(ksi), torch.zeros_like(ksi)
    Cn, Cs = torch.zeros_like(ksi), torch.zeros_like(ksi)
    uint, us, uints = torch.zeros_like(ksi), torch.zeros_like(ksi), torch.zeros_like(ksi)

    i, ip, im = np.arange(1, ny-2), np.arange(2, ny-1), np.arange(0, ny-3)
    j, jp, jm = np.arange(1, nx-1), np.arange(2, nx), np.arange(0, nx-2)
    hm = (h[:,i][:,:,j]+h[:,ip][:,:,j]) / 2.
    uij = u[:,i][:,:,j]
    usij = uij - dt*cd/hm*uij*torch.abs(uij) 
    usij -= (1.-w_imp)*dt*g*(ksi[:,ip][:,:,j]-ksi[:,i][:,:,j])/dx
    usij += dt*ah*(u[:,ip][:,:,j]+ u[:,im][:,:,j] + u[:,i][:,:,jp] + u[:,i][:,:,jm] - 4.*uij)/dx**2
    usi, uinti, uintsi = us[:,i], uint[:,i], uints[:,i]
    usi[:,:,j] = usij
    us[:,i] = usi
    uinti[:,:,j] = hm * uij
    uint[:,i] = uinti
    uintsi[:,:,j] = hm * usij
    uints[:,i] = uintsi

    Cei = Ce[:,i]
    Cei[:,:,j] = dt**2/dx**2 * w_imp**2 * g * hm 
    Ce[:,i] = Cei

    vint, vs, vints = torch.zeros_like(ksi), torch.zeros_like(ksi), torch.zeros_like(ksi)
    i, ip, im = np.arange(1, ny-1), np.arange(2, ny), np.arange(0, ny-2)
    j, jp, jm = np.arange(1, nx-2), np.arange(2, nx-1), np.arange(0, nx-3)
    hm = (h[:,i][:,:,j]+h[:,i][:,:,jp]) / 2.
    vij = v[:,i][:,:,j]
    vsij = vij - dt*cd/hm*vij*torch.abs(vij) 
    vsij -= (1.-w_imp)*dt*g*(ksi[:,i][:,:,jp]-ksi[:,i][:,:,j])/dx
    vsij += dt*ah*(v[:,ip][:,:,j]+ v[:,im][:,:,j] + v[:,i][:,:,jp] + v[:,i][:,:,jm] - 4.*vij)/dx**2

    vsi, vinti, vintsi = vs[:,i], vint[:,i], vints[:,i]
    vsi[:,:,j] = vsij
    vs[:,i] = vsi
    vinti[:,:,j] = hm * vij
    vint[:,i] = vinti
    vintsi[:,:,j] = hm * vsij
    vints[:,i] = vintsi

    Cni = Cn[:,i]
    Cni[:,:,j] = dt**2/dx**2 * w_imp**2 * g * hm 
    Cn[:,i] = Cni

    i, im, j = np.arange(2, ny-1), np.arange(1, ny-2), np.arange(1, nx-1)
    hm = (h[:,i][:,:,j]+h[:,im][:,:,j]) / 2.
    Cwi = Cw[:,i]
    Cwi[:,:,j] = dt**2/dx**2 * w_imp**2 * g * hm
    Cw[:,i] = Cwi

    i, j, jm = np.arange(1, ny-1), np.arange(2, nx-1), np.arange(1, nx-2)
    hm = (h[:,i][:,:,j]+h[:,i][:,:,jm]) / 2.
    Csi = Cs[:,i]
    Csi[:,:,j] = dt**2/dx**2 * w_imp**2 * g * hm
    Cs[:,i] = Csi

    if comp_u == 'solve':
        M_true = torch.zeros((N,3,3,5,ny,nx))
    elif comp_u == 'calculate':
        M_true = torch.zeros((N,1,5,ny,nx))

    M_true[:,0,0] = - Cw / (1. + Ce + Cw + Cs + Cn) # in the Fortran code, i+/-1 signifies east-west
    M_true[:,0,1] = - Cs / (1. + Ce + Cw + Cs + Cn) #                      j+/-1 signifies north-south
    M_true[:,0,2] = 1.                              # torch.nn.unfold works row-wise (first row, second row...)
    M_true[:,0,3] = - Cn / (1. + Ce + Cw + Cs + Cn) # i.e. A*vec(x)=b means vec(x) is first row, second row...
    M_true[:,0,4] = - Ce / (1. + Ce + Cw + Cs + Cn) # and hence [[     0=W     ]
                                                    #            [1=S   2   3=N] for a 3x3 stencil ... ?
                                                    #            [     4=E   0]]

    if comp_u == 'solve':
        raise NotImplementedError()

    elif comp_u == 'calculate':

        b_true = torch.zeros((N, ny, nx))
        i, im = np.arange(1, ny-1), np.arange(0, ny-2)
        j, jm = np.arange(1, nx-1), np.arange(0, nx-2)

        # water height
        # old state
        b_true[:,:ny] = ksi
        # closed boundaries
        b_true[:,0] = b_true[:,ny-1] = 0.
        b_true[:,:,0] = b_true[:,:,nx-1] = 0.
        # div
        b_truei = b_true[:,i]
        b_truei[:,:,j] -= dt*(1.-w_imp)/dx * (uint[:,i][:,:,j] - uint[:,im][:,:,j])
        b_truei[:,:,j] -= dt*w_imp/dx * (uints[:,i][:,:,j] - uints[:,im][:,:,j])
        b_truei[:,:,j] -= dt*(1.-w_imp)/dx * (vint[:,i][:,:,j] - vint[:,i][:,:,jm])
        b_truei[:,:,j] -= dt*w_imp/dx * (vints[:,i][:,:,j] - vints[:,i][:,:,jm])
        b_true[:,i] = b_truei

        # normalize rows of A & b:
        b_true[:,:ny] /= (1. + Ce + Cw + Cn + Cs)

        b_true = b_true.reshape(N,1,ny,nx)

    return b_true, M_true, us, vs


def swe2D_true_solve(x, dt, dx, g, w_imp, cd, ah, data_scales, settings, comp_u='solve'):

    assert comp_u in ['solve', 'calculate']
    assert settings['stencil_offsets'].shape[1] == 5 # 5-point stencil !

    b_true, M_true, us, vs = swe2D_true_bM(x, dt, dx, g, w_imp, cd, ah, data_scales, comp_u)

    pred_hat, diagnostics = banded_gauss_seidel_redblack(M_true, b_true, **settings)

    if comp_u == 'solve':
        raise NotImplementedError()
        
    elif comp_u == 'calculate':
        
        zetan = pred_hat[:,0]
        ny, nx = zetan.shape[-2:]

        un, vn = torch.zeros_like(us), torch.zeros_like(vs)
        i, ip, j  = np.arange(1, ny-2), np.arange(2, ny-1), np.arange(1, nx-1)

        uni = un[:,i]
        uni[:,:,j] = us[:,i][:,:,j] - w_imp*dt/dx*g*(zetan[:,ip][:,:,j]-zetan[:,i][:,:,j])
        un[:,i] = uni

        i, j, jp = np.arange(1, ny-1), np.arange(1, nx-2), np.arange(2, nx-1)

        vni = vn[:,i]
        vni[:,:,j] = vs[:,i][:,:,j] - w_imp*dt/dx*g*(zetan[:,i][:,:,jp]-zetan[:,i][:,:,j])
        vn[:,i] = vni

        pred_hat_out = torch.stack((zetan, un, vn), axis=1)

    return pred_hat_out, diagnostics


def swe2D_get_init(N, my, nx):

    # set constant depth profile d [m]
    sigma, tau, mu = 5., 100., 15
    mean = np.ones((nx*my,1)) * mu
    x,y = np.stack(np.meshgrid(np.arange(my), np.arange(nx))).reshape(2,-1)
    dists = (x.reshape(-1,1) - x.reshape(1,-1))**2 + (y.reshape(-1,1) - y.reshape(1,-1))**2
    cov = sigma * np.exp(- dists / tau) + 1e-4 * np.eye(nx*my)
    L = np.linalg.cholesky(cov)
    d = np.random.normal(size=(nx*my,N))
    d = (L.dot(d) + mean).reshape(nx,my,N)
    d[:,0] = d[:,-1] = -mu
    d[0,:] = d[-1,:] = - mu       
    assert np.all(d[1:-1][:,1:-1] > 0) # no dry cells within domain for now

    # set initial sea surface elevation zeta [m]
    zeta = 0. * np.sqrt(0.00001) * np.random.normal(size=(nx,my,N))

    for i in range(N):
        idn, idm = np.random.randint(nx), np.random.randint(my)
        pert_size = 5 + np.random.randint(6)
        idn = range(max(0, -(pert_size//2)+idn),  min(nx, (pert_size+1)//2+idn))
        idm = range(max(0, -(pert_size//2)+idm),  min(my, (pert_size+1)//2+idm))
        zetai = zeta[:,:,i]
        zetai[np.ix_(idn, idm)] = 0.05 + np.random.random() * 0.1 # initial disturbance
        zeta[:,:,i] = zetai
    
    # set initial velocities u [m/s]
    u = 0. * np.random.normal(size=(nx,my,N))
    v = 0. * np.random.normal(size=(nx,my,N))

    # boundary mask (boundaries = dry cells)
    mask = np.ones((N,my,nx))
    mask[:,0] = mask[:,-1] = 0.
    mask[:,:,0] = mask[:,:,-1] = 0.

    return zeta.transpose(), u.transpose(), v.transpose(), d.transpose(), mask


def swe2D_sim(N, T, my, nx, dt, dx, g, w_imp, cd, ah, data_scales, settings, 
              verbose=False, comp_u='calculate', init_vals=None):

    zeta, u, v, d, mask = swe2D_get_init(N=N, nx=nx, my=my)
    if not init_vals is None:
        assert init_vals.shape == d.shape
        d = 1. * init_vals
    statics = as_tensor(np.stack((d, mask), axis=1))

    out = [as_tensor(np.stack((zeta, u, v, d + zeta, mask), axis=1))] # d+zeta because 2D code expects h!
    with torch.no_grad():
        for t in range(T):
            if verbose and np.mod(t,10)==0:
                print(f'simulating {t}/{T}')
            x_est, diagnostics = swe2D_true_solve(out[-1], dt, dx, g, w_imp, cd, ah, 
                                             data_scales, settings=settings, comp_u=comp_u)
            x_est = x_est.squeeze(-2)/as_tensor(data_scales[0,:,:3])
            out.append(torch.cat((x_est, statics), dim=1))
            out[-2] = out[-2].detach().cpu() # freeing up GPU memory
    out[-1] = out[-1].detach().cpu()

    return torch.stack(out, axis=1)


def swe_sim(N, T, setup, solver_settings, verbose=False):
    
    assert setup['instance_dimensionality'] in ['1D', '2D']
    if not 'w_imp' in setup.keys():
        setup['w_imp'] = 0.5

    if setup['instance_dimensionality'] == '1D':
        if not 'data_scales' in setup.keys():
                setup['data_scales'] = np.ones((1,1,4,1))
        out = swe1D_sim(N, T, 
                        nx=setup['nx'],
                        dt=setup['dt'],
                        dx=setup['dx'],
                        g=setup['g'],
                        w_imp=setup['w_imp'],
                        cd=setup['cd'],
                        data_scales=setup['data_scales'], 
                        settings=solver_settings, 
                        verbose=verbose, 
                        comp_u=setup['comp_u'])
    else:
        if not 'data_scales' in setup.keys():
            setup['data_scales'] = np.ones((1,1,5,1,1))
        if not 'my' in setup.keys():
            setup['my'] = 1 * setup['nx']
        out = swe2D_sim(N, T, 
                        my=setup['my'],
                        nx=setup['nx'],
                        dt=setup['dt'],
                        dx=setup['dx'],
                        g=setup['g'],
                        w_imp=setup['w_imp'],
                        cd=setup['cd'],
                        ah=setup['ah'],
                        data_scales=setup['data_scales'], 
                        settings=solver_settings, 
                        verbose=verbose, 
                        comp_u=setup['comp_u'])
    return out

def plot_results_swe(data_numerical, data_model, i=0, swe_model='1D', if_save=False, fig_path=None, plot_times=None):

    if if_save:
        assert not fig_path is None

    assert swe_model in ['1D', '2D']

    fontsize = 14
    if swe_model == '1D':
        plt.figure(figsize=(16,10))
        x, x_hat = data_numerical, data_model

        vmax_diff = np.abs(x - x_hat).max()
        vmin, vmax = np.min((x.min(), x_hat.min())), np.max((x.max(), x_hat.max()))
        plt.subplot(3,1,1)
        plt.imshow(x.T, vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('data', fontsize=fontsize)
        plt.colorbar()

        plt.subplot(3,1,2)
        plt.imshow(x_hat.T, vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('model', fontsize=fontsize)
        plt.colorbar()

        plt.subplot(3,1,3)
        plt.imshow(x.T - x_hat.T, vmin=-vmax_diff, vmax=vmax_diff, cmap='bwr', aspect='auto')
        plt.yticks([])
        plt.xticks(fontsize=fontsize)
        plt.ylabel('difference', fontsize=fontsize)
        plt.xlabel('time step', fontsize=fontsize)
        plt.colorbar()

        if i == 0:
            plt.suptitle('water depth', fontsize=fontsize, x=0.45, y=0.93)
        elif i == 1:
            plt.suptitle('velocity u', fontsize=fontsize, x=0.45,y=0.93)
        elif i == 2:
            plt.suptitle('mask', fontsize=fontsize, x=0.45, y=0.93)

    elif swe_model == '2D':

        plot_times = np.arange(0, len(data_model), len(data_model)//4) if plot_times is None else plot_times
        num_times = len(plot_times)
        plt.figure(figsize=(16,14))
        x, x_hat = data_numerical, data_model

        vmax_diff = np.abs(x[plot_times] - x_hat[plot_times]).max()

        for j,t in enumerate(plot_times):
            vmin, vmax = np.min((x[t].min(), x_hat[t].min())), np.max((x[t].max(), x_hat[t].max()))
            plt.subplot(len(plot_times),num_times,1+num_times+j)
            plt.imshow(x[t].T, vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.title(f't = {t}')
            if j == 0:
                plt.ylabel('data', fontsize=fontsize)
            plt.colorbar()
            plt.subplot(len(plot_times),num_times,1+(2)*num_times+j)
            plt.imshow(x_hat[t].T, vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.ylabel('model', fontsize=fontsize)
            plt.colorbar()
            plt.subplot(len(plot_times),num_times,1+(3)*num_times+j)
            plt.imshow(x[t].T - x_hat[t].T, vmin=-vmax_diff, vmax=vmax_diff, cmap='bwr', aspect='auto')
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.ylabel('difference', fontsize=fontsize)
            plt.colorbar()

        if i == 0:
            plt.suptitle('water depth', fontsize=fontsize, x=0.45, y=0.73)
        elif i == 1:
            plt.suptitle('velocity u', fontsize=fontsize, x=0.45,y=0.73)
        elif i == 2:
            plt.suptitle('velocity v', fontsize=fontsize, x=0.45,y=0.73)

    if if_save:
        if i == 0:
            plt.savefig(fig_path + 'example_rollout_water_depth_2D_no_boundaries.pdf')
        elif i == 1:
            plt.savefig(fig_path + 'example_rollout_velocities_u_2D_no_boundaries.pdf')
        elif i == 2:
            plt.savefig(fig_path + 'example_rollout_velocities_v_2D_no_boundaries.pdf')

    plt.show()


################################################
# SWE-specific code for implem implicit models #
################################################

def enforce_bcs(x, z):
    # boundary conditions right on boundary pixels. x = (water height, u, v) is full state !
    cond_mask = z[:,-1:] # z could be data point, has to have boundary mask on last channel.
    x[:,:3] = cond_mask * x[:,:3]
    x[:,1,-2,:] = 0. # u at southern border is zero
    x[:,2,:,-2] = 0. # v at eastern border is zero
    return x


def fix_bM_BCs(settings, x, b, M):
    # ensures that boundary conditions are met by externally overwriting parts of M,b.
    # Note that when writing M = mask * Md + (1-mask) * M in torch, gradients are tracked!

    cond_mask = 1. - x[:,-1:]
    with torch.no_grad():
        ba = b[:,:1].detach()
        Ma = M[:,:1].detach() if settings['sep_eqs_per_field'] else M[:,0,:1].detach()
        bd, Md = gen_bM_fixed_BCs(settings=settings,
                                cond_mask=cond_mask,
                                M=Ma, b=ba)

    # paste in manualy written BC equations over boundary grid points:
    if settings['sep_eqs_per_field']:
        M[:,:1] = cond_mask.unsqueeze(-3) * Md + (1.-cond_mask.unsqueeze(-3)) * M[:,:1]
    else:
        # M[:,0,:1] dependence of first field on itself, M[:,0,:1] dependence on other fields.
        M[:,0,:1] = cond_mask.unsqueeze(-3) * Md + (1.-cond_mask.unsqueeze(-3)) * M[:,0,:1]
        M[:,0,1:] = cond_mask.unsqueeze(-3) * 0. + (1.-cond_mask.unsqueeze(-3)) * M[:,0,1:]
    b[:,:1] = cond_mask * bd + (1.-cond_mask) * b[:,:1]

    # tricky one: boundary conditions bleed into neighbouring pixels !
    # could be more elegantly done by convolving the boundary mask cond_mask with
    # kernels [0,0,1], [1,0,0] resp. [[0],[0],[1]], [[1],[0],[0]] to find grid points
    # neighouring the respective boundary 
    stencil_offsets = settings['stencil_offsets'].detach().cpu().numpy()
    idxt = np.where((stencil_offsets[0] == -1) * (stencil_offsets[1] == 0))[0][0] # top
    M[...,idxt, 1, :] =  0. # Northern end
    idxb = np.where((stencil_offsets[0] ==  1) * (stencil_offsets[1] == 0))[0][0] # bottom
    M[...,idxb,-2, :] =  0. # Southern end
    idxl = np.where((stencil_offsets[0] ==  0) * (stencil_offsets[1] ==-1))[0][0] # left
    M[...,idxl, :, 1] =  0. # Western end
    idxr = np.where((stencil_offsets[0] ==  0) * (stencil_offsets[1] == 1))[0][0] # right
    M[...,idxr, :,-2] =  0. # Eastern end

    backwards_mask = None

    return b, M, backwards_mask

def gen_bM_fixed_BCs(settings, cond_mask, b=None, M=None):
    # ensure that the system of linear equations defined by M,b have equations 
    # 1. * x_i = 0. 
    # for boundary grid points x_i, meaning we force these values to be zero.
    # Note that this is different from just zero-ing these values after the solve,
    # since the grid points within the domain could depend on them during the solve!

    ny, nx = cond_mask.shape[-2:]
    stencil_offsets = settings['stencil_offsets'].detach().cpu().numpy()

    # output matrix for system of linear equations that satisfies BCs
    M_shape = (cond_mask.shape[0],1,stencil_offsets.shape[1],ny,nx)
    if M is None:
        M = torch.zeros(M_shape,dtype=v_cond.dtype)
        M[:,:,idxc] = 1.
    else:
        assert np.all(M.shape == M_shape)
        M = 0. * cond_mask.unsqueeze(-3) + (1 - cond_mask.unsqueeze(-3)) * M

    # retrieve axis indices for different stencil positions (we need a 5-point cross stencil)
    idxc = np.where((stencil_offsets[0] ==  0) * (stencil_offsets[1] == 0))[0][0] # center
    M[:,:,idxc] =  1. * cond_mask + (1 - cond_mask) * M[:,:,idxc] # center

    # right-hand side of system of linear equations
    b_shape = (cond_mask.shape[0],1,ny,nx)
    if b is None:
        b = torch.zeros(b_shape,dtype=v_cond.dtype)
    else:
        assert np.all(b.shape == b_shape)

    b[...,0] = b[...,-1] = 0.
    b[...,0,:] = b[...,-1,:] = 0.

    return b, M


def network_init_swe(model, x, data_scales, dt=300., dx=1e4, w_imp=0.5, g=9.81, cd=1e-3, ah=1e3):
    # network initialization for implicit network to solve 2D SWE
    # ensures that water height is computed in the first implicit layer (one or zero solved fields)
    # ensures that u,v are computed in the second implicit layer with zero solved fields (explicit mode!)

    # first implicit layer (block of several conv layers!)
    if model.impl_layers[0].hparams.implicit_mode: # if training (semi-)implicit model

        # initialize deep implicit layer to some constant fixed 5-point stencil
        dp  = x[:,3] * data_scales.flatten()[3]
        fcs = dp.mean() * dt**2/dx**2 * w_imp**2 * g # typical values for Cn,Cs,Cw,Ce averaged across space
        final_forward_layer = model.impl_layers[0].forward_model.layers[-1]
        bias, weight = final_forward_layer.bias, final_forward_layer.weight
        bias_new, weight_new = 1.* bias, 1.* weight
        bias_new[:5] = - fcs + 0.*bias[:5]
        bias_new[2] = 1. + 0.*bias[2]
        weight_new[:5] = weight_new[:5]/1e2
        final_forward_layer.bias = torch.nn.Parameter(bias_new)
        final_forward_layer.weight = torch.nn.Parameter(weight_new)

        # 50 iterations are okay for convergence of systems for dt=300,900,1500, but beyond
        # the diagonal dominance of the true matrix A is so weak that more iterations are needed.
        model.impl_layers[0].settings['max_iter_forward'] = 50
        model.impl_layers[0].settings['max_iter_backward'] = 50
        model.impl_layers[0].settings['thresh_forward'] = 1e-14  # input normalized to scale ~1
        model.impl_layers[0].settings['thresh_backward'] = 1e-24 # higher tolerance since dL/dz can be tiny!

    if not model.impl_layers[0].hparams.implicit_mode:  # if training purely explicit model
        pass

    # fix first out_conv to ensure passing on results of implicit layer output to second out_conv (skip connection!)
    n = model.impl_layers[0].out_conv.weight.shape[1]
    model.impl_layers[0].out_conv.weight = torch.nn.Parameter(torch.eye(n,dtype=x.dtype).unsqueeze(-1).unsqueeze(-1))
    model.impl_layers[0].out_conv.bias = torch.nn.Parameter(torch.zeros_like(model.impl_layers[0].out_conv.bias))
    model.impl_layers[0].out_conv.weight.requires_grad = False
    model.impl_layers[0].out_conv.bias.requires_grad = False

    # second `implicit` layer: here explicit layer with single 3x3 conv ! 

    # fix second out_conv to copy water height from first layer, and u,v from second layer
    # we fix this also for explicit models for fair comparison 
    # (otherwise they have 1 more 3x3 conv for water height)!
    n = model.impl_layers[1].out_conv.weight.shape[1]
    mat = torch.zeros_like(model.impl_layers[1].out_conv.weight)
    mat[1,0] = 1. # ordering of input is: (out [0 channels], out_expl [2 channels], skip connection [5+ channels] 
    mat[2,1] = 1. # so incoming indeces 0,1 take input from out_expl of second layer, 
    mat[0,2] = 1. # and index 2 takes input from first channel of skip connection from first out_conv
    model.impl_layers[1].out_conv.weight = torch.nn.Parameter(mat)
    model.impl_layers[1].out_conv.bias = torch.nn.Parameter(torch.zeros_like(model.impl_layers[1].out_conv.bias))
    model.impl_layers[1].out_conv.weight.requires_grad = False
    model.impl_layers[1].out_conv.bias.requires_grad = False
