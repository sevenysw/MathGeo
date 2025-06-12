import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar


def CourantCondition(dx,num_dims,Vmax):
    """Courant–Friedrichs–Lewy stability condition. Find the maximum stable
    time step allowed by the grid cell size and maximum velocity."""
    return np.min(dx)/(num_dims**.5*Vmax)


def DataLoad_Train(device):

    # Set default dtype to float32
    torch.set_default_dtype(torch.float)

    # PyTorch random number generator
    torch.manual_seed(1234)

    # Random number generators in other libraries
    np.random.seed(1234)

    
    # device = torch.device('cpu')
    ny = 450
    nx = 120
    dx = 30
    v_true = torch.from_file('/home/wanghan/test_BP/portion/04BP_120_450.bin',size=ny*nx).reshape(ny, nx)

    v_true= v_true[:250,:].to(device) 
    vmax, vmin = torch.max(v_true.detach()), torch.min(v_true.detach())
    model_dim = v_true.shape
    print('v_true.shape', v_true.shape)
    plt.figure(figsize=(10.5, 3.5))
    plt.imshow(v_true.detach().cpu().T, aspect='auto', cmap='RdBu_r',vmin=vmin, vmax=vmax)
    plt.savefig('example_v.png')
    print('vmin, vmax', vmin, vmax)


    n_shots = 25

    n_sources_per_shot = 1
    d_source = 10  # 20 * 4m = 80m
    first_source = 0  # 10 * 4m = 40m
    source_depth = 1  # 2 * 4m = 8m

    n_receivers_per_shot = 250
    d_receiver = 1  # 6 * 4m = 24m
    first_receiver = 0  # 0 * 4m = 0m
    receiver_depth = 1  # 2 * 4m = 8m

    freq = 6
    nt = 1500
    dt = 0.004
    peak_time = 1.1 / freq

    dtmax = CourantCondition(dx, 2, 500.0)
    print("Grid size:", dx)
    print("Time step, number of time samples", dt, nt)
    print("Stability condition on the time step dt:", dt, "<", dtmax)

    # source_locations, [shot, source, space]
    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                   dtype=torch.long, device=device)
    source_locations[..., 1] = source_depth
    source_locations[:, 0, 0] = torch.arange(n_shots) * d_source + first_source

    # receiver_locations [shot, receiver, space]
    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                     dtype=torch.long, device=device)
    receiver_locations[..., 1] = receiver_depth
    receiver_locations[:, :, 0] = (
        (torch.arange(n_receivers_per_shot) * d_receiver + first_receiver)
            .repeat(n_shots, 1)
    )

    # source_amplitudes [shot, source, time]
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
            .repeat(n_shots, n_sources_per_shot, 1)
            .to(device)
    )

    # Propagate [num_shots, n_r, n_t]
    observed_data = scalar(v_true, dx, dt, source_amplitudes=source_amplitudes,
                           source_locations=source_locations,
                           receiver_locations=receiver_locations,
                           accuracy=8,
                           pml_freq=freq)[-1]

    print('observed_data.shape', observed_data.shape)

    # Propagate [num_shots, n_r, n_t] observed_data

    data_dim = observed_data[0].shape
    omax, omin = torch.max(observed_data.detach()), torch.min(observed_data.detach())

    
    # train_set_norm = (observed_data / (observed_data ** 2).mean().sqrt().item()).reshape(n_shots, 1, data_dim[0] * data_dim[1])

    # print(torch.sum(observed_data, dim=1)/data_dim[1])
    # (torch.sum(observed_data, dim=1)/data_dim[1])

    # for index in range(n_shots):
    #
    #     plt.figure(figsize=(4.5, 4.5))
    #     plt.plot(np.arange(data_dim[1]),
    #              (torch.sum(torch.abs(observed_data[index]), dim=0)/data_dim[0]).detach().cpu())
    #     plt.savefig(str(index)+'_data_comparison.png')
    #     plt.show()

  #  observed_data /= (observed_data ** 2).mean().sqrt().item()

    return observed_data, data_dim, model_dim, source_amplitudes, source_locations, receiver_locations, v_true, vmin,vmax,dx,dt,freq,n_shots,omax, omin

