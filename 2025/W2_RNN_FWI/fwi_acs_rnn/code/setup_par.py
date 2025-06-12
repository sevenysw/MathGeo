import numpy as np
from wavelets import ricker
from forward2d import forward2d

def setup_par():
    """Prepare the models and dataset metadata.

    Args: None
    Returns:
        A dictionary containing model-related arrays and properties
    """

    model_path = '../data/camembert' 
    seis_path = model_path + '/seis/cam_data'
    seis2_path = model_path + '/seis/cam_outreceiver_data'
    vpmodel_inv_path = model_path + '/model_inv/'
    source_path = model_path +'/seis/cam_source'
    

    num_epochs = 5 # maximum epochs
    n_epochs_stop = 3  # early stopping
    batch_size = 2

    #learning_rate = 1000000.  # adam_NIM_shift=100 working
    #learning_rate = 1000.  # adam_NIM_shift=3 working


    #learning_rate = 10.  # adam_W2_shift=100 working
    #learning_rate = 4.  # adam_W2_shift=100 working
    #learning_rate = 100.  # adam_W2_shift=3 working
    #learning_rate = 10.  # adam_W2_shift=130 working
    #learning_rate = 100000.  # SGD_W2_shift=100 working
    #learning_rate = 60000.  # SGD_W2_shift=100_mygradient1 working
    #learning_rate = 100000.  # SGD_W2_shift=100_mygradient2 working
    #learning_rate = 10.  # Adam_W2_shift=100_mygradient2 working
    learning_rate = 10.  # Adam_W2_shift=3_mygradient2 working


    #learning_rate = 10.  #Adam_L2  working
    #learning_rate = 10000000000000.  #SGD_NIM_shift=100 working
    #learning_rate = 100000000.  #sgd_ L2/NIM working,epoch为40以上


    nx = 201
    nz = 201
    #vpmodel_true = (np.fromfile(vpmodel_true_file, dtype=np.float32)
    #              .reshape([nx, nz]).T)


    dx = 10
    dt = 0.001
    num_train_shots = 12
    num_vali_shots = 2
    num_sources = num_train_shots + num_vali_shots
    dsrc = 10
    num_receivers = 201
    drec = 1
    nt = 1.2 // dt 
    f0 = 10.
    v_bound=[2800.,4000.]

    sources = ricker(f0, nt, dt, 1./f0).reshape([-1, 1, 1])
    sources = np.tile(sources, [1, num_sources, 1])
    sources_x = np.zeros([num_sources, 1, 2], np.int)
    sources_x[:, 0, 0] = 5
    sources_x[:num_train_shots, 0, 1] = np.linspace(0, nx-1, num_train_shots)
    sources_x[num_train_shots:, 0, 1] = np.round(np.linspace(nx/(num_vali_shots+1), nx-nx/(num_vali_shots+1), num_vali_shots))
    receivers_x = np.zeros([1, num_receivers, 2], np.int)
#    receivers_x[0, :, 0] = 0  # top
    receivers_x[0, :, 0] = nz-1 # bottom
    receivers_x[0, :, 1] = np.arange(0, num_receivers*drec, drec)
    receivers_x = np.tile(receivers_x, [num_sources, 1, 1])
    propagator = forward2d

    # define true velocity model
    vp = 3000.
    vpmodel_true = vp * np.ones([nz,nx],dtype=np.float32)
    cent1_x = 100
    cent1_z = 100
    for ix in range(nx):
       for iz in range(nz):
           if np.sqrt(np.square(ix-cent1_x)+np.square(iz-cent1_z)) < 60:
               vpmodel_true[iz,ix] = vpmodel_true[iz,ix]*1.2

    # Initial model: Homo
    vpmodel_init = vp  * np.ones([nz,nx],dtype=np.float32)

    # Initial model: Previous inversion
   # vpmodel_inv2=np.load('./vpmodel_inv_adam2.npy', allow_pickle= True)
   # vpmodel_init = vpmodel_inv2[-1][1] # #Epoch #Step


    return {'vpmodel_true': vpmodel_true,
            'vpmodel_init': vpmodel_init,
            'seis_path': seis_path,
            'seis2_path': seis2_path,
            'source_path': source_path,
            'vpmodel_inv_path': vpmodel_inv_path,
            'dx': dx,
            'dt': dt,
            'nx': nx,
            'nz': nz,
            'nt': nt,
            'v_bound': v_bound,
            'model_path': model_path,
            'num_train_shots': num_train_shots,
            'sources': sources,
            'sources_x': sources_x,
            'receivers_x': receivers_x,
            'propagator': propagator,
            'num_epochs':num_epochs,
            'n_epochs_stop':n_epochs_stop,
            'batch_size':batch_size,
            'learning_rate':learning_rate}
