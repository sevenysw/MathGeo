import numpy as np
from wavelets import ricker
from forward2d import forward2d

def setup_par():
    """Prepare the models and dataset metadata.

    Args: None
    Returns:
        A dictionary containing model-related arrays and properties
    """

    model_path = '../data/marmousi/' 
    vpmodel_true_path = model_path + 'model/mar_130_340.vp'
    vpmodel_inv_path = model_path + 'model/vp_inv.bin'

    seis_path = model_path + '/seis/seis_data'
    seis_inv_path = model_path + '/seis/seis_inv'
    seis_init_path = model_path + '/seis/seis_init'

    source_path = model_path +'/seis/source'

    num_epochs = 2000 # maximum epochs
    n_epochs_stop = 50  # early stopping
    batch_size = 1

    #learning_rate = 200.  #Adagrad_W2_shift=100_batchsise=1
    #learning_rate = 1000.  #Adagrad_W2_shift=100_batchsise=4
    #learning_rate = 25000.  #Adadelta_W2_shift=100_batchsise=1
    #learning_rate = 60000.  #Adadelta_W2_shift=100_batchsise=4

    #learning_rate = 5.  #Adam_L2
    #learning_rate = 1000000.  #SGD_L2_反演浅层

    #learning_rate = 8.  #Adam_NIM_shift=100 scale =10000.
    #learning_rate = 6000000.  #Adam_NIM_shift=100 scale=1无效果
    learning_rate = 10.  #Adam_W2_shift=100
    #learning_rate = 100.  #Adam_W2_shift=300滤波
    #learning_rate = 120.  #SGD_W2_shift=100
    #learning_rate = 120.  #SGDM_W2_shift=100_batchsize=1_momentum=0.8
    #learning_rate = 300.  #SGDM_W2_shift=100_batchsize=1
    #learning_rate = 120.  #SGDM_W2_shift=100_batchsize=1

    #learning_rate = 800.  #SGD_W2_shift=100_mask=10


    #learning_rate = 20.  #SGD_W2_shift=100 scale=10000000000.无效果
    
    
    stage_total = 2


    nx = 340
    nz = 130
    vpmodel_true = (np.fromfile(vpmodel_true_path, dtype=np.float32)
                  .reshape([nx, nz]).T)

    dx = 30
    dt = 0.003
    num_train_shots = 24
    num_vali_shots = 4
    num_sources = num_train_shots + num_vali_shots
    dsrc = 10
    num_receivers = 340
    drec = 1
    nt = 4.5 // dt
    f0 = 7.
    v_bound=[1480.,5000.]

    sources = ricker(f0, nt, dt, 1./f0).reshape([-1, 1, 1])
    sources = np.tile(sources, [1, num_sources, 1])
    sources_x = np.zeros([num_sources, 1, 2], np.int)
    sources_x[:, 0, 0] = 1   #source position
    sources_x[:num_train_shots, 0, 1] = np.linspace(0, nx-1, num_train_shots)
    sources_x[num_train_shots:, 0, 1] = np.linspace(nx/(num_vali_shots+1), nx-nx/(num_vali_shots+1), num_vali_shots)
    receivers_x = np.zeros([1, num_receivers, 2], np.int)
    receivers_x[0, :, 0] = 1  # top receivers position
    receivers_x[0, :, 1] = np.arange(0, num_receivers*drec, drec)
    receivers_x = np.tile(receivers_x, [num_sources, 1, 1])
    propagator = forward2d

   ##  The initial guess model is a strongly smoothed vpmodel_true
    ##vpmodel_init = vpmodel_true[1,1] * np.ones([nz,nx],dtype=np.float32)
    vpmodel_init = np.fromfile(model_path + "model/mar_130_340_init.vp", dtype=np.float32).reshape([nx, nz]).T

    #vpmodel_inv2=np.load('./vpmodel_inv_swi2.npy', allow_pickle= True)
    #vpmodel_init = vpmodel_inv2[-1][1] # #Epoch #Step

    #vpmodel_init = np.fromfile(model_path+"model/mar_vp_sm.bin", dtype=np.float32).reshape([nx, nz]).T


    return {'vpmodel_true': vpmodel_true,
            'vpmodel_init': vpmodel_init,
            'seis_path': seis_path,
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
            'stage_total':stage_total,
            'learning_rate':learning_rate}
