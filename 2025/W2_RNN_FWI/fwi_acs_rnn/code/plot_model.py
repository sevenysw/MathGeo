import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from setup_mar import setup_par


model = setup_par()
dx = model['dx']
dt = model['dt']
vpmodel_true = model['vpmodel_true']
vpmodel_init = model['vpmodel_init']
vpmodel_inv_path = model['vpmodel_inv_path']

#vpmodel_inv=np.load(vpmodel_inv_path + 'vpmodel_inv_adam.npy', allow_pickle= True) 
#vpmodel_inv=np.load('vpmodel_inv_mar_W2_Adagrad.npy', allow_pickle= True) 
vpmodel_inv=np.load('vpmodel_inv_mar_W2_adam_lr=10.npy', allow_pickle= True) 
#vpmodel_inv=np.load('vpmodel_inv_adam_L2.npy', allow_pickle= True) 
#vpmodel_inv=np.load('vpmodel_inv_sgdM.npy', allow_pickle= True) 
#vpmodel_inv=np.load('vpmodel_inv_sgd.npy', allow_pickle= True) 
vpmodel_inv1 = vpmodel_inv[0][1] # #Epoch #Step
#vpmodel_inv2=np.load(vpmodel_inv_path + 'vpmodel_inv2.npy')
vpmodel_inv2 = vpmodel_inv[-1][1] # #Epoch #Step


# Model plots
fig = plt.figure(figsize=(5.4, 4.85))
plt.style.use('grayscale')
plt.rc({'font.size': 8})
extent = [0, vpmodel_true.shape[1]*dx/1e3, vpmodel_true.shape[0]*dx/1e3, 0]
aspect = 'equal'
#vmin = 1490/1e3
#vmax = 4480/1e3
vmin = np.amin(vpmodel_true)/1e3
vmax = np.amax(vpmodel_true)/1e3
gs1 = gridspec.GridSpec(3, 2, width_ratios=[1, 1],
                        height_ratios=[1, 1, 0.075])
ax = []
ax.append(plt.subplot(gs1[0]))
ax.append(plt.subplot(gs1[1]))
ax.append(plt.subplot(gs1[2]))
ax.append(plt.subplot(gs1[3]))
im = ax[0].imshow(vpmodel_true/1e3, extent=extent, aspect=aspect,cmap='jet',
                  vmin=vmin, vmax=vmax)
ax[0].set_title('True')
ax[0].set_ylabel('Depth (km)')
plt.setp(ax[0].get_xticklabels(), visible=False)
ax[1].imshow(vpmodel_init/1e3, extent=extent, aspect=aspect,cmap='jet',
             vmin=vmin, vmax=vmax)
ax[1].set_title('Initial')
plt.setp(ax[1].get_xticklabels(), visible=False)
plt.setp(ax[1].get_yticklabels(), visible=False)
ax[2].imshow(vpmodel_inv1/1e3, extent=extent, aspect=aspect,cmap='jet',
             vmin=vmin, vmax=vmax)
ax[2].set_title('First iteration')
ax[2].set_xlabel('x (km)')
ax[2].set_ylabel('Depth (km)')
ax[3].imshow(vpmodel_inv2/1e3, extent=extent, aspect=aspect,cmap='jet',
             vmin=vmin, vmax=vmax)
#ax[3].imshow((vpmodel_inv2-vpmodel_init)/1e3, extent=extent, aspect=aspect)
ax[3].set_title('Final iteration')
plt.setp(ax[3].get_yticklabels(), visible=False)
ax[3].set_xlabel('x (km)')
cax = plt.subplot(gs1[-2:])
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Wave speed (km/s)')
plt.tight_layout(pad=0.1, h_pad=0.6, w_pad=0.5, rect=[0, 0, 1, 0.98])
plt.show()


