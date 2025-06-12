import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from setup_mar import setup_par
import tensorflow as tf
model = setup_par()
dx = model['dx']
dt = model['dt']
nx = model['nx']
nt = model['nt']
vpmodel_true = model['vpmodel_true']
out_wave = np.load('./seam_wave.npy')
print('outwavefield.shape =',np.shape(out_wave))




snapshot = out_wave[1100,0,:,:] # snap
#snapshot = out_wave[:,:] # model
clip = np.max(np.abs(snapshot))/40.
extent = [0, vpmodel_true.shape[1]*dx/1e3, (vpmodel_true.shape[0])*dx/1e3, 0.]

fig, ax2 = plt.subplots(figsize=(5.4, 4.85))
#im2=ax2.imshow(np.transpose(receivers[:,:,1]),extent=[0,12,90*dx/1000.,0],vmin=-clip,vmax=clip,cmap='bwr')
im2=ax2.imshow(np.transpose(snapshot),extent=extent,vmin=-clip,vmax=clip,cmap='bwr')
# divider = make_axes_locatable(ax2)
# cax2 = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im2,ax=ax2,cax=cax2).set_label('Relative amplitude')
ax2.set_xlabel('Position (km)')
ax2.set_ylabel('Depth (s)')
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')

plt.show()
