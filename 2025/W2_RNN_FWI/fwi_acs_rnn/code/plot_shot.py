import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import butter, lfilter

from setup_mar import setup_par


def butter_bandpass(lowcut, highcut, dt, order=5):
    fs = 1./dt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, dt, order=5):
    b, a = butter_bandpass(lowcut, highcut, dt, order=order)
    y = lfilter(b, a, data)
    return y

model = setup_par()
dx = model['dx']
dt = model['dt']
nx = model['nx']
nt = model['nt']
seis_path = model['seis_path'] + '.npy'
#seis_path =  'receiverdata_inv_adam.npy'
#seis_path =  'adj_source.npy'
receivers = np.load(seis_path)
print(receivers.shape)

shot = receivers[:,13,:].T
clip = np.max(np.abs(shot))/20
print(np.min(shot),np.max(shot))


# bandpass filter; can be implemented later
#for ishot in range(shot.shape[0]):
#    shot[ishot,:] = butter_bandpass_filter(shot[ishot,:], 0.1, 1., dt, order=3)

fig, ax2 = plt.subplots()
im2=ax2.imshow(np.transpose(shot),extent=[0, nx*dx/1000.,nt*dt,0],vmin=-clip,vmax=clip,cmap='bwr')
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2,ax=ax2,cax=cax2).set_label('Relative amplitude')
ax2.set_xlabel('Position (km)')
ax2.set_ylabel('Time (s)')
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')

plt.show()


