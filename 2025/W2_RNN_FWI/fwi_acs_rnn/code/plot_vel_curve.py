import numpy as np
import matplotlib.pyplot as plt
from setup_mar import setup_par



model = setup_par()
dx = model['dx']
dt = model['dt']
nz = model['nz']
depth = np.linspace(0, dx*nz/1000., num=nz)
vpmodel_true = model['vpmodel_true']
print(np.max(vpmodel_true))
vpmodel_init = model['vpmodel_init']
vpmodel_inv_path = model['vpmodel_inv_path']
vpmodel_inv=np.load('./inv/vpmodel_inv_mar_W2_RMSProp.npy', allow_pickle= True)

vpmodel_inv1 = vpmodel_inv[0][1] # #Epoch #Step
vpmodel_inv2 = vpmodel_inv[-1][1] # #Epoch #Step

nx = 200 #the position of velocity
vel_true = vpmodel_true[:,nx]/1000.
vel_init = vpmodel_init[:,nx]/1000.
vel_inv = vpmodel_inv2[:,nx]/1000.


_, ax1 = plt.subplots(figsize=(5, 8))
left=-0.1
top=1.02
ax1.text(left,top,'d)',horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=13)
ax1.set_xlabel("Vp(km/s)",fontsize=14)
ax1.set_ylabel("Depth(km)",fontsize=14)
ax1.plot(vel_true, depth,color='r',label='vpmodel_true',lw=2)
ax1.plot(vel_init, depth,color='black',label='vpmodel_init',linestyle=':',lw=2)
ax1.plot(vel_inv, depth,color='b',label='vpmodel_inv',linestyle='--',lw=2)
plt.legend(loc=1,fontsize=13)
# ax1.xaxis.tick_top()
# ax1.xaxis.set_label_position('top')
ax1.invert_yaxis()
ax1.set_title('RMSProp',fontsize=15)
plt.show()