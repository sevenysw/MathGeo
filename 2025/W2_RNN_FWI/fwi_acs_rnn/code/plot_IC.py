import numpy as np
np.random.seed(19680801)
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from setup_mar import setup_par
from calc_r2 import calc_r2


model = setup_par()
dx = model['dx']
dt = model['dt']
nx = model['nx']
nt = model['nt']
vpmodel_true = model['vpmodel_true']/1000.
#vpmodel_init = model['vpmodel_init']


vpmodel_inv=np.load('./inv/vpmodel_inv_mar_W2_Adam_lr=10.npy',allow_pickle=True)/1000.
vpmodel_inv1 = vpmodel_inv[-1][1]  # #Epoch #Step
vpmodel_inv=np.load('./inv/vpmodel_inv_mar_W2_adam_noise.npy',allow_pickle=True)/1000.
vpmodel_inv2 = vpmodel_inv[-1][1]  # #Epoch #Step
print(vpmodel_inv2.shape)
vpmodel_true = vpmodel_true[4:100,70:270]
vpmodel_inv1 = vpmodel_inv1[4:100,70:270]
vpmodel_inv2 = vpmodel_inv2[4:100,70:270]
# i = 10
# j = 330
# vpmodel_true = vpmodel_true[:,i:j]
# vpmodel_inv1 = vpmodel_inv1[:,i:j]
# vpmodel_inv2 = vpmodel_inv2[:,i:j]

vp_true = vpmodel_true.reshape([-1])
vp_inv = vpmodel_inv1.reshape([-1])
vp_inv2 = vpmodel_inv2.reshape([-1])

vp_true = vp_true[::2]
vp_inv = vp_inv[::2]
vp_inv2 = vp_inv2[::2]


scale = 10

limits_vp = [np.min(vp_true)*0.8, np.max(vp_true)*1.2] # pwave

plt.style.use('grayscale')
plt.rc({'font.size': 8})
aspect = 'equal'
gs1 = gridspec.GridSpec(2, 1, width_ratios=[1],
                        height_ratios=[1, 1])
# ax = []
# ax.append(plt.subplot(gs1[0]))
# ax.append(plt.subplot(gs1[1]))
fig, ax = plt.subplots()
plt.style.use('grayscale')
plt.rc({'font.size': 8})
gs_cnt = 0
left=-0.15
top=1.02
txt_left=0.37+0.2
txt_top = 1.0-0.9

ax.text(left,top,'c)',horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes,fontsize=13)
ax.plot(limits_vp, limits_vp,linewidth=1,color='black')
ax.scatter(vp_true, vp_inv2, c='tab:blue', s=scale, label='with noise',
           alpha=0.5, edgecolors='none')
ax.scatter(vp_true, vp_inv, c='tab:orange', s=scale, label='without noise',
           alpha=0.5, edgecolors='none')
R_2 = calc_r2(vp_true, vp_inv2)
R_2_str=str(float("{:.4f}".format(R_2)))
ax.text(txt_left,txt_top,'${R^2}$='+R_2_str,fontsize=13,horizontalalignment='left',
        verticalalignment='bottom',transform=ax.transAxes,color='tab:blue')
R_2 = calc_r2(vp_true, vp_inv)
R_2_str=str(float("{:.4f}".format(R_2)))
ax.text(txt_left,txt_top-0.1,'${R^2}$='+R_2_str,fontsize=13,horizontalalignment='left',
        verticalalignment='bottom',transform=ax.transAxes,color='tab:orange')
ax.set_aspect(aspect=1)
ax.legend(loc='upper left',fontsize=13)
ax.set_xlabel('True Vp (km/s)',fontsize=15)
ax.set_ylabel('Inverted Vp (km/s)',fontsize=15)
ax.tick_params(labelleft=True, labelright=False)
ax.set_xlim(limits_vp[0], limits_vp[1])
ax.set_ylim(limits_vp[0], limits_vp[1])
ax.yaxis.set_major_locator(MaxNLocator(4))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()
