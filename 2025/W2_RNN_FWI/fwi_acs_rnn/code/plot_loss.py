import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from setup_mar import setup_par


model = setup_par()
dx = model['dx']
dt = model['dt']
num_train_shots = model['num_train_shots']
vpmodel_true = model['vpmodel_true']
vpmodel_init = model['vpmodel_init']
vpmodel_inv_path = model['vpmodel_inv_path']

#loss_adam=np.load(vpmodel_inv_path + 'loss_adam.npy',allow_pickle= True) 
#loss_adam=np.load('loss_Adagrad.npy',allow_pickle= True) 
#loss_adam=np.load('loss_mar_W2_SGDM.npy',allow_pickle= True) 
loss_adam=np.load('loss_mar_W2_adam_lr=10.npy',allow_pickle= True) 
#loss_adam=np.load('loss_sgdM.npy',allow_pickle= True) 
#loss_adam=np.load('loss_sgd.npy',allow_pickle= True) 
#vpmodel_inv_sgd=np.load(vpmodel_inv_path + 'vpmodel_inv_sgd.npy')
#vpmodel_inv_lbfgs=np.load('adam_vs_lbfgs_vpmodel_inv_lbfgs.npy')

#print np.array(loss_adam).shape

batch_size = model['batch_size']

train_loss_adam = np.array(loss_adam[0])[:,1]
train_evals_adam = np.array(loss_adam[0])[:,0] * batch_size  / num_train_shots
dev_loss_adam = np.array(loss_adam[1])[:, 1]
dev_evals_adam = np.array(loss_adam[1])[:, 0] * batch_size / num_train_shots

max_train = np.max(train_loss_adam)
train_loss_adam = train_loss_adam / max_train
max_dev = np.max(dev_loss_adam)
dev_loss_adam = dev_loss_adam / max_dev

epoch_str = 0
#epoch_end = 26
epoch_end = np.array(loss_adam[1])[-1, 0] * batch_size / num_train_shots+1
print ("Total number of epochs:", epoch_end)
epoch_end_idx = epoch_end * num_train_shots / batch_size
epoch_str_idx = epoch_str * num_train_shots / batch_size

## Loss plot
_, ax = plt.subplots(figsize=(5.4, 3.3))
plt.style.use('grayscale')
plt.rc({'font.size': 8})
# training loss
#plt.plot(train_evals_adam[np.int(epoch_str_idx):np.int(epoch_end_idx)], train_loss_adam[np.int(epoch_str_idx):np.int(epoch_end_idx)], '.--', label='Adam_train_loss')

# validation loss
plt.plot(dev_evals_adam[np.int(epoch_str):np.int(epoch_end)],dev_loss_adam[np.int(epoch_str):np.int(epoch_end)], '.-', label='Validation loss')

ax.set_ylim(ymin=0)
ax.set_xlim(xmin=0)
plt.xlabel('Number of epochs')
plt.ylabel('Normalized loss')
plt.legend(loc=7)
#plt.tight_layout(pad=0)
#plt.savefig('adam_vs_lbfgs_loss.eps')

plt.show()


