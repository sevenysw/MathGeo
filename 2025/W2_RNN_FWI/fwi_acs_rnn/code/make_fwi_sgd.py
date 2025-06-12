import sys
import csv
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
from fwi import (Fwi, shuffle_shots, extract_datasets, _get_dev_loss)
from setup_mar import setup_par

def make_fwi_sgd():
    np.random.seed(0)

   # Load the initial model, datasets, and other relevant values.
    model = setup_par()
    vpmodel_true = model['vpmodel_true']
    vpmodel_init = model['vpmodel_init']
    model_path = model['model_path']
    dx = model['dx']
    dt = model['dt']
    v_bound = model['v_bound']
    sources = model['sources']
    sources_x = model['sources_x']
    receivers_x = model['receivers_x']
    propagator = model['propagator']
    num_train_shots = model['num_train_shots']

    seis_path = model['seis_path'] + '.npy'
    receivers = np.load(seis_path)

    vpmodel_inv_path = model['vpmodel_inv_path']

    num_epochs = model['num_epochs']
    n_epochs_stop = model['n_epochs_stop']
    batch_size = model['batch_size']
    learning_rate = model['learning_rate']


    dataset = sources, sources_x, receivers, receivers_x

    dataset = shuffle_shots(dataset, num_train_shots)
    train_dataset, dev_dataset = extract_datasets(dataset, num_train_shots)

    # Starting training
    model_inv_sgd, loss_sgd = _fwi_launcher(vpmodel_init,
                                      dx, dt,
                                      num_epochs, n_epochs_stop, batch_size,
                                      learning_rate, v_bound,
                                      train_dataset,
                                      dev_dataset,
                                      propagator)
    # Saving results
    np.save(vpmodel_inv_path + 'vpmodel_inv_sgd', model_inv_sgd)
    np.save(vpmodel_inv_path + 'loss_sgd', loss_sgd)





def _fwi_launcher(vpmodel_init, dx, dt, 
                 num_epochs, n_epochs_stop,
                 batch_size, learning_rate, v_bound,
                 train_dataset, dev_dataset,
                 propagator):
    """Run optimization using the Adam optimizer (with the best hyperparameters
    found from make_hyperparameter_selection_figure) and L-BFGS-B (from
    Scipy).

    Returns:
        model_inv_adam: The final Adam model
        model_inv_sgd: The final SGD model
        model_inv_lbfgs: The final L-BFGS-B model
        adam_evals: A list containing the number of shots that have been
                    evaluated (forward and backward propagated to calculate
                    gradient) corresponding to dev_loss_adam
        dev_loss_adam: A list containing the cost function value after the
                       number of shot evaluations in the corresponding list
                       entry in adam_evals, when using Adam
        sgd_evals: Same as adam_evals, but for SGD
        dev_loss_sgd: Same as dev_loss_adam, but for SGD
        lbfgs_evals: Same as adam_evals, but for L-BFGS-B
        dev_loss_lbfgs: Same as dev_loss_adam, but for L-BFGS-B
    """
    
    num_train_shots = train_dataset[0].shape[1]

    print ("Starting SGD test....")
    # Stochastic Gradient Descent
    assert num_train_shots % batch_size == 0
    tf.compat.v1.reset_default_graph()
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)  #good for seam
    #optimizer = tf.compat.v1.train.GradientDescentOptimizer(733.35)  

    num_batches_in_one_epoch = num_train_shots // batch_size
    num_steps = num_epochs * num_batches_in_one_epoch
    fwi = Fwi(vpmodel_init, dx, dt, train_dataset, dev_dataset, propagator,
              optimizer=optimizer,v_bound=v_bound,
              batch_size=batch_size)
    model_inv_sgd, loss_sgd = fwi.train(num_steps, num_batches_in_one_epoch, num_batches_in_one_epoch, n_epochs_stop)

    return (model_inv_sgd, loss_sgd)

if __name__ == '__main__':
    make_fwi_sgd()
