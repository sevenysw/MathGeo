"""FWI implemented using TensorFlow
"""
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import time
import scipy.optimize
from tensorflow.core.protobuf import rewriter_config_pb2
from NIM_loss import NIM
from W2_loss import W2

#from tensorflow.core.protobuf import rewriter_config_pb2
#from tensorflow.compat.v1.keras.backend import set_session


class Fwi(object):
    """Optimize model to fit data.

    Args:
        model: Numpy array initial velocity model
        dx: Float specifying grid cell spacing
        dt: Float specifying time between data samples
        train_dataset: List containing
            sources: Numpy array of source amplitudes,
                [num_time_steps, num_shots, num_sources_per_shot]
            sources_x: Numpy integer array of source cell positions,
                [num_shots, num_sources_per_shot, coords]
            receivers: Numpy array containing recorded data,
                [num_time_steps, num_shots, num_receivers_per_shot]
            receivers_x: Numpy integer array of receiver cell positions,
                [num_shots, num_receivers_per_shot, coords]
        dev_dataset: Same as train_dataset, but for dev data (may be empty)
        propagator: Propagator function to use
        optimizer: Optimization method to use (tf.train.Optimizer object)
        batch_size: Integer specifying number of shots to use per minibatch
        l2_regularizer_scale: Scale factor of L2 regularization on model
                              parameters (0.0 == no regularization)
        autodiff: Boolean specifying whether to use automatic differentiation
                  (True) or the adjoint-state method (False)
        save_gradient: Boolean specifying whether the gradient should be
                       stored in the gradient attribute when using automatic
                       differentation.
    """

    def __init__(self, model, dx, dt, train_dataset, dev_dataset, propagator,
                 optimizer=tf.compat.v1.train.GradientDescentOptimizer(1e7),
                 v_bound=[1400.,4100.],batch_size=1,
                 l2_regularizer_scale=0.0,
                 autodiff=True, save_gradient=False):

        ndim = model.ndim
        train_sources = train_dataset[0]
        train_receivers = train_dataset[2]
        num_time_steps = int(train_sources.shape[0])
        num_sources_per_shot = int(train_sources.shape[2])
        num_receivers_per_shot = int(train_receivers.shape[2])

        regularizer = \
                tf.keras.regularizers.l2(l=0.5 * (l2_regularizer_scale))
#        regularizer = tf.keras.regularizers.l2(l2_regularizer_scale)

        #mask the part of water
        shallow_vp = tf.constant(model[:4,:])
        model1 = tf.compat.v1.get_variable('model',initializer=tf.constant(model[4:,:]),regularizer=regularizer)
        model = tf.concat([shallow_vp,model1],0)
        
        
        #model = tf.compat.v1.get_variable('model', initializer=tf.constant(model),regularizer=regularizer)
        #model = tf.clip_by_value(model, clip_value_min=v_bound[0], clip_value_max=v_bound[1]) # need to be commented when using adjoint for gradient

        batch_placeholders = _create_batch_placeholders(ndim,
                                                        num_time_steps,
                                                        num_sources_per_shot,
                                                        num_receivers_per_shot)

        out_wavefields = propagator(model, batch_placeholders['sources'],
                                    batch_placeholders['sources_x'], dx, dt)

        out_receivers = _extract_receivers(out_wavefields,
                                           batch_placeholders['receivers_x'],
                                           num_receivers_per_shot)

        reg_ws = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)

        regularization_loss = tf.reduce_sum(input_tensor=reg_ws)
        #loss = (tf.compat.v1.losses.mean_squared_error(batch_placeholders['receivers'], out_receivers) + regularization_loss)
        #shift = tf.math.reduce_min(batch_placeholders['receivers'])*1.1
        #loss,residual_batch = W2(batch_placeholders['receivers'], out_receivers, num_time_steps, batch_size, num_receivers_per_shot, dt)
        loss = W2(batch_placeholders['receivers'], out_receivers, num_time_steps, batch_size, num_receivers_per_shot, dt)
        #loss = NIM(batch_placeholders['receivers'], out_receivers)

        if autodiff:
            if save_gradient:
                gradient = optimizer.compute_gradients(loss)
            else:
                gradient = None
            train_op = optimizer.minimize(loss)
        else:
            gradient = (_adjoint(model, dx, dt, out_wavefields, out_receivers,
                                 batch_placeholders['receivers'],
                                 batch_placeholders['receivers_as_sources_x'],
                                 propagator,num_time_steps,batch_size, num_receivers_per_shot)
                        + l2_regularizer_scale * model)
            train_op = optimizer.apply_gradients([(gradient, model)])

        # to cope with large Graphs
        config = tf.compat.v1.ConfigProto()
        off = rewriter_config_pb2.RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization  = off
        sess = tf.compat.v1.Session(config = config) # for large models

#        sess = tf.compat.v1.Session() # original
        sess.run(tf.compat.v1.global_variables_initializer())

        self.sess = sess
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model
        self.batch_size = batch_size
        self.batch_placeholders = batch_placeholders
        self.loss = loss
        self.train_op = train_op
        self.gradient = gradient

    def train(self, num_steps, num_batches_in_one_epoch, save_model_interval, n_epochs_stop):
        """Train (invert/optimize) the model.

        Args:
            num_steps: The number of minibatches to train
            num_batches_in_one_epoch: Number of steps between printing loss (cost
                            function value) of training and development
                            datasets
            save_model_interval: Number of steps between saveing the vpmodel

        Returns:
            model: The trained model
            loss: A list containing the training and development losses
        """
        loss, vpmodel_inv = _train_loop(num_steps, self.train_dataset, self.dev_dataset,
                           self.model.shape, self.batch_size,
                           self.batch_placeholders, self.loss,
                           self.train_op, self.sess, num_batches_in_one_epoch,
                           save_model_interval, self.model, n_epochs_stop)

        #return self.sess.run(self.model), loss
        vpmodel_inv.append((num_steps, self.sess.run(self.model)))
        return vpmodel_inv, loss

    def train_lbfgs(self, options, loss_file=None):
        """Train the model using the L-BFGS-B optimizer.

        I constrain the inverted model range to be between 1490 and 5000 m/s to
        avoid possibly using values that may cause the finite difference
        wave propagator to become unstable.

        Args:
            options: Dictionary containing options for Scipy's L-BFGS-B
                     optimizer
            loss_file: File to write training and development losses to

        Returns:
            The trained model
        """
        if loss_file is not None:
            loss_fh = open(loss_file, 'w')
        init_model = self.sess.run(self.model)
        res = scipy.optimize.minimize(_entire_dataset_loss_and_gradient,
                                      init_model.ravel(),
                                      args=(self.model, self.train_dataset,
                                            self.dev_dataset,
                                            self.batch_size,
                                            self.batch_placeholders,
                                            self.gradient, self.loss,
                                            self.sess, init_model.shape,
                                            loss_fh),
                                      method='L-BFGS-B',
                                      jac=True,
                                      bounds=[(1490, 5000)] \
                                              * len(init_model.ravel()),
                                      options=options)
        if loss_file is not None:
            loss_fh.close()
        return res


def shuffle_shots(dataset,num_train_shots):
    """Shuffle shots so each batch contains a more representative sample of the
    whole dataset.

    Args:
        dataset: A tuple containing sources, sources_x, receivers, receivers_x

    Returns:
        The shuffled dataset
    """
    sources, sources_x, receivers, receivers_x = dataset
    num_shots = int(sources.shape[1])
    orig_idxs = np.arange(num_shots)
    shuffled_idxs = np.arange(num_train_shots)
    np.random.shuffle(shuffled_idxs)
    shuffled_idxs = np.concatenate((shuffled_idxs,orig_idxs[num_train_shots:]))


    return (sources[:, shuffled_idxs, :], sources_x[shuffled_idxs, :, :],
            receivers[:, shuffled_idxs, :], receivers_x[shuffled_idxs, :, :])



def extract_datasets(dataset, num_train_shots):
    """Split the dataset into a training dataset and a development dataset.

    Args:
        dataset: A tuple containing sources, sources_x, receivers, receivers_x
        num_train_shots: An integer specifying the number of shots to use
                         in the training dataset (must be <= the number of
                         shots in the dataset)

    Returns:
        The training dataset and the development dataset
    """
    sources, sources_x, receivers, receivers_x = dataset
    train_sources = sources[:, :num_train_shots, :]
    train_sources_x = sources_x[:num_train_shots, :, :]
    train_receivers = receivers[:, :num_train_shots, :]
    train_receivers_x = receivers_x[:num_train_shots, :, :]

    dev_sources = sources[:, num_train_shots:, :]
    dev_sources_x = sources_x[num_train_shots:, :, :]
    dev_receivers = receivers[:, num_train_shots:, :]
    dev_receivers_x = receivers_x[num_train_shots:, :, :]

    return ((train_sources, train_sources_x,
             train_receivers, train_receivers_x),
            (dev_sources, dev_sources_x,
             dev_receivers, dev_receivers_x))


def _create_batch_placeholders(ndim, num_time_steps,
                               num_sources_per_shot,
                               num_receivers_per_shot):
    """Create TensorFlow placeholders that will store minibatches.

    Args:
        ndim: An integer specifying the dimensionality of the model
        num_time_steps: An integer specifying the number of time steps in
                        each shot
        num_sources_per_shot: An integer specifying the number of sources
                              in each shot
        num_receivers_per_shot: An integer specifying the number of receivers
                                in each shot

    Returns:
        A dictionary of placeholders
    """
    batch_sources = tf.compat.v1.placeholder(tf.float32,
                                   [num_time_steps, None,
                                    num_sources_per_shot])
    batch_sources_x = tf.compat.v1.placeholder(tf.int32,
                                     [None,
                                      num_sources_per_shot, ndim + 1])
    batch_receivers = tf.compat.v1.placeholder(tf.float32,
                                     [num_time_steps, None,
                                      num_receivers_per_shot])
    batch_receivers_x = tf.compat.v1.placeholder(tf.int32,
                                       [None, num_receivers_per_shot])
    batch_receivers_as_sources_x = \
            tf.compat.v1.placeholder(tf.int32, [None, num_receivers_per_shot,
                                      ndim + 1])

    return {'sources': batch_sources,
            'sources_x': batch_sources_x,
            'receivers': batch_receivers,
            'receivers_x': batch_receivers_x,
            'receivers_as_sources_x': batch_receivers_as_sources_x}


def _extract_receivers(out_wavefields, batch_receivers_x,
                       num_receivers_per_shot):
    """Extract the receiver data from the wavefields.

    Because of the way tf.gather works, this is slightly complicated,
    requiring that I reshape out_wavefields into a 2D array.

    Args:
        out_wavefields: A Tensor containing wavefields. The first dimension
                        is time, the second dimension is shot index with the
                        minibatch, and the remaining dimensions are the
                        model domain (depth and x for a 2D model)
        batch_receivers_x: A 2D Tensor [batch_size, num_receivers_per_shot]
                           containing receiver coordinates indexing into
                           out_wavefield after it has been flattened into 2D
        num_receivers_per_shot: An integer specifying the number of receivers
                                per shot

    Returns:
        A 3D Tensor [num_time_steps, batch_size, num_receivers_per_shot]
        containing the extracted receiver data
    """
    batch_receivers_x = tf.reshape(batch_receivers_x, [-1])
    num_time_steps = out_wavefields.shape[0]
    out_wavefields = tf.reshape(out_wavefields, [num_time_steps, -1])
    out_receivers = tf.gather(out_wavefields, batch_receivers_x, axis=1)
    out_receivers = tf.reshape(out_receivers, [num_time_steps, -1,
                                               num_receivers_per_shot])

    return out_receivers


def _train_loop(num_steps, train_dataset, dev_dataset, model_shape, batch_size,
                batch_placeholders, loss, train_op, sess, num_batches_in_one_epoch, 
                save_model_interval, model, n_epochs_stop):
    """The main training loop: train using num_steps minibatches.

    Args:
        num_steps: Integer specifying the number of minibatches to train with
        train_dataset: A tuple containing the training dataset
        dev_dataset: A tuple containing the development dataset
        model_shape: A tuple containing the shape of the model
        batch_size: An integer specifying the number of shots in a minibatch
        batch_placeholders: The placeholders created using
                            _create_batch_placeholders
        loss: The Tensor that gives the cost function value when evaluated
        train_op: The operation that performs one optimization step when
                  evaluated
        sess: A TensorFlow session object
        num_batches_in_one_epoch: Number of steps between printing loss (cost
                        function value) of training and development
                        datasets
        save_model_interval: Number of steps between saveing the vpmodel

    Returns:
        Lists containing the training and development losses. The training
        loss is recorded after each minibatch, but the development loss
        is only recorded every num_batches_in_one_epoch minibatches, and so the
        minibatch index (step) is also recorded with the development loss
    """
    train_losses = []
    dev_losses = []
    vpmodel_inv = []
    tic = time.time()

    for step in range(num_steps):
        epoch_idx = step // num_batches_in_one_epoch
        train_l = _train_step(step, train_dataset, model_shape, batch_size,
                              batch_placeholders, loss, train_op, sess)
        train_losses.append((step, train_l))

        if step % num_batches_in_one_epoch == 0:
            toc = time.time()
            print ("Time for this epoch:",  (toc - tic)/60, "minutes")
            dev_l = _get_dev_loss(model_shape, dev_dataset, batch_placeholders,
                                  loss, sess)
            ## debug
           # vpmodel_inv.append((epoch_idx, sess.run(model)))
           # np.save('vpmodel_inv_adam', vpmodel_inv)
            # early stopping
            if epoch_idx == 0:
               min_val_loss = dev_l
               epochs_no_improve = 0
            if dev_l < min_val_loss:  # still decreasing
               epochs_no_improve = 0
               min_val_loss = dev_l
               dev_losses.append((step, dev_l))
               np.save('loss_mar_W2_adam', (train_losses, dev_losses))

               vpmodel_inv.append((epoch_idx, sess.run(model)))
               print(step, train_l, dev_l, epoch_idx)
               print('Saving the inverted vpmodel')
               np.save('vpmodel_inv_mar_W2_adam', vpmodel_inv)
            else:
               epochs_no_improve += 1
               print (dev_l)
               #dev_losses.append((step, dev_l))
               #np.save('loss_adam_L2', (train_losses, dev_losses))
               #vpmodel_inv.append((epoch_idx, sess.run(model)))
               #np.save('vpmodel_inv_Adagrad', vpmodel_inv)
               if epochs_no_improve == n_epochs_stop:
                   print ('Minimal Loss: %.10f'%(dev_l))
                   raise ValueError('Early stopping!')
                    
            tic = time.time()


    return (train_losses, dev_losses), vpmodel_inv


def _get_dev_loss(model_shape, dev_dataset, batch_placeholders, loss, sess,
                  model=None, test_model=None):
    """Compute the development loss (cost function value on the development
    dataset).

    Args:
        model_shape: Tuple containing the shape of the model
        dev_dataset: Tuple containing the development dataset
        batch_placeholders: The placeholders created using
                            _create_batch_placeholders
        loss: The Tensor that gives the cost function value when evaluated
        sess: A TensorFlow session object
        model: The model tensor used in the FWI graph (so that we can
               temporarily replace it with test_model, if desired; optional)
        test_model: Array containing the model to compute the development
                    loss on (optional). If not specified, the current value
                    of the model in the session is used.

    Returns:
        A float containing the development loss
    """
    if dev_dataset is None:
        return
    num_dev_shots = dev_dataset[0].shape[1]
    dev_batch_size = 1 # Must be a divisor of the dev dataset size
    num_dev_batches = num_dev_shots // dev_batch_size
    dev_l = []
    # Loop over minibatches in the development dataset and compute the
    # mean loss over them. This allows the development dataset to be
    # larger than can be processed in one minibatch.
    for dev_step in range(num_dev_batches):
        dev_l.append(_test_dev_step(dev_step, dev_dataset, model_shape,
                                    dev_batch_size, batch_placeholders,
                                    loss, sess, model=model,
                                    test_model=test_model))
    dev_l = np.mean(dev_l) / dev_batch_size
    return dev_l


def _prepare_batch(step, dataset, model_shape, batch_size, batch_placeholders):
    """Create a minibatch of data and assign it to the placeholders.

    Args:
        step: Integer specifying the training step
        dataset: Tuple containing the dataset
        model_shape: Tuple specifying the shape of the model
        batch_size: Integer specifying the number of shots in a minibatch
        batch_placeholders: The placeholders created using
                            _create_batch_placeholders

    Returns:
        A feed dictionary appropriate for passing to sess.run, assigning
        minibatches of data to the batch placeholders
    """

    sources, sources_x, receivers, receivers_x = dataset

    ndim = len(model_shape)
    num_sources_per_shot = int(sources.shape[2])
    num_receivers_per_shot = int(receivers.shape[2])
    num_shots = int(sources.shape[1])

    if num_shots > batch_size:
        batch_start = (step * batch_size) % (num_shots - batch_size + 1)
    else:
        batch_start = 0
    batch_end = batch_start + batch_size

    src_batch = sources[:, batch_start : batch_end, :]

    # Augment sources_x with an additional coordinate (stored in [:, :, 0])
    # specifying the shot index within the minibatch
    src_x_batch = np.zeros([batch_size, num_sources_per_shot, ndim + 1],
                           np.int)
    src_x_batch[:, :, 1:] = sources_x[batch_start : batch_end, :, :]
    src_x_batch[:, :, 0] = \
            np.tile(np.arange(batch_size).reshape([-1, 1]),
                    [1, num_sources_per_shot])

    rec_batch = receivers[:, batch_start : batch_end, :]

    # Because of the way tf.gather works (which is used to extract the
    # receiver data from the wavefields), I flatten the wavefields from
    # [num_time_steps, batch_size, :] (where : represents the model space) into
    # [num_time_steps, batch_size * :], so the receiver coordinates need
    # to index into this flattened array. The gather will extract slices in
    # the time dimension, so we only need to index into the other dimension
    # of the array. In the case of a 2D model, of shape nz, nx, if the
    # coordinates in receivers_x[batch_idx, receiver_idx, :] are rz, rx, then
    # the resulting coordinates to index into the flattened array should be:
    # batch_idx * nz * nx + rz * nx + rx. I do this in several steps.
    rec_x_batch = np.zeros([batch_size, num_receivers_per_shot], np.int)
    # For the 2D model described, this loop adds the rz * nx term
    for dim in range(ndim-1):
        rec_x_batch += (receivers_x[batch_start : batch_end, :, dim]
                        * np.prod(np.array(model_shape, np.int)[dim+1:]))
    # The next line adds the rx term
    rec_x_batch += receivers_x[batch_start : batch_end, :, -1]
    # This adds the batch_idx * nz * nx term
    rec_x_batch += \
            (np.tile(np.reshape(np.arange(batch_size), [batch_size, 1]),
                     [1, num_receivers_per_shot])
             * np.prod(np.array(model_shape, np.int)))

    # If using the adjoint-state method to calculate the gradient, we treat
    # the receivers as sources during backpropagation, so we also need to
    # store the receiver coordinates in the form used for sources: augment
    # with an additional coordinate (stored in [:, :, 0])
    # specifying the shot index within the minibatch
    rec_as_src_x_batch = np.zeros([batch_size, num_receivers_per_shot,
                                   ndim + 1], np.int)
    rec_as_src_x_batch[:, :, 1:] = \
            receivers_x[batch_start : batch_end, :, :]
    rec_as_src_x_batch[:, :, 0] = \
            np.tile(np.arange(batch_size).reshape([-1, 1]),
                    [1, num_receivers_per_shot])

    feed_dict = {batch_placeholders['sources']: src_batch,
                 batch_placeholders['sources_x']: src_x_batch,
                 batch_placeholders['receivers']: rec_batch,
                 batch_placeholders['receivers_x']: rec_x_batch,
                 batch_placeholders['receivers_as_sources_x']: \
                         rec_as_src_x_batch,
                }

    return feed_dict


def _train_step(step, dataset, model_shape, batch_size, batch_placeholders,
                loss, train_op, sess):
    """Run one step of training.

    Args:
        step: Integer specifying the training step
        dataset: Tuple containing the training dataset
        model_shape: Tuple containing the shape of the model
        batch_size: Integer specifying the number of shots in a minibatch
        batch_placeholders: The placeholders created using
                            _create_batch_placeholders
        loss: The Tensor that gives the cost function value when evaluated
        train_op: The operation that performs one optimization step when
                  evaluated
        sess: A TensorFlow session object

    Returns:
        A float containing the training loss
    """

    feed_dict = _prepare_batch(step, dataset, model_shape, batch_size,
                               batch_placeholders)

    l, _ = sess.run([loss, train_op], feed_dict=feed_dict)

    return l


def _test_dev_step(step, dataset, model_shape, batch_size, batch_placeholders,
                   loss, sess, model=None, test_model=None):
    """Calculate the loss (cost function value) on one development minibatch.

    Args:
        step: Integer specifying the minibatch index
        dataset: Tuple containing the development dataset
        model_shape: Tuple containing the shape of the model
        batch_size: Integer specifying the number of shots in a minibatch
        batch_placeholders: The placeholders created using
                            _create_batch_placeholders
        loss: The Tensor that gives the cost function value when evaluated
        sess: A TensorFlow session object
        model: The model tensor used in the FWI graph (so that we can
               temporarily replace it with test_model, if desired; optional)
        test_model: Array containing the model to compute the development
                    loss on (optional). If not specified, the current value
                    of the model in the session is used.

    Returns:
        A float containing the development loss for this minibatch
    """

    feed_dict = _prepare_batch(step, dataset, model_shape, batch_size,
                               batch_placeholders)
    if test_model is not None:
        feed_dict[model] = test_model

    l = sess.run(loss, feed_dict=feed_dict)

    return l


def _adjoint(model, dx, dt, source_wavefields, modeled_receivers,
             batch_true_receivers, batch_receivers_as_sources_x, propagator,
             num_time_steps, batch_size, num_receivers_per_shot):
    """Use the adjoint-state method to calculate the gradient of the cost
    function with respect to the wave speed model.

    Args:
        model: Tensor containing the wave speed model
        dx: Float specifying grid cell spacing
        dt: Float specifying time between data samples
        source_wavefields: Tensor [num_time_steps, batch_size, :] containing
                           the forward propagated source wavefields
        modeled_receivers: Tensor [num_time_steps, batch_size,
                                   num_receivers_per_shot] containing the
                           receiver data extracted from the source wavefields
        batch_true_receivers: Tensor [num_time_steps, batch_size,
                                      num_receivers_per_shot] containing the
                              true receiver data
        batch_receivers_as_sources_x: Tensor [batch_size,
                                              num_receivers_per_shot, ndim + 1]
                                      containing the receiver coordinates in
                                      the format used for sources
        propagator: A propagator function (forward1d/forward2d)

    Returns:
        A Tensor containing the gradient of the cost function value with
        respect to the model
    """
    #residual = modeled_receivers - batch_true_receivers
    loss, residual = W2(batch_true_receivers,modeled_receivers, num_time_steps,batch_size, num_receivers_per_shot,dt)
    #loss, residual = NIM(modeled_receivers,batch_true_receivers)
    data_wavefields = propagator(model, residual[::-1],
                                 batch_receivers_as_sources_x,
                                 dx, dt)
    source_wavefields = tf.reshape(source_wavefields,
                                   tf.shape(input=data_wavefields))
    source_wavefields = (source_wavefields[:-2]
                         - 2 * source_wavefields[1:-1]
                         + source_wavefields[2:]) / dt**2
    gradient = tf.reduce_sum(input_tensor=source_wavefields * data_wavefields[0:-2][::-1],
                             axis=0) * 2 * dx * dt / model**3

    # As the cost function used is mean squared error, we need to sum
    # over shots and divide by the number of samples, which is the total
    # number of receivers
    tot_num_receivers = tf.reduce_prod(input_tensor=tf.shape(input=batch_true_receivers)[1:3])
    tot_num_receivers = tf.cast(tot_num_receivers, dtype=tf.float32)
    gradient = tf.reduce_sum(input_tensor=gradient, axis=0) / tot_num_receivers

    return gradient


def _entire_dataset_loss_and_gradient(x, model, train_dataset,
                                      dev_dataset,
                                      batch_size,
                                      batch_placeholders,
                                      gradient, loss,
                                      sess, model_shape, loss_file=None):
    """Compute the loss and gradient using the entire dataset instead of a
    minibatch.

    This is designed to work with SciPy's optimizers, allowing me to apply
    L-BFGS-B when the loss and gradient are calculating using the entire
    dataset.

    Args:
        x: A 1D Numpy array containing the model to evaluate
        model: The model tensor used in the FWI graph (so that we can
               temporarily replace it with x)
        train_dataset: A tuple containing the training dataset
        dev_dataset: A tuple containing the development dataset
        batch_size: An integer specifying the number of shots to process
                    simultaneously (should be a divisor of the number of shots)
        batch_placeholders: The placeholders created using
                            _create_batch_placeholders
        gradient: Tensor that gives the gradient when evaluated
        loss: The Tensor that gives the cost function value when evaluated
        sess: A TensorFlow session object
        model_shape: A tuple containing the shape of the model
        loss_file: A file handle to write the training and development loss
                   values to (optional)

    Returns:
        total_loss: Float specifying the mean loss using the entire training
                    dataset
        total_gradient: 64-bit float (as required by SciPy) 1D array containing
                        the mean gradient calculated using the entire training
                        dataset
    """
    x = x.reshape(model_shape)
    num_shots = int(train_dataset[0].shape[1])
    num_batches = num_shots // batch_size

    total_loss = np.float32(0.0)
    total_gradient = np.zeros(model_shape, np.float32)
    for batch_idx in range(num_batches):
        feed_dict = _prepare_batch(batch_idx, train_dataset, model_shape,
                                   batch_size,
                                   batch_placeholders)
        feed_dict[model] = x

        tmp_loss, tmp_grad = sess.run([loss, gradient], feed_dict=feed_dict)
        total_loss += tmp_loss / num_batches
        total_gradient += tmp_grad[0][0] / num_batches

    dev_loss = _get_dev_loss(model_shape, dev_dataset, batch_placeholders,
                             loss, sess, model=model, test_model=x)

    if loss_file is not None:
        loss_file.write('{}, {}\n'.format(total_loss, dev_loss))

    return total_loss, np.float64(total_gradient.ravel())
