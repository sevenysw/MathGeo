import tensorflow as tf
tf.compat.v1.disable_eager_execution()


#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import numpy as np
from fwi import (_create_batch_placeholders, _extract_receivers,
                 _prepare_batch)
from W2_loss import W2
from NIM_loss import NIM

def gen_resi(model, dx, dt, sources, sources_x, receivers_x, receivers_true,propagator,
             batch_size=1):
    """Generate synthetic receiver data.

    Args:
        model: A Numpy array containing the wave speed model to use
        dx: A float specifying the cell spacing
        dt: A float specifying the time step interval
        sources: A 3D array [num_time_steps, num_shots, num_sources_per_shot]
                 containing source waveforms
        sources_x: A 3D array [num_shots, num_sources_per_shot, :] containing
                   the source coordinates
        receivers_x: A 3D array [num_shots, num_receivers_per_shot, :]
                     containing the receiver coordinates
        propagator: An wave propagator function (forward1d/forward2d)
        batch_size: The number of shots to process simultaneously (optional)

    Returns:
        A 3D array [num_time_steps, num_shots, num_receivers_per_shot]
        containing the recorded receiver data
    """

    ndim = int(model.ndim)
    num_time_steps = int(sources.shape[0])
    num_shots = int(sources.shape[1])
    num_sources_per_shot = int(sources.shape[2])
    num_receivers_per_shot = int(receivers_x.shape[1])

    assert num_shots % batch_size == 0
    num_batches = num_shots // batch_size

    model = tf.constant(model)

    residual = np.zeros([num_time_steps, num_shots, num_receivers_per_shot],
                         np.float32)

    dataset = sources, sources_x, receivers_true, receivers_x

    batch_placeholders = _create_batch_placeholders(ndim,
                                                    num_time_steps,
                                                    num_sources_per_shot,
                                                    num_receivers_per_shot)

    out_wavefields = propagator(model, batch_placeholders['sources'],
                                batch_placeholders['sources_x'], dx, dt)

    out_receivers = _extract_receivers(out_wavefields,
                                       batch_placeholders['receivers_x'],
                                       num_receivers_per_shot)
    #loss, residual_batch = W2(batch_placeholders['receivers'], out_receivers, num_time_steps, batch_size, num_receivers_per_shot, dt)
    loss = W2(batch_placeholders['receivers'], out_receivers, num_time_steps, batch_size, num_receivers_per_shot, dt)
    #loss = NIM(batch_placeholders['receivers'], out_receivers)
    #loss = tf.compat.v1.losses.mean_squared_error(batch_placeholders['receivers'], out_receivers)
    var_grad = tf.reshape(tf.gradients(loss, [out_receivers])[0],[num_time_steps,batch_size,num_receivers_per_shot])

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for batch_idx in range(num_batches):
        print (batch_idx, num_batches)
        feed_dict = _prepare_batch(batch_idx, dataset, model.shape, batch_size,
                                   batch_placeholders)

        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size

        residual[:, batch_start : batch_end, :] = \
                sess.run(var_grad, feed_dict=feed_dict)
                #sess.run(residual_batch, feed_dict=feed_dict)
        #### print and save snapshot
        #print batch_idx, num_batches
        #if batch_idx == 0:
        #       out_wave = sess.run(out_wavefields, feed_dict=feed_dict)
        #       np.save('seam_wave', out_wave )

    #return receivers, out_wave   # output snapshot in test
    return residual
