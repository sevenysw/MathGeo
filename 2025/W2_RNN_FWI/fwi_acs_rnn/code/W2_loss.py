import tensorflow as tf
import numpy as np

def normalize(x):
    return tf.divide(x,tf.reduce_sum(x,axis=0,keepdims=True))

def my_numpy_func(x,y,z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    #print(x.shape)
    T = np.zeros(x.shape,dtype=np.float32)
    for i in range(x.shape[1]):
       #T[:,i] = np.interp(x[:,i],y[:,i],z)
       T[:,i] = np.interp(x[:,i],y[:,i],z[:,i])
    T = T.astype(np.float32)
    return T

def my_grad_func(t,T,f,mu):
    t = np.array(t)
    T = np.array(T)
    f = np.array(f)
    mu = np.array(mu)
    grad = np.zeros(mu.shape,dtype=np.float32)
    #print(t.shape,T.shape,mu.shape)
    for i in range(mu.shape[1]):
       int_mu = np.sum(mu[:,i])
       #grad[:,i] = np.cumsum(t - T[:,i]) - np.sum(t - T[:,i])
       #grad[:,i] = 2*np.cumsum(t[:,i] - T[:,i]) #my_gradient1
       #grad[:,i] = 2*(np.cumsum(t[:,i] - T[:,i]) - np.sum(t[:,i] - T[:,i]) - (t[:,i] - T[:,i]))#my_gradient2
       grad[:,i] = 2*(np.cumsum(t[:,i] - T[:,i]) - np.sum(t[:,i] - T[:,i]))
       grad[:,i] = grad[:,i] - np.sum(mu[:,i] * grad[:,i])
       grad[:,i] = grad[:,i] / int_mu
    grad = grad.astype(np.float32)
    return grad

@tf.custom_gradient
def W2(labels, predictions, num_time_steps, batch_size, num_receivers_per_shot,dt):
    #print(tf.shape(labels)[0])
    #print(tf.shape(predictions)[0])
    batch_num_source = batch_size
    num_receivers = num_receivers_per_shot
    #W2_loss = tf.Variable(0.)
    inputs_shape=tf.shape(labels)
    labels = tf.reshape(labels,[num_time_steps, batch_num_source * num_receivers])
    predictions = tf.reshape(predictions,[num_time_steps, batch_num_source * num_receivers])
   # dt = 0.001
    t = tf.range(num_time_steps)
    t = tf.cast(t, dtype=tf.float32)
#    b = tf.constant([1, batch_num_source*num_receivers],tf.int32)
    t = tf.reshape(t, [num_time_steps,1]) * dt
    t = tf.tile(t, [1,batch_num_source*num_receivers])

   # c = shift
    c =  tf.constant(100.)
    g = labels + c
    f = predictions + c

 #   for r in range(batch_num_source*num_receivers):
    #scale=10000000000.
    scale = 1.
    mu = normalize(f)*scale
    nu = normalize(g)*scale
    #int_mu = tf.reduce_sum(f, 0)

    ###Cumulative
    F = tf.cumsum(mu, axis=0) #observation
    G = tf.cumsum(nu, axis=0) #modeled

    ###interpolation
    T = tf.numpy_function(func=my_numpy_func, inp=[F, G, t], Tout=[tf.float32])
    T = tf.reshape(tf.cast(T, dtype=tf.float32), [num_time_steps, batch_num_source * num_receivers])
    loss = tf.reduce_sum(tf.reduce_mean(tf.square(t-T)*mu))
    loss = tf.cast(loss, dtype=tf.float32)
    W2_loss = loss

    ###gradient w.r.t.f
    def grad(dy):
        residual = tf.numpy_function(func = my_grad_func, inp = [t,T,f,mu] , Tout=tf.float32)
        return None, dy*residual, None, None, None, None
    #return W2_loss/(batch_num_source*num_receivers),tf.reshape(residual*-1.,[num_time_steps, batch_num_source, num_receivers])
    return W2_loss, grad
