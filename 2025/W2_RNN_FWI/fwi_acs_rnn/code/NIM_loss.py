import tensorflow as tf

def normalize(x):
    inputs_shape=tf.shape(x)
    nt = inputs_shape[0]
    #return tf.divide(x,tf.tile(tf.reduce_sum(x,axis=0,keepdims=True),[nt,1,1]))
    return tf.divide(x,tf.reduce_sum(x,axis=0,keepdims=True))

def NIM(labels, predictions):
        shift = tf.constant(100.)
        c = shift
        g = labels + c
        f = predictions + c
        scale = 10000.
        mu = normalize(f)*scale
        nu = normalize(g)*scale

        F = tf.cumsum(mu,axis=0)
        G = tf.cumsum(nu,axis=0)
        #scale = 10000.
        #scale = 1.
        #mu = normalize(F)*scale
        #nu = normalize(G)*scale
        # tf op
        losses = tf.compat.v1.losses.mean_squared_error(F,G)
        return losses
        ## man crafted op
        #@tf.custom_gradient
        #def mse(mu,nu):
        #   losses = tf.compat.v1.losses.mean_squared_error(mu,nu)
        #   div = tf.shape(labels)[0]*tf.shape(labels)[1]*tf.shape(labels)[2]
        #   div = tf.cast(div, dtype=tf.float32)
        #   def grad(dy):
        #        return None, dy*(nu - mu)/div*2.
        #   return losses, grad
        #return mse(mu,nu)
