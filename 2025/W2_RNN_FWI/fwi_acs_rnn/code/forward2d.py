"""2D scalar wave equation forward modeling implemented using TensorFlow
"""
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()



class TimeStepCell(tf.compat.v1.nn.rnn_cell.RNNCell):
#class TimeStepCell(tf.nn.rnn_cell.RNNCell):

    """One forward modeling step of scalar wave equation with PML.

    Args:
        model_padded2_dt2: Tensor containing squared wave speed times squared
            time step size
        dt: Float specifying time step size
        sigmaz: 1D Tensor that is only non-zero in z direction PML regions
        sigmax: 1D Tensor that is only non-zero in x direction PML regions
        first_z_deriv: Function to calculate the first derivative of the input
                       2D Tensor in the z direction
        first_x_deriv: Function to calculate the first derivative of the input
                       2D Tensor in the x direction
        laplacian: Function to calculate the Laplacian of the input
                   2D Tensor
        sources_x: 3D Tensor [batch_size, num_sources_per_shot, 3]
                   where [:, :, 0] contains the index in the batch, and
                   [:, :, 1] contains the integer z cell coordinate of
                   the source, and [:, :, 2] contains the integer x cell
                   coordinate of the source

    """

    def __init__(self, model_padded2_dt2, dt, sigmaz, sigmax,
                 first_z_deriv, first_x_deriv, laplacian,
                 sources_x):
        super(TimeStepCell, self).__init__()
        self.model_padded2_dt2 = model_padded2_dt2
        self.dt = dt
        self.sigmaz = sigmaz
        self.sigmax = sigmax
        self.sigma_sum = sigmaz + sigmax
        self.sigma_prod_dt2 = (sigmaz * sigmax) * dt**2
        self.factor = 1 / (1 + dt * self.sigma_sum / 2)
        self.first_z_deriv = first_z_deriv
        self.first_x_deriv = first_x_deriv
        self.laplacian = laplacian
        self.sources_x = sources_x
        self.nz_padded = model_padded2_dt2.shape[0]
        self.nx_padded = model_padded2_dt2.shape[1]
        self.nzx_padded = self.nz_padded * self.nx_padded

    @property
    def state_size(self):
        """The RNN state (passed between RNN units) contains two time steps
        of the wave field, and the PML auxiliary wavefields phiz and phix.
        """
        return [self.nzx_padded, self.nzx_padded,
                self.nzx_padded, self.nzx_padded]

    @property
    def output_size(self):
        """The output of the RNN unit contains one time step of the wavefield.
        """
        return self.nzx_padded

    def __call__(self, inputs, state):
        """Propagate the wavefield forward one time step.

        Args:
            inputs: An array containing the source amplitudes for this time
                    step
            state: A list containing the two previous wave field time steps
                   and the auxiliary wavefields phiz and phix

        Returns:
            output: The current wave field
            state: A list containing the current and one previous wave field
                   time steps and the updated auxiliary wavefields phiz and
                   phix
        """
        inputs_shape = tf.shape(input=state[0])
        batch_size = inputs_shape[0]
        model_shape = [batch_size, self.nz_padded, self.nx_padded]
        wavefieldc = tf.reshape(state[0], model_shape)
        wavefieldp = tf.reshape(state[1], model_shape)
        phizc = tf.reshape(state[2], model_shape)
        phixc = tf.reshape(state[3], model_shape)

        lap = self.laplacian(wavefieldc)
        wavefieldc_z = self.first_z_deriv(wavefieldc)
        wavefieldc_x = self.first_x_deriv(wavefieldc)
        phizc_z = self.first_z_deriv(phizc)
        phixc_x = self.first_x_deriv(phixc)

        # The main evolution equation:

#        wavefieldf = self.model_padded2_dt2*lap  + (2. * wavefieldc - wavefieldp)
        wavefieldf = self.factor * \
                (self.model_padded2_dt2
                 * (lap + phizc_z + phixc_x)
                 + self.dt * self.sigma_sum * wavefieldp / 2.
                 + (2. * wavefieldc - wavefieldp)
                 - self.sigma_prod_dt2 * wavefieldc)

        # Update PML variables phix, phiz
        phizf = (phizc - self.dt * self.sigmaz * phizc
                 - self.dt * (self.sigmaz - self.sigmax) * wavefieldc_z)
        phixf = (phixc - self.dt * self.sigmax * phixc
                 - self.dt * (self.sigmax - self.sigmaz) * wavefieldc_x)

        # Add the sources
        # f(t+1, z_s, x_s) += c(z_s, x_s)^2 * dt^2 * s(t)
        # We need to expand "inputs" to be the same size as f(t+1), so we
        # use tf.scatter_nd. This will create an array
        # of the right size, almost entirely filled with zeros, with the
        # source amplitudes (multiplied by c^2 * dt^2) in the right places.

        wavefieldf += tf.scatter_nd(self.sources_x, inputs, model_shape)

        return (tf.reshape(wavefieldf, inputs_shape),
                [tf.reshape(wavefieldf, inputs_shape),
                 tf.reshape(wavefieldc, inputs_shape),
                 tf.reshape(phizf, inputs_shape),
                 tf.reshape(phixf, inputs_shape)])


def forward2d(model, sources, sources_x,
              dx, dt, pml_width=None, pad_width=None,
              profile=None):
    """Forward modeling using the 2D wave equation.

    Args:
        model: 2D tf.Variable or tf.Tensor velocity model
        sources: 3D Tensor [num_time_steps, batch_size, num_sources_per_shot]
                 containing source amplitudes
        sources_x: 3D Tensor [batch_size, num_sources_per_shot, 3]
                   where [:, :, 0] contains the index in the batch, and
                   [:, :, 1] contains the integer z cell coordinate of
                   the source, and [:, :, 2] contains the integer x cell
                   coordinate of the source
        dx: float specifying size of each cell (dx == dz)
        dt: float specifying time between time steps
        pml_width: number of cells in PML (optional)
        pad_width: number of padding cells outside PML (optional)
        profile: 1D array specifying PML profile (optional)

    Returns:
        4D Tensor [num_time_steps, batch_size, nz, nx] containing time steps of
        wavefields. Padding that was added is removed.
    """

    if pml_width is None:
        pml_width = 10
    if pad_width is None:
        pad_width = 8

    total_pad = pml_width + pad_width

    nz_padded, nx_padded = _set_x(model, total_pad)

    model_padded2_dt2 = _set_model(model, total_pad, dt)

    profile, pml_width = _set_profile(profile, pml_width, dx)

    sigmaz, sigmax = _set_sigma(nz_padded, nx_padded, total_pad, pad_width,
                                profile)

    sources, sources_x = _set_sources(sources, sources_x, total_pad,
                                      model_padded2_dt2)

    d1_kernel, d2_kernel = _set_kernels(dx)

    first_z_deriv, first_x_deriv, laplacian = _set_deriv_funcs(d1_kernel,
                                                               d2_kernel)

    cell = TimeStepCell(model_padded2_dt2, dt, sigmaz, sigmax,
                        first_z_deriv, first_x_deriv, laplacian, sources_x)

    out, _ = tf.compat.v1.nn.dynamic_rnn(cell, sources,
                               dtype=tf.float32, time_major=True)

    out = tf.reshape(out, [int(out.shape[0]), # time
                           tf.shape(input=out)[1], # batch
                           nz_padded,
                           nx_padded])

    return out[:, :, total_pad : -total_pad, total_pad : -total_pad]


def _set_x(model, total_pad):
    """Calculate the size of the model after padding has been added.

    Args:
        model: 2D tf.Variable or tf.Tensor velocity model
        total_pad: Integer specifying padding to add to each edge

    Returns:
        Integers specifying number of cells in padded model in z and x
    """
    nz = int(model.shape[0])
    nx = int(model.shape[1])
    nz_padded = nz + 2 * total_pad
    nx_padded = nx + 2 * total_pad
    return nz_padded, nx_padded


def _set_model(model, total_pad, dt):
    """Add padding to the model (extending edge values) and compute c^2 * dt^2.

    TensorFlow does not provide the option to extend the edge values into
    the padded region (unlike Numpy, which has an 'edge' option to do this),
    so we need to split the 2D array into 1D columns, pad the top with
    the first value from the column, and pad the bottom with the final value
    from the column, and then repeat it for rows.

    Args:
        model: 2D tf.Variable or tf.Tensor velocity model
        total_pad: Integer specifying padding to add to each edge
        dt: Float specifying time step size

    Returns:
        A 2D Tensor containing the padded, squared model times the squared
        time step size
    """
    def pad_tensor(tensor, axis, pad_width):
        """Split the 2D Tensor into rows/columns along the specified axis, then
        iterate through those rows/columns padding the beginning and end with
        the first and last elements from the row/column. Then recombine back
        into a 2D Tensor again.
        """
        tmp1 = []
        for row in tf.unstack(tensor, axis=axis):
            tmp2 = tf.pad(tensor=row, paddings=[[pad_width, 0]], mode='CONSTANT',
                          constant_values=row[0])
            tmp2 = tf.pad(tensor=tmp2, paddings=[[0, pad_width]], mode='CONSTANT',
                          constant_values=row[-1])
            tmp1.append(tmp2)
        return tf.stack(tmp1, axis=axis)

    model_padded = pad_tensor(model, 0, total_pad)
    model_padded = pad_tensor(model_padded, 1, total_pad)
    return tf.square(model_padded) * dt**2


def _set_profile(profile, pml_width, dx):
    """Create a profile for the PML.

    Args:
        profile: User supplied profile, if None use default
        pml_width: Integer. If profile is None, create a PML of this width.
        dx: Float specifying spacing between grid cells

    Returns:
        profile: 1D array containing PML profile
        pml_width: Integer specifying the length of the profile
    """
    # This should be set to approximately the maximum wave speed at the edges
    # of the model
    max_vel = 5000.
    if profile is None:
        profile = ((np.arange(pml_width)/(1.0*pml_width))**2
                   * 3. * max_vel * np.log(1000.)
                   / (2. * dx * pml_width))
    else:
        pml_width = len(profile)
    return profile, pml_width


def _set_sigma(nz_padded, nx_padded, total_pad, pad_width, profile):
    """Create 1D sigma arrays that contain the PML profile in the PML regions.

    Args:
        nz_padded: Integer specifying the number of depth cells in the padded
                   model
        nx_padded: Integer specifying the number of x cells in the padded model
        total_pad: Integer specifying the number of cells of padding added to
                   each edge of the model
        pad_width: Integer specifying the number of cells of padding that are
                   not part of the PML
        profile: 1D array containing the PML profile for the bottom/right side
                 of the model (for the top/left side, it will be reversed)

    Returns:
        1D sigma arrays for the depth and x directions
    """
    def sigma_1d(n_padded, total_pad, pad_width, profile):
        """Create one 1D sigma array."""
        sigma = np.zeros(n_padded, np.float32)
        sigma[total_pad-1:pad_width-1:-1] = profile
        sigma[-total_pad:-pad_width] = profile
        sigma[:pad_width] = sigma[pad_width]
        sigma[-pad_width:] = sigma[-pad_width-1]
        return sigma

    sigmaz = sigma_1d(nz_padded, total_pad, pad_width, profile)
    sigmaz = sigmaz.reshape([-1, 1])
    sigmaz = np.tile(sigmaz, [1, nx_padded])

    sigmax = sigma_1d(nx_padded, total_pad, pad_width, profile)
    sigmax = sigmax.reshape([1, -1])
    sigmax = np.tile(sigmax, [nz_padded, 1])

    return tf.constant(sigmaz), tf.constant(sigmax)

def _set_sources(sources, sources_x, total_pad, model_padded2_dt2):
    """Set the source amplitudes, and the source positions.

    Args:
        sources: 3D Tensor [num_time_steps, batch_size, num_sources_per_shot]
                 containing source amplitudes
        sources_x: 3D Tensor [batch_size, num_sources_per_shot, 3]
                   where [:, :, 0] contains the index in the batch, and
                   [:, :, 1] contains the integer z cell coordinate of
                   the source, and [:, :, 2] contains the integer x cell
                   coordinate of the source
        total_pad: Integer specifying padding added to each edge of the model
        model_padded2_dt2: Tensor containing squared wave speed times squared
                           time step size

    Returns:
        sources: 3D Tensor containing source amplitude * c^2 * dt^2
        sources_x: 3D Tensor like the input, but with total_pad added to
                   [:, :, 1] and [:, :, 2]
    """
    # I add "total_pad" to the source coordinates as the coordinates currently
    # refer to the coordinates in the unpadded model, but we need them to
    # refer to the coordinates when padding has been added. We only want to add
    # this to [:, :, 1] and [:, :, 2], which contains the depth and x
    # coordinates, so I multiply by an array that is 0 for [:, :, 0], and 1
    # for [:, :, 1] and [:, :, 2].
    sources_x += (tf.ones_like(sources_x) * total_pad
                  * np.array([0, 1, 1]).reshape([1, 1, 3]))

    # The propagator injected source amplitude multiplied by c(x)^2 * dt^2
    # at the locations of the sources, so we need to extract the wave speed
    # at these locations. I do this using tf.gather
    sources_v = tf.gather_nd(model_padded2_dt2, sources_x[:, :, 1:])

    # The propagator does not need the unmultiplied source amplitudes,
    # so I will save space by only storing the source amplitudes multiplied
    # by c(x)^2 * dt^2
    sources = sources * sources_v

    return sources, sources_x


def _set_kernels(dx):
    """Create spatial finite difference kernels.

    The kernels are reshaped into the appropriate shape for a 2D
    convolution, and saved as constant tensors.

    Args:
        dx: Float specifying the grid cell spacing

    Returns:
        d1_kernel: 3D Tensor for 1D first derivative
        d2_kernel: 3D Tensor for 2D second derivative (Laplacian)
    """
    # First derivative
    d1_kernel = (np.array([1./12., -2./3., 0., 2./3., -1./12.], np.float32)/ dx)
    d1_kernel = tf.constant(d1_kernel)

    # Second derivative
    d2_kernel = np.array([[0.0,   0.0, -1./12., 0.0, 0.0],
                          [0.0,   0.0, 4./3.,   0.0, 0.0],
                          [-1./12., 4./3., -10./2.,  4./3., -1./12.],
                          [0.0,   0.0, 4./3.,   0.0, 0.0],
                          [0.0,   0.0, -1./12., 0.0, 0.0]],
                         np.float32)
    d2_kernel /= dx**2
    d2_kernel = d2_kernel.reshape([5, 5, 1, 1]) 
    d2_kernel = tf.constant(d2_kernel)

    return d1_kernel, d2_kernel


def _set_deriv_funcs(d1_kernel, d2_kernel):
    """Create functions to apply first and second derivatives.

    Args:
        d1_kernel: 3D Tensor for 1D first derivative
        d2_kernel: 3D Tensor for 2D second derivative (Laplacian)

    Returns:
        Functions for applying first (in depth and x) and second derivatives
    """
    def make_deriv_func(kernel, shape):
        """Returns a function that takes a derivative of its input."""
        def deriv(x):
            """Take a derivative of the input."""
            return tf.squeeze(tf.nn.conv2d(input=tf.expand_dims(x, -1),
                                           filters=tf.reshape(kernel, shape),
                                           strides=[1, 1, 1, 1], padding='SAME'))
        return deriv

    first_z_deriv = make_deriv_func(d1_kernel, [-1, 1, 1, 1])
    first_x_deriv = make_deriv_func(d1_kernel, [1, -1, 1, 1])
    laplacian = make_deriv_func(d2_kernel, d2_kernel.shape)

    return first_z_deriv, first_x_deriv, laplacian
