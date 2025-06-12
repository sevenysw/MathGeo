import sys
import numpy as np
from setup_mar import setup_par
#from setup_par import setup_par
from gen_data import gen_data

def make_seis_data():
    """Generate synthetic data from the chosen section of the model."""
#    seam_model_path = sys.argv[1]
    model = setup_par()
    vpmodel_true = model['vpmodel_true']
    dx = model['dx']
    dt = model['dt']
    sources = model['sources']
    sources_x = model['sources_x']
    receivers_x = model['receivers_x']
    propagator = model['propagator']
    seis_path = model['seis_path']
    batch_size = model['batch_size']
    source_path = model['source_path']

    receivers = gen_data(vpmodel_true, dx, dt, sources, sources_x, receivers_x,
                         propagator, batch_size)

    np.save(seis_path, receivers)
    np.save(source_path, sources)
    #np.save('seam_wave', out_wave )

if __name__ == '__main__':
    make_seis_data()
