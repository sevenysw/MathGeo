import sys
import numpy as np
from setup_mar import setup_par
from gen_resi import gen_resi

def make_seis_data():
    """Generate synthetic data from the chosen section of the model."""
#    seam_model_path = sys.argv[1]
    model = setup_par()
    vpmodel_true = model['vpmodel_init']
    dx = model['dx']
    dt = model['dt']
    sources = model['sources']
    sources_x = model['sources_x']
    receivers_x = model['receivers_x']
    propagator = model['propagator']
    seis_file = model['seis_path']
    adj_file = './adj_source'
    batch_size = model['batch_size']
    source_file = model['source_path']

    seis_file = model['seis_path'] + '.npy'
    receivers_true = np.load(seis_file)

    receivers = gen_resi(vpmodel_true, dx, dt, sources, sources_x, receivers_x,receivers_true,
                         propagator, batch_size)

    np.save(adj_file, receivers)
    #np.save('seam_wave', out_wave )

if __name__ == '__main__':
    make_seis_data()
