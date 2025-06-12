import numpy as np

def calc_r2(inpt1, inpt2):
    nt = inpt1.size
    true_avr = np.mean(inpt1)
    init_avr = np.mean(inpt2)
    diff1 = inpt1 - true_avr
    diff2 = inpt2 - init_avr
    diff1_2 = np.sum(np.multiply(diff1,diff1))
    diff2_2 = np.sum(np.multiply(diff2,diff2))

    r = np.sum(np.multiply(diff1, diff2))/np.sqrt(diff1_2*diff2_2)
    r2 = np.square(r)


    return r2

