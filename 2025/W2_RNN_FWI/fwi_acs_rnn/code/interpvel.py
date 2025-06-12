from sys import argv
import numpy as np
from scipy import interpolate

def interpvel():
    """Interpolate the SEAM model onto a coarse grid with and write the result
    to a file.

    Command-line arguments:
        1. Path to input SEAM model
        2. Path to output interpolated model
        3. Number of x cells in input model
        4. Number of depth cells in input model
        5. X cell spacing in input model
        6. Depth cell spacing in input model
        7. X cell spacing in interpolated model
        8. Depth cell spacing in interpolated model
    """
    filebinary = argv[1]
    fileinterp = argv[2]
    n1old = int(argv[3])
    n2old = int(argv[4])
    d1old = float(argv[5])
    d2old = float(argv[6])
    d1new = float(argv[7])
    d2new = float(argv[8])

    print(filebinary, fileinterp, n1old, n2old, d1old, d2old, d1new, d2new)

    # Load data
    array = np.fromfile(filebinary, dtype=np.float32).reshape([n1old, n2old])
    print('n1old', n1old, 'n2', n2old)

    # Interpolate
    xold = np.arange(0, n1old*d1old, d1old)
    zold = np.arange(0, n2old*d2old, d2old)
    print('z', zold.shape, 'x', xold.shape, 'a', array.shape)
    f = interpolate.interp2d(zold, xold, array, kind='quintic')

    xnew = np.arange(0, n1old*d1old, d1new)
    znew = np.arange(0, n2old*d2old, d2new)

    print(znew.shape, xnew.shape)

    new_array = f(znew, xnew).astype(np.float32)
    print(new_array.shape)

    new_array.tofile(fileinterp)

if __name__ == '__main__':
    interpvel()
