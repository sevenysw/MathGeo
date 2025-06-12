import math
import numpy as np
def SNR(img,imgn):
    return 10*math.log10(np.linalg.norm(img)**2 / np.linalg.norm(imgn - img)**2)