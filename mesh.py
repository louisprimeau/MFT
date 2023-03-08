import numpy as np
from numpy import sqrt, cos
from util import normalize

def make_triangular_lattice(N, A):
    k1, k2 = np.meshgrid(np.arange(N, dtype='float64'), np.arange(int(N * 2 / sqrt(3)), dtype='float64'))
    k1[::2] += 0.5
    k2 *= 3**0.5 / 2
    
    def plot_to_k(R):
        return (R / N - 0.5) * 2 * A

    def k_to_plot(R):
        return N * ((R / (2*A)) + 0.5)
    
    k_int = np.stack((k1, k2)).reshape(2, -1).T
    k = plot_to_k(k_int)

    return k, plot_to_k, k_to_plot

def hexagon_mask(k, A, a, b, c):
    r = A * cos(np.pi/6)
    
    d = (k @ normalize(a + b)).reshape(-1)
    mask = (d > -r) & (d < r)
    
    d = (k @ normalize(b + c)).reshape(-1)
    mask = mask & (d > -r) & (d < r)
    
    d = (k @ normalize(c - a)).reshape(-1)
    mask = mask & (d > -r) & (d < r)

    return k[mask, :]
