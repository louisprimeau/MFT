import numpy as np
from util import gaussian
import scipy

def filling(E):
    es, counts = np.unique(E, return_counts=True)
    idxs = np.argsort(es)
    es, counts = es[idxs], counts[idxs]
    ns = np.cumsum(counts)
    return es, ns

# dos is just radial basis smoothing
def DoS(sample_points, E, n=10000):
    sigma = 40 * (sample_points[1] - sample_points[0])
    dos = np.zeros(n)
    for e in E: dos += gaussian(sample_points, e, sigma)  
    return dos / E.size

def fermi_dirac(E, mu, beta):
    return 1 / (np.exp(beta * (E - mu)) + 1)

def solve_chemical_potential(E, rho, beta, n_e):
    eq = lambda mu: n_e - np.trapz(rho * fermi_dirac(E, mu, beta), E)
    res = scipy.optimize.root_scalar(eq, method='bisect', bracket=(min(E), max(E)))
    return res.root
    
def avg_particles(E, vs, mu, beta):
    return np.sum(np.abs(vs)**2 * fermi_dirac(E, mu, beta), axis=(1, 2))

def order_parameter(E, v, mu, beta):
    Z = np.sum(exp(-E * beta), axis=1)
    D = np.zeros((16, 16), dtype='complex128')
    for i in range(16):
        for j in range(16):
            D[i,j] = sum(np.sum(exp(-beta * E[:, n]) / Z * np.sum(v[n].conj() * v[n] + v[n, :, i:i+1].conj() * v[n] + v[n, :, j:j+1].conj() * v[n] - 2 * np.abs(v[n, :, i:i+1])**2 - 2 * np.abs(v[n, :, j:j+1])**2, axis=1),axis=0) for n in range(16))
    return D

