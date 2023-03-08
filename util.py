import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

def lorentzian(x, x0, gamma):
    return ((1 / (x - x0 + 1j*gamma)) / np.pi).real

def hermite1(x, x0):
    return 0.5 * (1 - scipy.special.erf(x - x0)) - (1 / (4 * np.pi)) * 2 * x * np.exp(-(x - x0)**2)
