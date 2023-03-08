import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, tan, exp, dot, array, sqrt, pi
import itertools
from numpy import ones, zeros, kron, diag, array, linspace, pi
from numpy import concatenate as cat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import matplotlib as mpl
import scipy
import numpy as np

from hamiltonian import make_H_bdg
from mesh import make_triangular_lattice, hexagon_mask
from observables import DoS, filling, solve_chemical_potential

# Let the lattice spacing be 1
a = array([1, 0]).reshape(2, 1)
b = array([0.5, sqrt(3)/2]).reshape(2,1)
c = b - a

# Two useful reciprocal space lengths
A = 4 * pi / 3
B = pi / sqrt(3)

# k-space sampling size
N = 30
k, plot_to_k, k_to_plot = make_triangular_lattice(N, A)
k = hexagon_mask(k, A, a, b, c)
k_plot = k_to_plot(k)

# BZ coordinates
bz1_pts = k_to_plot(A * array([a, b, c, -a, -b, -c, a]).reshape(7, 2))

# initial occupation
#D = np.ones((8, 8)) * 1
#D = np.ones((16, 16)) * 0.5
D = np.zeros((16, 16))

# Diagonalize
H = make_H_bdg(-1, 0.6, 0.4, D, k)
e, v = np.linalg.eigh(H)

v = v / np.linalg.norm(v, axis=(0, 1)) # normalize over k
v = np.transpose(v, axes=(2, 0, 1)) # bring identities to front

energies, counts = filling(e)
counts = counts * 2 / k.size
filling_func = scipy.interpolate.interp1d(energies, counts)
erange = np.linspace(np.min(energies), np.max(energies), 5000)
counts = filling_func(erange)
dos = DoS(erange, energies, n=5000)
mu = solve_chemical_potential(erange, dos, 10, 0.75)
#D = order_parameter(e, v, mu, 1)
