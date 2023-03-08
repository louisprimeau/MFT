import numpy as np
from numpy import sqrt, exp, array


def make_1_hopping(t1,k):

    a = array([1, 0]).reshape(2, 1)
    b = array([0.5, sqrt(3)/2]).reshape(2,1)
    c = b - a

    zeros = np.zeros_like(k@b)
    return t1 * np.stack([np.hstack([zeros, 1+exp(-1j*k@a), 1 + exp(-1j*k@b), exp(-1j*k@a)+exp(-1j*k@b)]),
                          np.hstack([1+exp(1j*k@a), zeros, 1 + exp(-1j*k@c), 1+exp(-1j*k@b)]),
                          np.hstack([1 + exp(1j*k@b), 1 + exp(1j*k@c), zeros, 1+exp(-1j*k@a)]),
                          np.hstack([exp(1j*k@a)+exp(1j*k@b), 1+exp(1j*k@b), 1+exp(1j*k@a), zeros])], axis=1)

    
def make_2_hopping(t2, k):

    a = array([1, 0]).reshape(2, 1)
    b = array([0.5, sqrt(3)/2]).reshape(2,1)
    c = b - a

    zeros = np.zeros_like(k@b)
    h = t2 * np.stack([np.hstack([zeros,            exp(1j*k@c) + exp(-1j*k@b), exp(-1j*k@c) + exp(-1j*k@a), 1 + exp(-1j*k@(a+b))]),
                          np.hstack([zeros, zeros,            exp(1j*k@a) + exp(-1j*k@b),  exp(-1j*k@a)+exp(-1j*k@c)]),
                          np.hstack([zeros, zeros, zeros,             exp(-1j*k@b) + exp(1j*k@c)]),
                          np.hstack([zeros, zeros, zeros,  zeros            ])], axis=1)
    return h.transpose((0, 2, 1)).conj() + h             

def abcd_spins(J):
    s_x = np.array([[0, 1], [1, 0]])
    s_y = np.array([[0,-1j],[1j, 0]])
    s_z = np.array([[1, 0], [0, -1]])
    
    A_s = (s_x + s_y + s_z) / 2
    B_s = (-s_x + s_y - s_z) / 2
    C_s = (s_x - s_y - s_z) / 2
    D_s = (-s_x - s_y + s_z) / 2
        
    return J*A_s, J*B_s, J*C_s, J*D_s

def make_spinH(J, k):
    H = np.zeros((k.shape[0], 8, 8), dtype='complex128')
    A_s, B_s, C_s, D_s = abcd_spins(J)
    H[:, 0:2, 0:2] = A_s
    H[:, 2:4, 2:4] = B_s
    H[:, 4:6, 4:6] = C_s
    H[:, 6:8, 6:8] = D_s
    return H

def make_1_superconducting(D, n_avg, k):

    a = array([1, 0]).reshape(2, 1)
    b = array([0.5, sqrt(3)/2]).reshape(2,1)
    c = b - a

    zeros = np.zeros_like(k@b)
    H = np.stack([np.hstack([zeros, 1+exp(-1j*k@a), 1 + exp(-1j*k@b), exp(-1j*k@a)+exp(-1j*k@b)]),
                  np.hstack([1+exp(1j*k@a), zeros, 1 + exp(-1j*k@c), 1+exp(-1j*k@b)]),
                  np.hstack([1 + exp(1j*k@b), 1 + exp(1j*k@c), zeros, 1+exp(-1j*k@a)]),
                  np.hstack([exp(1j*k@a)+exp(1j*k@b), 1+exp(1j*k@b), 1+exp(1j*k@a), zeros])], axis=1)
    H = np.kron(H, np.array([[1, 1], [1, 1]]))
    H = D * H * np.repeat(n_avg.reshape(-1, 1), repeats=8, axis=1)
    return H
    
    
def make_H(t1, t2, J, D, n_avg, k):

    # hopping
    H = np.kron(make_1_hopping(t1, k), np.eye(2)) + \
        np.kron(make_2_hopping(t2, k), np.eye(2))
    
    # spin coupling
    A_s, B_s, C_s, D_s = abcd_spins(J, k)
    H[:, 0:2, 0:2] = A_s
    H[:, 2:4, 2:4] = B_s
    H[:, 4:6, 4:6] = C_s
    H[:, 6:8, 6:8] = D_s
    
    # superconducting
    H = H + make_1_superconducting(D, n_avg, k)
    
    return H


def make_H_bdg(t1, t2, J, D, k):

    a = array([1, 0]).reshape(2, 1)
    b = array([0.5, sqrt(3)/2]).reshape(2,1)
    c = b - a

    hopping1 = make_1_hopping(1, k)
    hopping2 = make_2_hopping(1, k)
    copy = lambda m: np.repeat(m.reshape(1, *m.shape), repeats=k.shape[0], axis=0)
  
    N_11 = t1 * hopping1 + t2 * hopping2 + copy(J * np.diag([1, -1, -1, 1]) / 2)
    N_22 = t1 * hopping1 + t2 * hopping2 + copy(J * np.diag([-1, 1, 1, -1]) / 2)
    N_12 = copy(J * (-1j * np.diag([1, 1, -1, -1]) + np.diag([1, -1, 1, -1])) / 2)
    N_21 = copy(J * (1j * np.diag([1, 1, -1, -1]) + np.diag([1, -1, 1, -1])) / 2)
    
    N = np.concatenate([np.concatenate([N_11, N_12], axis=2), 
                        np.concatenate([N_21, N_22], axis=2)], axis=1)

    zeros = np.zeros_like(k@b)
    o = np.ones_like(k@b)
    """M_12 = np.stack([ np.hstack([zeros, o*D[0, 13], o*D[0, 14], o*D[0, 15]]),
                      np.hstack([zeros,   zeros, o*D[1, 14], o*D[1, 15]]),
                      np.hstack([zeros,   zeros, zeros,   o*D[2, 15]]),
                      np.hstack([zeros, zeros,  zeros,  zeros])], axis=1)
    M_12 = (M_12 + M_12.conj().transpose(0, 2, 1)) * hopping1
    M_21 = np.stack([ np.hstack([zeros, o*D[5, 9], o*D[6, 10], o*D[7, 11]]),
                      np.hstack([zeros,   zeros, o*D[6, 10], o*D[7, 11]]),
                      np.hstack([zeros,   zeros, zeros,   o*D[7, 11]]),
                      np.hstack([zeros,   zeros, zeros,   zeros])], axis=1) 
    M_21 = (M_21 + M_21.conj().transpose(0, 2, 1)) * hopping1
    """
    M_11 = copy(np.zeros((4, 4)))
    M_22 = copy(np.zeros((4, 4)))
    
    
    M_12 = hopping1 * D[0:4, 12:16].reshape(1, 4, 4)
    M_21 = hopping1 * D[4:8, 8:12].reshape(1, 4, 4)
    #M_11 = hopping1 * D[0:4, 8:12].reshape(1, 4, 4)
    #M_22 = hopping1 * D[4:8, 12:16].reshape(1, 4, 4)
    
    M = np.concatenate([np.concatenate([M_11, M_12], axis=2), 
                        np.concatenate([M_21, M_22], axis=2)], axis=1)
    
    return np.concatenate([np.concatenate([N              , M], axis=2), 
                           np.concatenate([M.transpose(0, 2, 1).conjugate(), -N.transpose(0, 2, 1)], axis=2)],axis=1)
