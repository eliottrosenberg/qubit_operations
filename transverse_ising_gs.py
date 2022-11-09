# analytical solution to the Transverse Field Ising Model, written by Eliott Rosenberg

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import coo_array
from scipy.linalg import svd

def fermion_H(n,g,J=1):
    i = np.append( np.arange(n-1), np.arange(n) )
    j = np.append( np.arange(1,n), np.arange(n) )
    data = np.append( J*np.ones(n-1), -J*g*np.ones(n) )
    H = coo_array( (data, (i,j)) )
    return H

def ising_ground_state_energy(n,g,J=1):
    # returns the ground state energy of the transverse field Ising model
    H = fermion_H(n,g,J).toarray()
    l = svd(H, compute_uv=False)
    return -sum(l)

def ising_ground_state(n,g,J=1):
    H = fermion_H(n,g,J).toarray()
    Otilde, l, OT = svd(H)
    E = -sum(l)
    M = Otilde @ OT
    ZZ = np.array([M[i,i+1] for i in range(n-1)])
    X = -np.diag(M)
    return E, ZZ, X