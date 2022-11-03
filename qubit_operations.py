## Qubit operations, written by Eliott Rosenberg, 2022

from scipy.sparse import coo_array
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy import sparse

def digit(num,base,i):
    d = num//base**i - base * (num//base**(i+1))
    return d

def Z_vector(n,i):
    j = np.arange(2**n)
    Z = -2*digit(j,2,i) + 1
    return Z
    
def Zi(n,i):
    data = Z_vector(n,i)
    j = np.arange(2**n)
    return coo_array( (data, (j, j)) )

def Xi(n,i):
    i1 = np.arange(2**n)
    data = np.ones(2**n)
    dig = -2*digit(i1,2,i) + 1
    i2 = i1 + dig * 2**i
    return coo_array( (data, (i1,i2) ) )
    
def Yi(n,i):
    i1 = np.arange(2**n)
    dig = -2*digit(i1,2,i) + 1
    i2 = i1 + dig * 2**i
    data = -dig*1j
    return coo_array( (data, (i1,i2) ) )

def Hi(n,i):
    return 1/np.sqrt(2) * ( Xi(n,i) + Zi(n,i) )

def X90(n,i):
    return 1/np.sqrt(2) * (sparse.eye(2**n) - 1j*Xi(n,i) )

def pauli_i(n,i,type):
    if type == 1:
        return Xi(n,i)
    elif type == 2:
        return Yi(n,i)
    elif type == 3:
        return Zi(n,i)


def hamiltonian(which_paulis,coeffs):
    which_paulis = np.array(which_paulis)
    n = len(which_paulis[0])
    H = coo_array((2**n,2**n),dtype=np.complex128 if 2 in which_paulis else float)
    for term in range(len(coeffs)):
        which_pauli = which_paulis[term]
        initialized = False
        for q in range(n):
            if which_pauli[q] == 0:
                continue
            else:
                Piq = pauli_i(n,q,which_pauli[q])
                if initialized:
                    Pi = Pi @ Piq
                else:
                    Pi = Piq
                    initialized = True
        if not initialized:
            Pi = sparse.eye(2**n)
        H += coeffs[term] * Pi
    return H

def transverse_ising_H(n,g,J=1):
    paulis = np.zeros((2*n-1,n),dtype=int)
    coeffs = [-J]*(n-1) + [-J*g]*n
    for q in range(n-1):
        paulis[q,q:(q+2)] = 3
    for q in range(n):
        paulis[n-1+q,q] = 1
    return hamiltonian(paulis, coeffs)

def ground_state(H, return_eigenvector=False):
    out = eigsh(H,k=1,which='SA', return_eigenvectors=return_eigenvector)
    if return_eigenvector:
        E, psi = out
        E = E[0]
        psi = psi.flatten()
        return E, psi
    else:
        E = out[0]
        return E
