## Copyright Eliott Rosenberg, 2022

from itertools import combinations
from math import comb
from scipy.sparse import coo_array

import numpy as np
from copy import deepcopy
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os


def number_operator(N,L,i,excitation_vectors=[],truncate=False):
    if len(excitation_vectors) == 0:
        excitation_vectors = oscillator_states(N,L)
    d = comb(N+L-1,N)
    data = []
    for state_index in tqdm(range(d)):
        state = excitation_vectors[state_index]
        if truncate:
            data.append( min( 2, state[i] ) )
        else:
            data.append(state[i])
    return coo_array((data,(range(d),range(d))),shape=(d,d))


def rect_lattice_neighbors(i,Lx,Ly):
    x = i % Lx
    y = i // Lx
    neighbors = []
    if x != 0:
        neighbors.append(i-1)
    if x != Lx-1:
        neighbors.append(i+1)
    if y != 0:
        neighbors.append(i-Lx)
    if y != Ly-1:
        neighbors.append(i+Lx)
    return neighbors
        

def bose_hubbard_hamiltonian(N,Lx,Ly,neighbors,t,U,mu=0,checkerboard=True,excitation_vectors=[],inv_lookup=[]):
    # neighbors is a function such that that neighbors(s) returns the neighbors of site s.
    L = Lx*Ly
    if len(excitation_vectors) == 0:
        excitation_vectors = oscillator_states(N,L)
    if len(inv_lookup)==0:
        inv_lookup = inverse_dictionary(excitation_vectors)
    d = comb(N+L-1,N)
    i0 = []
    i1 = []
    data = []
    
    #for i in tqdm(range(d)):
    for i in range(d):
        state = np.array(excitation_vectors[i])
        
        # the U and mu terms:
        if checkerboard:
            checkerboard_signs = np.array([ (1 - 2*((_%Lx)%2))*( 1 - 2*((_//Ly)%2)) for _ in range(L)   ])
            diag_contribution = U/2 * np.sum( state**2 - state) + mu * checkerboard_signs @ state
        else:
            diag_contribution = U/2 * np.sum( state**2 - state) - mu*N
        i0.append(i)
        i1.append(i)
        data.append(diag_contribution)
        
        # the t terms:
        for site in range(L):
            if state[site] == 0:
                continue
            for site2 in neighbors(site):
                new_state = deepcopy(state)
                new_state[site] -= 1
                new_state[site2] += 1
                new_state = tuple(new_state)
                i1.append(i)
                #i0.append( excitation_vectors.index(new_state) )
                i0.append( inv_lookup[str(new_state)])
                data.append(-t*np.sqrt(state[site]*new_state[site2]))
    
    
    H = coo_array((data,(i0,i1)),shape=(d,d))
    return H

def oscillator_states(N,L):
    # L is number of sites, N is number of excitations
    # Enumerate using the representation on p. 55 of Schroeder's Thermal Physics
    # we have N + L - 1 symbols and choose N of them to be dots (the remaining are lines)
    
    excitation_vectors = [ dot_locs_to_excitations_vector(which_dots,L) for which_dots in combinations(range(N+L-1),N)]
    
    return excitation_vectors

def inverse_dictionary(excitation_vectors):
    return { str(excitation_vectors[_]):_ for _ in range(len(excitation_vectors))  }

def dot_locs_to_excitations_vector(which_dots,L):
    #L is number of sites, N is number of excitations
    # Enumerate using the representation on p. 55 of Schroeder's Thermal Physics
    # we have N + L - 1 symbols and choose N of them to be dots (the remaining are lines)
    N = len(which_dots)
    num_symbols = N+L-1
    excitations_vector = np.zeros(L,dtype=np.uint)
    site = 0
    for symbol in range(num_symbols):
        if symbol in which_dots:
            excitations_vector[site] += 1
        else:
            site += 1
    return tuple(excitations_vector)
