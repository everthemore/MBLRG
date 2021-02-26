import numpy as np
from scipy.sparse import diags, spmatrix, linalg, save_npz, lil_matrix
from math import factorial

def binary_conv(x, L):
	'''
	convert base-10 integer to binary and adds zeros to match length
	returns: Binary
	'''
	b = bin(x).split('b')[1]
	while len(b) < L:
		b = '0'+b
	return b

def count_ones(bitstring):
    return np.sum([1 for a in bitstring if a == '1'])

def binomial(n, k='half'):
	'''
	find binomial coefficient of n pick k,
	returns interger
	'''
	if k=='half':
		k = n//2
	return int(factorial(n)/factorial(k)**2)

def energy_diag(bitString, V, U):
	E = 0
	for index, i in enumerate(bitString):
		if i =='1':
			E += V[index]
			try:
				if bitString[index+1] == '1':
					E += U[index]

			except IndexError:
				continue
	return E

def construct_basis(L, n=0):
    s2i = {} # State_to_index
    i2s = {} # index_to_State

    index = 0
    for i in range(int(2**L)): # We could insert a minimum
        binary = binary_conv(i, L)

        if n != 0:
            ones = count_ones(binary)
            if ones == n:
            	s2i[binary] = index
            	i2s[i] = binary
            	index +=1
        else:
            s2i[binary] = index
            i2s[i] = binary
            index +=1

    return s2i, i2s

def construct_hamiltonian(onsite_coeff, nn_coeff, hopping_coeff):
    L = len(onsite_coeff)
    s2i, i2s = construct_basis(L, L//2)
    num_states = len(s2i)

    H = lil_matrix((num_states,num_states))

    for key in s2i.keys():

        E = energy_diag(key, onsite_coeff, nn_coeff)
        H[s2i[key],s2i[key]] = E

        for site in range(L):
            print(site)
            try:
                if (key[site] == '1' and key[site+1] == '0'):
                    new_state = key[:site] + '0' + '1' + key[site+2:]
                    H[s2i[key],s2i[new_state]] = hopping_coeff[site]
                    H[s2i[new_state],s2i[key]] = np.conjugate(hopping_coeff[site])

            except IndexError: # periodic boundary conditions
                continue
    return H

def compute_eigenvalues(onsite_coeff, nn_coeff, hopping_coeff):
    H = construct_hamiltonian(onsite_coeff, nn_coeff, hopping_coeff)
    evals, evecs = np.linalg.eigh(H.todense())
    return evals
