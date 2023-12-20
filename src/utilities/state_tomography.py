import scipy
import numpy as np
import pickle

from src.utilities.path import *
from src.utilities.helpers import theoretical_probs

# Loading POVM and vector data for SIC-POVMs in 2D and 4D spaces.
sic_4d_povm = pickle.load(open(path + "data/povms/sic_4d_povm.p", "rb"))
sic_2d_povm = pickle.load(open(path + "data/povms/sic_2d_povm.p", "rb"))

sic_4d_vectors = pickle.load(open(path + "data/povms/sic_4d_vectors.p", "rb"))
sic_2d_vectors = pickle.load(open(path + "data/povms/sic_2d_vectors.p", "rb"))

# Defining Pauli bases quantum states in 4D and 2D spaces.
states_4d_x = {'x0x0': np.array([[1,1,1,1]])/2,
               'x0x1': np.array([[1,-1,1,-1]])/2,
               'x1x0': np.array([[1,1,-1,-1]])/2,
               'x1x1': np.array([[1,-1,-1,1]])/2}
states_4d_y = {'y0y0': np.array([[1,1.j,1.j,-1]])/2,
               'y0y1': np.array([[1,1.j,-1.j,1]])/2,
               'y1y0': np.array([[1,-1.j,1.j,1]])/2,
               'y1y1': np.array([[1,-1.j,-1.j,-1]])/2}
states_4d_z = {'z0z0': np.array([[1,0,0,0]]),
               'z0z1': np.array([[0,1,0,0]]),
               'z1z0': np.array([[0,0,1,0]]),
               'z1z1': np.array([[0,0,0,1]])}

states_2d_x = {'x0': np.array([[1,1]])/np.sqrt(2),
               'x1': np.array([[1,-1]])/np.sqrt(2)}
states_2d_y = {'y0': np.array([[1,1.j]])/np.sqrt(2),
               'y1': np.array([[1,-1.j]])/np.sqrt(2)}
states_2d_z = {'z0': np.array([[1,0]]),
               'z1': np.array([[0,1]])}

# Defining Pauli bases density matrices in 4D and 2D spaces.
rhos_2d_x = {'x0':states_2d_x['x0'].T.conj()@states_2d_x['x0'],
             'x1':states_2d_x['x1'].T.conj()@states_2d_x['x1']}
rhos_2d_y = {'y0':states_2d_y['y0'].T.conj()@states_2d_y['y0'],
             'y1':states_2d_y['y1'].T.conj()@states_2d_y['y1']}
rhos_2d_z = {'z0':states_2d_z['z0'].T.conj()@states_2d_z['z0'],
             'z1':states_2d_z['z1'].T.conj()@states_2d_z['z1']}

rhos_4d_x = {'x0x0': states_4d_x['x0x0'].T.conj()@states_4d_x['x0x0'],
             'x0x1': states_4d_x['x0x1'].T.conj()@states_4d_x['x0x1'],
             'x1x0': states_4d_x['x1x0'].T.conj()@states_4d_x['x1x0'],
             'x1x1': states_4d_x['x1x1'].T.conj()@states_4d_x['x1x1']}
rhos_4d_y = {'y0y0': states_4d_y['y0y0'].T.conj()@states_4d_y['y0y0'],
             'y0y1': states_4d_y['y0y1'].T.conj()@states_4d_y['y0y1'],
             'y1y0': states_4d_y['y1y0'].T.conj()@states_4d_y['y1y0'],
             'y1y1': states_4d_y['y1y1'].T.conj()@states_4d_y['y1y1']}
rhos_4d_z = {'z0z0': states_4d_z['z0z0'].T.conj()@states_4d_z['z0z0'],
             'z0z1': states_4d_z['z0z1'].T.conj()@states_4d_z['z0z1'],
             'z1z0': states_4d_z['z1z0'].T.conj()@states_4d_z['z1z0'],
             'z1z1': states_4d_z['z1z1'].T.conj()@states_4d_z['z1z1']}

rhos_2d_z = {'z0': states_2d_z['z0'].T.conj()@states_2d_z['z0'],
             'z1': states_2d_z['z1'].T.conj()@states_2d_z['z1']}

# Defining expected probabilities for 4D and 2D states using SIC POVMs
theoretical_probs_4d = {'binary_tree': 
                        {'z0z0': theoretical_probs(sic_4d_povm, states_4d_z['z0z0'], binary_tree=True),
                         'z0z1': theoretical_probs(sic_4d_povm, states_4d_z['z0z1'], binary_tree=True),
                         'z1z0': theoretical_probs(sic_4d_povm, states_4d_z['z1z0'], binary_tree=True),
                         'z1z1': theoretical_probs(sic_4d_povm, states_4d_z['z1z1'], binary_tree=True)},
                        'naimark':
                        {'z0z0': theoretical_probs(sic_4d_povm, states_4d_z['z0z0']),
                         'z0z1': theoretical_probs(sic_4d_povm, states_4d_z['z0z1']),
                         'z1z0': theoretical_probs(sic_4d_povm, states_4d_z['z1z0']),
                         'z1z1': theoretical_probs(sic_4d_povm, states_4d_z['z1z1'])}
                        }
theoretical_probs_2d = {'binary_tree': 
                        {'z0': theoretical_probs(sic_2d_povm, states_2d_z['z0'], binary_tree=True),
                         'z1': theoretical_probs(sic_2d_povm, states_2d_z['z1'], binary_tree=True)},
                        'naimark':
                        {'z0': theoretical_probs(sic_2d_povm, states_2d_z['z0']),
                         'z1': theoretical_probs(sic_2d_povm, states_2d_z['z1'])}
                        }

# Defining Pauli matrices for one-qubit and two-qubit systems.
one_qubit_paulis_dict = {'I': np.array([[1,0],[0,1]], dtype='complex128'),
                         'X': np.array([[0,1],[1,0]], dtype='complex128'),
                         'Y': np.array([[0,-1j],[1j,0]], dtype='complex128'),
                         'Z': np.array([[1,0],[0,-1]], dtype='complex128')}
one_qubit_paulis_list = [one_qubit_paulis_dict['I'], 
                         one_qubit_paulis_dict['X'], 
                         one_qubit_paulis_dict['Y'], 
                         one_qubit_paulis_dict['Z']]

# Defining labels, eigenstates, and density matrices for one-qubit and two-qubit systems based on Pauli states.
two_qubit_paulis_dict = {}
two_qubit_paulis_list = []
for key1 in ['I','X','Y','Z']:
    for key2 in ['I','X','Y','Z']:
        two_qubit_paulis_dict[key1+key2] = np.kron(one_qubit_paulis_dict[key1], one_qubit_paulis_dict[key2])
        two_qubit_paulis_list.append(np.kron(one_qubit_paulis_dict[key1], one_qubit_paulis_dict[key2]))

one_qubit_pauli_states_labels = ['x0','x1','y0','y1','z0','z1']
two_qubit_pauli_states_labels = ['x0x0','x0x1','x0y0','x0y1','x0z0','x0z1',
                                 'x1x0','x1x1','x1y0','x1y1','x1z0','x1z1',
                                 'y0x0','y0x1','y0y0','y0y1','y0z0','y0z1',
                                 'y1x0','y1x1','y1y0','y1y1','y1z0','y1z1',
                                 'z0x0','z0x1','z0y0','z0y1','z0z0','z0z1',
                                 'z1x0','z1x1','z1y0','z1y1','z1z0','z1z1']

one_qubit_pauli_states_list = [states_2d_x['x0'],states_2d_x['x1'],
                         states_2d_y['y0'],states_2d_y['y1'],
                         states_2d_z['z0'],states_2d_z['z1']]
one_qubit_pauli_states_dict = {**states_2d_x, **states_2d_y, **states_2d_z}
one_qubit_pauli_rhos_dict = {**rhos_2d_x, **rhos_2d_y, **rhos_2d_z}

two_qubit_pauli_states_list = []
two_qubit_pauli_states_dict = {}
two_qubit_pauli_rhos_dict = {}
for s1_label in one_qubit_pauli_states_labels:
    for s2_label in one_qubit_pauli_states_labels:
        s1 = one_qubit_pauli_states_dict[s1_label]
        s2 = one_qubit_pauli_states_dict[s2_label]
        state = np.kron(s2,s1)
        two_qubit_pauli_states_dict[s1_label+s2_label] = state
        two_qubit_pauli_states_list.append(state)  
        two_qubit_pauli_rhos_dict[s1_label+s2_label] = state.T.conj()@state

# Defining Mutually Unbiased Bases (MUBs) for two-qubit states.
m0 = [np.array([[1,0,0,0]]), 
      np.array([[0,1,0,0]]), 
      np.array([[0,0,1,0]]), 
      np.array([[0,0,0,1]])]
m1 = [np.array([[1,1,1,1]])/2, 
      np.array([[1,1,-1,-1]])/2, 
      np.array([[1,-1,-1,1]])/2, 
      np.array([[1,-1,1,-1]])/2]
m2 = [np.array([[1,-1,-1j,-1j]])/2,
      np.array([[1,-1,1j,1j]])/2,
      np.array([[1,1,1j,-1j]])/2,
      np.array([[1,-1,-1j,1j]])/2]
m3 = [np.array([[1,-1j,-1j,-1]])/2,
      np.array([[1,-1j,1j,1]])/2,
      np.array([[1,1j,1j,-1]])/2,
      np.array([[1,1j,-1j,1]])/2]
m4 = [np.array([[1,-1j,-1,-1j]])/2,
      np.array([[1,-1j,1,1j]])/2,
      np.array([[1,1j,-1,1j]])/2,
      np.array([[1,1j,1,-1j]])/2]

two_qubit_mub_states_labels = ['m00','m01','m02','m03',
                               'm10','m11','m12','m13',
                               'm20','m21','m22','m23',
                               'm30','m31','m32','m33',
                               'm40','m41','m42','m43']
two_qubit_mub_states_list = m0+m1+m2+m3+m4

def vectorize(M):
    '''Vectorize matrix M by stacking the columns of the matrix on top of one another.
    
    Parameters:
    M (np.array): A matrix to be vectorized.
    
    Returns:
    np.array: A vectorized form of the input matrix.
    '''
    return np.array([M.T.flatten()], dtype='complex128').T

def unvectorize(v):
    '''Undo the vectorize function, converting a vector back into a matrix.
    
    Parameters:
    v (np.array): A vector to be converted back to a square matrix.
    
    Returns:
    np.array: A square matrix form of the input vector.
    '''
    d = int(np.log2(len(v)))
    return v.reshape((d,d)).T

def counts_to_probs_array(counts, binary_tree=False):
    '''Convert a dictionary of counts to an array of probabilities.
    
    Parameters:
    counts (dict): A dictionary where keys are states and values are counts.
    binary_tree (bool): Whether to use binary tree indexing for the probabilities.
    
    Returns:
    np.array: An array of probabilities.
    '''
    shots = 0
    for key in counts.keys():
        shots += counts[key]
    n_elements = 2**len(list(counts.keys())[0])
    p = np.zeros((n_elements))
    for key in counts.keys():
        if binary_tree:
            p[int(key[::-1],2)] = counts[key]/shots
        else: 
            p[int(key,2)] = counts[key]/shots
    return p

def qst_lininv(povm, counts, binary_tree=False):
    '''Perform quantum state tomography using linear inversion method.
    
    Parameters:
    povm (list): A list of POVM elements.
    counts (dict): A dictionary of counts.
    binary_tree (bool): If true, uses binary tree indexing for probabilities.
    
    Returns:
    np.array: The reconstructed density matrix.
    '''
    N = len(povm)
    # frame operator
    S = np.zeros((N,N), dtype='complex128')
    for i in range(N):
        v = vectorize(povm[i])
        S += v@v.T.conj()
    # S is invertible if <povm> is informationally complete
    Sinv = np.linalg.inv(S)
    p = counts_to_probs_array(counts, binary_tree)
    rho_v = np.zeros((N,1), dtype='complex128')
    for i in range(N):
        rho_v += p[i]*Sinv@vectorize(povm[i])
    return unvectorize(rho_v)

def qst_psd(povm, counts, binary_tree=False):
    '''Perform quantum state tomography ensuring the output density matrix is positive semi-definite.
    
    Parameters:
    povm (list): A list of POVM elements.
    counts (dict): A dictionary of counts.
    binary_tree (bool): If true, uses binary tree indexing for probabilities.
    
    Returns:
    np.array: The reconstructed positive semi-definite density matrix.
    
    Refer to this work by John A. Smolin et al. for details:
    [1] https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.070502
    '''
    rho_lininv = qst_lininv(povm, counts, binary_tree=binary_tree)
    return make_psd(rho_lininv)

def make_psd(rho):
    '''Transform a density matrix into a positive semi-definite matrix.
    
    Parameters:
    rho (np.array): A density matrix.
    
    Returns:
    np.array: A positive semi-definite matrix.

    Refer to this work by John A. Smolin et al. for details:
    [1] https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.070502
    '''
    dim = len(rho)
    w,v = np.linalg.eig(rho)
    # sort eigenvalues/vectors in descending order
    w_sorted = np.array(sorted(w, reverse=True))
    v_sorted = np.array([vi for _, vi in sorted(zip(w,v.T), reverse=True)])

    w_rescaled = np.zeros_like(w_sorted)
    i = dim-1
    a = 0
    while True:
        if (w_sorted[i] + a/(i+1))>=0:
            for j in range(i+1):
                w_rescaled[j] = w_sorted[j] + a/(i+1)
            break
        else:
            w_rescaled[i] = 0
            a += w_sorted[i]
            i -= 1
    # construct density matrix from rescaled eigenvalues
    rho_psd = np.zeros_like(rho)
    for i in range(dim):
        vi = np.array([v_sorted[i]])
        rho_psd += w_rescaled[i]*vi.T@vi.conj()
    return rho_psd

def detector_tomography_2q(counts, binary_tree=False, two_qubit_states=None):
    '''Perform detector tomography for two-qubit systems.
    
    Parameters:
    counts (list): A list of count dictionaries for different states.
    binary_tree (bool): If true, uses binary tree indexing for probabilities.
    two_qubit_states (list): Optional list of two-qubit states for tomography.
    
    Returns:
    list: A list of reconstructed POVM elements.
    '''
    probs = [counts_to_probs_array(c, binary_tree=binary_tree) for c in counts]

    if two_qubit_states is None:
        # Create a list of two qubit states which are products of Pauli basis states
        two_qubit_states = two_qubit_pauli_states_list

    S = np.zeros((len(two_qubit_states),len(two_qubit_paulis_list)), dtype='complex128')
    for i in range(len(two_qubit_states)):
        for j in range(len(two_qubit_paulis_list)):
            rho = two_qubit_states[i].T.conj()@two_qubit_states[i]
            P = two_qubit_paulis_list[j]
            S[i,j] = np.trace(P@rho)

    # group together the probabilities corresponding to the same POVM element
    probs_m = [[] for _ in range(len(sic_4d_povm))]
    for i in range(len(two_qubit_states)):
        for m in range(len(sic_4d_povm)):
            probs_m[m].append(probs[i][m])

    # get the least-squares estimate for every POVM element
    S_ls = np.linalg.inv(S.T.conj()@S)@S.T.conj()
    povm = []
    for i in range(len(sic_4d_povm)):
        a = S_ls@probs_m[i]
        F = np.zeros((4,4), dtype='complex128')
        for j in range(len(two_qubit_paulis_list)):
            F += a[j]*two_qubit_paulis_list[j]
        povm.append(F)

    return povm

def detector_tomography_1q(counts, binary_tree=False, one_qubit_states=None):
    '''Perform detector tomography for one-qubit systems.
    
    Parameters:
    counts (list): A list of count dictionaries for different states.
    binary_tree (bool): If true, uses binary tree indexing for probabilities.
    one_qubit_states (list): Optional list of one-qubit states for tomography.
    
    Returns:
    list: A list of reconstructed POVM elements.
    '''
    probs = [counts_to_probs_array(c, binary_tree=binary_tree) for c in counts]

    if one_qubit_states is None:
        # Create a list of one qubit states which are Pauli basis states
        one_qubit_states = one_qubit_pauli_states_list

    S = np.zeros((len(one_qubit_states),len(one_qubit_paulis_list)), dtype='complex128')
    for i in range(len(one_qubit_states)):
        for j in range(len(one_qubit_paulis_list)):
            rho = one_qubit_states[i].T.conj()@one_qubit_states[i]
            P = one_qubit_paulis_list[j]
            S[i,j] = np.trace(P@rho)

    # group together the probabilities corresponding to the same POVM element
    probs_m = [[] for _ in range(len(sic_2d_povm))]
    for i in range(len(one_qubit_states)):
        for m in range(len(sic_2d_povm)):
            probs_m[m].append(probs[i][m])

    # get the least-squares estimate for every POVM element
    S_ls = np.linalg.inv(S.T.conj()@S)@S.T.conj()
    povm = []
    for i in range(len(sic_2d_povm)):
        a = S_ls@probs_m[i]
        F = np.zeros((2,2), dtype='complex128')
        for j in range(len(one_qubit_paulis_list)):
            F += a[j]*one_qubit_paulis_list[j]
        povm.append(make_psd(F))

    return povm