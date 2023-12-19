import numpy as np
import scipy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.extensions import UnitaryGate
from scipy.linalg import sqrtm, svd

### NOTE: Please be aware that certain functions within this module were specifically designed for targeted tasks over the course of the project. As a result, there may be instances of redundancy or repetition in their functionality.

def closest_unitary(A):
    '''
    Calculate the unitary matrix U that is closest with respect to the operator norm distance to the general matrix A.

    Parameters:
    A (np.array): A numpy array representing a general matrix.

    Returns:
    np.matrix: The closest unitary matrix to A.
    '''
    V, _, Wh = scipy.linalg.svd(A)
    U = np.matrix(V.dot(Wh))
    return U

def get_naimark_unitary_from_vectors(vecs, assert_unitarty=True, atol=1e-05):
    '''
    Construct a Naimark unitary matrix from a set of vectors.

    Parameters:
    vecs (list of np.array): A list of vectors for which the Naimark unitary is to be constructed.
    assert_unitarty (bool): If True, asserts that the constructed matrix is unitary.
    atol (float): Absolute tolerance for the unitarity assertion.

    Returns:
    np.matrix: The constructed Naimark unitary matrix.
    '''
    A = np.concatenate(([psi.conj() for psi in vecs]), axis=0)
    y = scipy.linalg.null_space(A.T.conj())
    U = np.concatenate((A,y),axis=1)
    if assert_unitarty:
        assert np.allclose(U.T.conj()@U, np.eye(len(vecs)),atol=atol), "Failed to construct U"
    return closest_unitary(U)

def get_naimark_circuit_from_vectors(vecs):
    '''
    Create a quantum circuit realizing Naimark's dilation from a set of vectors which correspond to rank-one POVM elements.

    Parameters:
    vecs (list of np.array): A list of vectors used to create the Naimark circuit.

    Returns:
    QuantumCircuit: The quantum circuit implementing the Naimark extension.
    '''
    _,d = vecs[0].shape
    n_qubits = np.log2(d)
    N = len(vecs)
    L = np.log2(N)

    system = QuantumRegister(n_qubits, name='system')
    anc = QuantumRegister(L-n_qubits,name='ancilla')
    qc = QuantumCircuit(system, anc, name="measurement-circuit")

    U = get_naimark_unitary_from_vectors(vecs)
    U_gate = UnitaryGate(U, label='U')
    qc.append(U_gate, system[:] + anc[:])
    qc.measure_all()
    return qc


def check_for_rank_one(povm):
    '''
    Check if a POVM is a rank-1 POVM.

    Parameters:
    povm (list of np.array): A list of POVM elements.

    Returns:
    bool: True if the POVM is rank-1, False otherwise.
    '''
    rank_one = True
    for p in povm:
        if np.linalg.matrix_rank(p)!=1:
            rank_one = False
            return rank_one
        else:
            continue
    return rank_one
            
def check_symmetric(p, rtol=1e-05, atol=1e-08):
    '''
    Check if a matrix is symmetric.

    Parameters:
    p (np.array): The matrix to be checked.
    rtol (float): Relative tolerance.
    atol (float): Absolute tolerance.

    Returns:
    bool: True if the matrix is symmetric, False otherwise.
    '''
    return np.allclose(p, p.T, rtol=rtol, atol=atol)

def povm_isometry(povm):
    '''
    Construct an isometry matrix from a rank-1 POVM.

    Parameters:
    povm (list of np.array): A list of POVM elements that are rank-1.

    Returns:
    np.array: The isometry matrix constructed from the POVM.
    '''
    assert check_for_rank_one(povm), "This is not a rank-1 povm"
    new_povm = []
    
    for p in povm:
        
        Eigenvalues, Eigenvectors = np.linalg.eig(p)

        non_zero_index = np.argwhere(~np.isclose(Eigenvalues, 0))[0]

        Lambda = Eigenvalues[non_zero_index]
        v = Eigenvectors[:, non_zero_index]
        
        psi = np.sqrt(Lambda) * v   # this is psi such that the povm element p = |psi> <psi|
        
        new_povm.append(psi)
        
        
    v = np.hstack(new_povm) # arrange the povm elements to form a matrix of dimension: Row X Col*len(povm)

    v = np.atleast_2d(v) # convert to 2d matrix
    
    return v

def compute_rank_one_unitary(povm):
    '''
    Create a coupling unitary between the system and ancilla using eigenvectors and eigenvalues of the POVM elements. 
    The matrix A is extended to unitary U by appending a basis of the null space of A.

    Parameters:
    povm (list of np.array): A list of POVM elements.

    Returns:
    np.array: The coupling unitary matrix.
    '''
    A = povm_isometry(povm)
    A = A.T
    
    U, Q = scipy.linalg.qr(A)
    
    # check to confirm that U is close to unitary
    assert np.allclose(U.T.conj()@U, np.eye(len(A)),atol=1e-03), "Failed to construct U"
    return U

def compute_full_rank_unitary(povm, atol=1e-13, rtol=0):
    '''
    Compute the unitary that rotates the system to the Hilbert space of the ancilla for a full-rank POVM.

    Parameters:
    povm (list of np.array): A list of POVM elements.
    atol (float): Absolute tolerance.
    rtol (float): Relative tolerance.

    Returns:
    np.array: The computed unitary matrix.
    '''
    # Here square root of the POVM elements were used as a replacement for the vector that form the povm
    povm = [sqrtm(M)for M in povm]
    
    v = np.hstack(povm) # arrange the povm elements to form a matrix of dimension: Row X Col*len(povm)
    v = np.atleast_2d(v) # convert to 2d matrix
    u, s, vh = svd(v)    # apply svd
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:]         # missing rows of v
    
    # add the missing rows of v to v
    V = np.vstack((v, ns)) 
    
    
    # make the unitary a square matrix of dimension N=2^n where n = int(np.ceil(np.log(V.shape[0])))
    n = int(np.ceil(np.log2(V.shape[0])))
    N = 2**n      # dimension of system and ancilla Hilber space
    r,c = V.shape  
    
    U = np.eye(N, dtype=complex) # initialize Unitary matrix to the identity. Ensure it is complex
    U[:r,:c] = V[:r,:c] # assign all the elements of V to the corresponding elements of U
    
    U = U.conj().T  # Transpose the unitary so that the rows are the povm
    
    # check for unitarity of U
    assert np.allclose(U.T.conj()@U, np.eye(N),atol=1e-07), "Failed to construct U"
    
    return U

def rank_one_circuit(povm, state, U):
    '''
    Define and create the quantum circuit for a rank-one POVM and a given state.

    Parameters:
    povm (list of np.array): A list of rank-one POVM elements.
    state (np.array): The quantum state to be measured.
    U (np.array): The unitary matrix for the system and ancilla.

    Returns:
    QuantumCircuit: The constructed quantum circuit.
    '''
    # Define the quantum and classical registers
    dim_system = state.shape[1] # dimension of state
    num_system_qubit = int(np.ceil(np.log2(dim_system))) # total number of qubits for system

    system_reg = QuantumRegister(num_system_qubit, name='system') # system register

    N = U.shape[0] # Dimension of the unitary to be applied to system and ancilla

    num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit # total number of qubits for system

    ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla') # ancilla register

    U_gate = UnitaryGate(U, label='U') # unitary gate to be applied between system and ancilla
    
    
    # create the quantum circuit for the system and ancilla
    qc = QuantumCircuit(system_reg, ancilla_reg, name='circuit')
    qc.initialize(state[0],system_reg)

    # reset ancilla to zero
    qc.reset(ancilla_reg)

    # append the unitary gate
    qc.append(U_gate, range(system_reg.size + ancilla_reg.size))

    # measure only the ancilliary qubits
    qc.measure_all()
    
    return qc

def full_rank_circuit(povm, state, U):
    '''
    Define and create the quantum circuit for a full-rank POVM and a given state.

    Parameters:
    povm (list of np.array): A list of POVM elements.
    state (np.array): The quantum state to be measured.
    U (np.array): The unitary matrix for the system and ancilla.

    Returns:
    QuantumCircuit: The constructed quantum circuit.
    '''
    # Define the quantum and classical registers
    dim_system = state.shape[1] # dimension of state
    num_system_qubit = int(np.ceil(np.log2(dim_system))) # total number of qubits for system

    system_reg = QuantumRegister(num_system_qubit, name='system') # system register

    N = U.shape[0] # Dimension of the unitary to be applied to system and ancilla

    num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit # total number of qubits for system

    ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla') # ancilla register

    classical_reg = ClassicalRegister(num_ancilla_qubit, name='measure') # classical register

    U_gate = UnitaryGate(U, label='U') # unitary gate to be applied between system and ancilla
    
    
    # create the quantum circuit for the system and ancilla
    qc = QuantumCircuit(system_reg, ancilla_reg, classical_reg, name='circuit')
    qc.initialize(state[0],system_reg)

    # reset ancilla to zero
    qc.reset(ancilla_reg)

    # append the unitary gate
    qc.append(U_gate, range(system_reg.size + ancilla_reg.size))

    # measure only the ancilliary qubits
    qc.measure(ancilla_reg, classical_reg)
    
    return qc

def construct_quantum_circuit(povm, state):
    '''
    Construct a quantum circuit based on a given POVM and state.

    Parameters:
    povm (list of np.array): A list of POVM elements.
    state (np.array): The quantum state for which the circuit is constructed.

    Returns:
    QuantumCircuit: The resulting quantum circuit.
    '''
    if check_for_rank_one(povm):
        U = compute_rank_one_unitary(povm)
        qc = rank_one_circuit(povm, state, U)
    else:
        U = compute_full_rank_unitary(povm)
        qc = full_rank_circuit(povm, state, U)
    
    return qc
