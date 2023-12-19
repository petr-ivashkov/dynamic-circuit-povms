from qiskit.quantum_info import Operator, state_fidelity
from qiskit import transpile
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

import numpy as np
import itertools
import pickle
import scipy

from src.utilities.path import *

sic_4d_povm = pickle.load(open(path + "data/povms/sic_4d_povm.p", "rb"))

def fidelity(rho1, rho2):
    '''
    Calculate Uhlmann's fidelity between two quantum states.

    Parameters:
    rho1 (np.array): Density matrix of the first quantum state.
    rho2 (np.array): Density matrix of the second quantum state.

    Returns:
    float: The fidelity value between the two states.
    '''
    sqrt_rho1 = scipy.linalg.sqrtm(rho1)
    x = np.trace(scipy.linalg.sqrtm(sqrt_rho1@rho2@sqrt_rho1))
    return (x**2).real

def are_equivalent(qc1, qc2, atol=10**-5):
    '''
    Determine if two quantum circuits are equivalent within a given tolerance.

    Parameters:
    qc1 (QuantumCircuit): The first quantum circuit.
    qc2 (QuantumCircuit): The second quantum circuit.
    atol (float): Absolute tolerance for the equivalence check.

    Returns:
    bool: True if circuits are equivalent within the specified tolerance, False otherwise.
    '''
    op1 = Operator(qc1)
    op2 = Operator(qc2)
    return op1.equiv(op2, atol=atol)

def find_missing_swaps_3q(qc, qc_true, return_circuit=False):
    '''
    Find a sequence of SWAPs to insert before and after a 3-qubit quantum circuit to make it equivalent to a target circuit.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    qc_true (QuantumCircuit): The target quantum circuit for comparison.
    return_circuit (bool): If True, return the modified circuit as well.

    Returns:
    list or tuple: A list of swap configurations or a tuple containing the modified circuit and swap configurations.
    '''
    l = [0,1]
    for init_01, init_02, init_12, final_01, final_02, final_12, in itertools.product(l,l,l,l,l,l):
        swap_init = QuantumCircuit(3)
        if init_01:
            swap_init.swap(0,1)
        if init_02:
            swap_init.swap(0,2)
        if init_12:
            swap_init.swap(1,2)
        swap_final = QuantumCircuit(3)
        if final_01: 
            swap_final.swap(0,1)
        if final_02:
            swap_final.swap(0,2)
        if final_12:
            swap_final.swap(1,2)
        qc0 = qc.copy()
        qc0.remove_final_measurements()
        qc_test = swap_init.compose(qc0)
        qc_test.compose(swap_final, inplace=True)
        if are_equivalent(qc_test, qc_true):
            if return_circuit:
                return qc_test, [(init_01, init_02, init_12),(final_01, final_02, final_12)]
            return [(init_01, init_02, init_12),(final_01, final_02, final_12)]
    raise Exception('No permutation found to transform <qc> into <qc_true>.')

def initialize(qc, state: str, initial_layout=None):
    '''
    Initialize a quantum circuit to one of the specified 1 or 2 qubit basis states.

    Parameters:
    qc (QuantumCircuit): Quantum circuit to be initialized.
    state (str): The desired basis state to initialize the circuit.
    initial_layout (list): Optional initial layout of qubits in the circuit.

    Returns:
    QuantumCircuit: The quantum circuit after initialization.
    '''
    assert state in ['x0','x1','y0','y1','z0','z1',
                     'x0x0','x0x1','x0y0','x0y1','x0z0','x0z1',
                     'x1x0','x1x1','x1y0','x1y1','x1z0','x1z1',
                     'y0x0','y0x1','y0y0','y0y1','y0z0','y0z1',
                     'y1x0','y1x1','y1y0','y1y1','y1z0','y1z1',
                     'z0x0','z0x1','z0y0','z0y1','z0z0','z0z1',
                     'z1x0','z1x1','z1y0','z1y1','z1z0','z1z1',
                     ], f'{state} is not a supported state!'
    qr = QuantumRegister(qc.num_qubits)
    qc_init = QuantumCircuit(qr)
    if initial_layout is None: initial_layout = [i for i in range(qc.num_qubits)]
    # initialize first qubit
    if state[0] == 'x': 
        qc_init.h(qr[initial_layout[0]])
        if state[1] == '1': 
            qc_init.z(qr[initial_layout[0]])
    if state[0] == 'y': 
        qc_init.h(qr[initial_layout[0]])
        qc_init.s(qr[initial_layout[0]])
        if state[1] == '1': 
            qc_init.z(qr[initial_layout[0]])
    if state[0] == 'z': 
        if state[1] == '1': 
            qc_init.x(qr[initial_layout[0]])

    if len(state)>2:
        # initialize second qubit
        if state[2] == 'x': 
            qc_init.h(qr[initial_layout[1]])
            if state[3] == '1': 
                qc_init.z(qr[initial_layout[1]])
        if state[2] == 'y': 
            qc_init.h(qr[initial_layout[1]])
            qc_init.s(qr[initial_layout[1]])
            if state[3] == '1': 
                qc_init.z(qr[initial_layout[1]])
        if state[2] == 'z': 
            if state[3] == '1': 
                qc_init.x(qr[initial_layout[1]])

    qc_init = transpile(qc_init, 
                       basis_gates=['cx', 'id', 'rz', 'sx', 'x', 'if_else'])

    return qc.compose(qc_init, qc.qubits[:], front=True)

def theoretical_probs(povm, state, binary_tree=False):
    '''
    Calculate the expected probabilities for a given pure state and POVM.

    Parameters:
    povm (list): A list of POVM elements.
    state (np.array): The pure quantum state (ket).
    binary_tree (bool): If True, use binary tree indexing for the probabilities.

    Returns:
    dict: A dictionary of probabilities keyed by measurement outcomes.
    '''
    if len(state)!=1:
        rho = np.array([state]).T.conj()@np.array([state]) # density matrix of the state
    else:
        rho = state.T.conj()@state # density matrix of the state
    probs = [np.trace(povm[i]@rho).real for i in range(len(povm))]

    l = len(probs)
    n = int(np.ceil(np.log2(l)))
    prob_counts = {}
    for i in range(l):
        if binary_tree: key = np.binary_repr(i, n)[::-1]
        else: key = np.binary_repr(i, n)
        value = probs[i]
        prob_counts[key] = value
    return prob_counts

def unitary_distance(u1, u2):
    '''
    Calculate the distance between two unitaries based on Hilbert-Schmidt inner product.

    Parameters:
    u1 (np.array): The first unitary matrix.
    u2 (np.array): The second unitary matrix.

    Returns:
    float: The calculated distance between the two unitaries.
    '''
    assert u1.shape == u2.shape
    num_qubits = int(round(np.log2(u1.shape[0])))
    return 1 - abs(np.trace(u1.T.conj()@u2)/2**num_qubits)

def evolve_kraus_channel(K, rho):
    '''
    Evolve a quantum state through a quantum channel represented by Kraus operators.

    Parameters:
    K (list): List of Kraus operators.
    rho (np.array): The density matrix of the quantum state to be evolved.

    Returns:
    np.array: The evolved density matrix.
    '''
    rho_new = np.zeros_like(rho, dtype='complex128')
    for k in K:
        rho_new += k@rho@k.T.conj()
    return rho_new

def evolve_detector_channel(povm, rho):
    '''
    Evolve a quantum state through a detector channel represented by POVM elements.

    Parameters:
    povm (list): List of POVM elements representing the detector channel.
    rho (np.array): The density matrix of the quantum state to be evolved.

    Returns:
    np.array: The evolved density matrix.
    '''
    N = len(povm)
    rho_new = np.eye(N, dtype='complex128')
    for i in range(N):
        rho_new[i,i] = np.trace(povm[i]@rho)
    return rho_new

def get_choi_matrix_from_povm(povm):
    '''
    Convert a set of POVM elements to the corresponding (unnormalized) Choi matrix.

    Parameters:
    povm (list): List of POVM elements.

    Returns:
    np.array: The corresponding Choi matrix.
    '''
    d = len(povm[0])
    N = len(povm)
    choi = np.zeros((N*d, N*d), dtype='complex128')
    for i in range(d):
        for j in range(d):
            p = np.zeros((d,d))
            p[i,j] = 1
            choi += np.kron(p, evolve_detector_channel(povm, p))
    return choi

def get_normalized_choi_matrix_from_povm(povm):
    '''
    Convert a set of POVM elements to the corresponding normalized Choi matrix.

    Parameters:
    povm (list): List of POVM elements.

    Returns:
    np.array: The corresponding normalized Choi matrix.
    '''
    choi = get_choi_matrix_from_povm(povm)
    return choi / np.trace(choi)

def get_choi_matrix_from_kraus(K):
    '''
    Convert a set of Kraus operators to the corresponding (unnormalized) Choi matrix.

    Parameters:
    K (list): List of Kraus operators.

    Returns:
    np.array: The corresponding Choi matrix.
    '''
    d = len(K[0])
    choi = np.zeros((d*d, d*d), dtype='complex128')
    for i in range(d):
        for j in range(d):
            p = np.zeros((d,d))
            p[i,j] = 1
            choi += np.kron(p, evolve_kraus_channel(K, p))
    return choi

def get_normalized_choi_matrix_from_kraus(K):
    '''
    Convert a set of Kraus operators to the corresponding normalized Choi matrix.

    Parameters:
    K (list): List of Kraus operators.

    Returns:
    np.array: The corresponding normalized Choi matrix.
    '''
    choi = get_choi_matrix_from_kraus(K)
    d = int(np.sqrt(len(choi)))
    return choi / d

def get_povm_from_binary_aqc_circuits(aqc_circuits):
    '''
    Construct two-qubit POVM elements from a set of binary adaptive quantum circuits.

    Parameters:
    aqc_circuits (dict): A dictionary of quantum circuits keyed by binary strings.

    Returns:
    list: List of POVM elements.
    '''
    bins = ['0000','0001','0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']
    povm = [None]*16
    for s in bins:
        s1 = s[3]
        s2 = s[2]
        s3 = s[1]
        s4 = s[0]
        
        A = np.eye(4)
        U3 = Operator(aqc_circuits[s3 + s2 + s1 + '0']).data
        U2 = Operator(aqc_circuits[s2 + s1 + '0']).data
        U1 = Operator(aqc_circuits[s1 + '0']).data
        U0 = Operator(aqc_circuits['0']).data
        if s4 == '0': A = A@U3[:4, :4]
        else: A = A@U3[4:8, :4]
        if s3 == '0': A = A@U2[:4, :4]
        else: A = A@U2[4:8, :4]
        if s2 == '0': A = A@U1[:4, :4]
        else: A = A@U1[4:8, :4]
        if s1 == '0': A = A@U0[:4, :4]
        else: A = A@U0[4:8, :4]
        povm[int(s[::-1],2)] = A.T@A.conj()
    return povm

def get_povm_fidelity_from_binary_aqc_circuits(aqc_circuits):
    '''
    Calculate the fidelity of POVM elements constructed from binary adaptive quantum circuits.

    Parameters:
    aqc_circuits (dict): A dictionary of quantum circuits keyed by binary strings.

    Returns:
    float: Fidelity of the reconstructed POVM.
    '''
    choi = get_normalized_choi_matrix_from_povm(sic_4d_povm)
    povm = get_povm_from_binary_aqc_circuits(aqc_circuits)
    choi_appx = get_normalized_choi_matrix_from_povm(povm)
    return state_fidelity(choi, choi_appx)

def get_povm_from_naimark_aqc_circuits(aqc_circuits):
    '''
    Construct POVM elements from the Naimark quantum circuit.

    Parameters:
    aqc_circuits (dict): A dictionary of quantum circuits (only one expected).

    Returns:
    list: List of POVM elements.
    '''
    def get_povm_from_naimark_unitary(U):
        # Reconstruct the POVM from Naimark circuit unitary
        povm = []
        for i in range(16):
            v = np.array([U[i][:4]])
            povm.append(v.T@v.conj())
        return povm
    U = Operator(aqc_circuits['0']).data
    povm = get_povm_from_naimark_unitary(U)
    return povm

def get_povm_fidelity_from_naimark_aqc_circuits(aqc_circuits):
    '''
    Calculate the fidelity of POVM elements constructed from the Naimark quantum circuit.

    Parameters:
    aqc_circuits (dict): A dictionary of quantum circuits (only one expected).

    Returns:
    float: Fidelity of the reconstructed POVM.
    '''
    choi = get_normalized_choi_matrix_from_povm(sic_4d_povm)
    povm = get_povm_from_naimark_aqc_circuits(aqc_circuits)
    choi_appx = get_normalized_choi_matrix_from_povm(povm)
    return state_fidelity(choi, choi_appx)

def get_povm_from_hybrid_aqc_circuits(aqc_circuits):
    '''
    Construct POVM elements from hybrid adaptive quantum circuits.

    Parameters:
    aqc_circuits (dict): A dictionary of quantum circuits.

    Returns:
    list: List of POVM elements.
    '''
    def get_povm_from_hybrid_unitaries(U0, U00, U10):
        # Reconstruct the POVM from hybrid circuit unitaries
        povm = []
        B0 = U0[:4,:4]
        B1 = U0[4:8,:4]    
        arr = np.vstack([U00, U10])
        for i in range(16):
            v = np.array([arr[i][:4]])
            if i < 8: v_tilde = B0.T.conj()@v.T
            else: v_tilde = B1.T.conj()@v.T
            povm.append(v_tilde@v_tilde.T.conj())
        return povm
    
    # aqc_circuits = scope_hybrid.loc[i]['AQC Circuits']
    U0 = Operator(aqc_circuits['0']).data
    U00 = Operator(aqc_circuits['00']).data
    U10 = Operator(aqc_circuits['10']).data
    povm = get_povm_from_hybrid_unitaries(U0, U00, U10)
    return povm

def get_povm_fidelity_from_hybrid_aqc_circuits(aqc_circuits):
    '''
    Calculate the fidelity of POVM elements constructed from hybrid adaptive quantum circuits.

    Parameters:
    aqc_circuits (dict): A dictionary of quantum circuits.

    Returns:
    float: Fidelity of the reconstructed POVM.
    '''
    choi = get_normalized_choi_matrix_from_povm(sic_4d_povm)
    povm = get_povm_from_hybrid_aqc_circuits(aqc_circuits)
    choi_appx = get_normalized_choi_matrix_from_povm(povm)
    return state_fidelity(choi, choi_appx)

def povm_fidelity(povm1, povm2):
    '''
    Calculate the fidelity between two sets of POVM elements.

    Parameters:
    povm1 (list): The first set of POVM elements.
    povm2 (list): The second set of POVM elements.

    Returns:
    float: Fidelity between the two POVMs.
    '''
    choi1 = get_normalized_choi_matrix_from_povm(povm1)
    choi2 = get_normalized_choi_matrix_from_povm(povm2)
    # return state_fidelity(choi1, choi2)
    return fidelity(choi1, choi2)

def split_circuit(circuit, start, end):  
    '''
    Splits a given quantum circuit into a subcircuit from the specified start to end gates.

    Parameters:
    circuit (QuantumCircuit): The quantum circuit to split.
    start (int): The starting index of the gate.
    end (int): The ending index of the gate.

    Returns:
    QuantumCircuit: The resulting subcircuit.
    '''  
    qr = circuit.qregs[0]
    qc = QuantumCircuit(qr)
    for x in circuit[start:end]:
        qc.compose(x[0], x[1], inplace=True)
    return qc