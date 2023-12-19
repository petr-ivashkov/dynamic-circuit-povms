import numpy as np

def rem_1q_naimark(counts, m_sys, m_anc):
    '''
    Perform readout error mitigation for a single-qubit system using the Naimark extension.

    Parameters:
    counts (dict): A dictionary of observed counts from the quantum circuit.
    m_sys (np.array): Measurement matrix for the system qubit.
    m_anc (np.array): Measurement matrix for the ancilla qubit.

    Returns:
    dict: A dictionary of corrected counts after applying readout error mitigation.
    '''
    keys = ['00','01','10','11']
    p = counts
    M_inv = np.kron(np.linalg.inv(m_anc),np.linalg.inv(m_sys))
    P = np.array([0 if k not in p else p[k] for k in keys])
    Q = M_inv@P
    q = {}
    for i in range(len(keys)):
        q[keys[i]] = Q[i]
    
    return q

def rem_2q_naimark(counts, m_sys0, m_sys1, m_anc0, m_anc1):
    '''
    Perform readout error mitigation for a two-qubit system using the Naimark extension.

    Parameters:
    counts (dict): A dictionary of observed counts from the quantum circuit.
    m_sys0, m_sys1 (np.array): Measurement matrices for the two system qubits.
    m_anc0, m_anc1 (np.array): Measurement matrices for the two ancilla qubits.

    Returns:
    dict: A dictionary of corrected counts after applying readout error mitigation.
    '''
    keys = ['0000','0001','0010','0011',
            '0100','0101','0110','0111',
            '1000','1001','1010','1011',
            '1100','1101','1110','1111']
    p = counts
    M_inv = np.kron(np.linalg.inv(m_anc1),
                    np.kron(np.linalg.inv(m_anc0),
                            np.kron(np.linalg.inv(m_sys1),
                                    np.linalg.inv(m_sys0))))

    P = np.array([0 if k not in p else p[k] for k in keys])
    Q = M_inv@P
    q = {}
    for i in range(len(keys)):
        q[keys[i]] = Q[i]
    
    return q

def crem_1q_binary(counts, m_anc):
    '''
    Perform conditional readout error mitigation for a single-qubit system for a binary circuit.

    Parameters:
    counts (tuple): A tuple of dictionaries representing observed counts for two conditional cases.
    m_anc (np.array): Measurement matrix for the ancilla qubit.

    Returns:
    dict: A dictionary of corrected counts after applying conditional readout error mitigation.
    '''
    p = counts[0]
    px = counts[1]
    M_inv = np.kron(np.linalg.inv(m_anc),np.linalg.inv(m_anc))

    P0 = np.array([p['00'], px['01'], p['10'], px['11']])
    P1 = np.array([px['00'], p['01'], px['10'], p['11']])

    Q0 = M_inv@P0
    Q1 = M_inv@P1

    q = {'00':Q0[0],'01':Q1[1],'10':Q0[2],'11':Q1[3]}
    qx = {'00':Q1[0],'01':Q0[1],'10':Q1[2],'11':Q0[3]}

    return q

def crem_2q_hybrid(counts, m_sys0, m_sys1, m_anc):
    '''
    Perform conditional readout error mitigation for a two-qubit system for a hybrid circuit.

    Parameters:
    counts (dict): A dictionary of observed counts from the quantum circuit.
    m_sys0, m_sys1 (np.array): Measurement matrices for the two system qubits.
    m_anc (np.array): Measurement matrix for the ancilla qubits.

    Returns:
    dict: A dictionary of corrected counts after applying conditional readout error mitigation.
    '''
    def get_bin_str(i, zfill=4):
        # Helper function to convert an integer to a binary string of a specified length.
        return bin(i)[2:].zfill(zfill)
    
    p = counts
    # Inverse of the Kronecker product of measurement matrices for system and ancilla qubits.
    M_inv = np.kron(np.linalg.inv(m_sys0),
                    np.kron(np.linalg.inv(m_sys1),
                            np.kron(np.linalg.inv(m_anc),
                                    np.linalg.inv(m_anc))))

    # Initializing probability arrays for conditional cases.
    P = [[0 for i in range(16)] for j in range(2)]
    for j in range(2):
        for i in range(16):
            if get_bin_str(i) in p[i%2^j].keys():
                P[j][i] = p[i%2^j][get_bin_str(i)]

    # Initializing probability arrays for conditional cases.
    Q = [M_inv@P[i] for i in range(2)]
    q = {}
    for i in range(16):
        q[get_bin_str(i)] = Q[i%2][i]

    return q

def crem_2q_binary(counts, m_anc):
    '''
    Perform conditional readout error mitigation for a two-qubit system in a binary setting.

    Parameters:
    counts (dict): A dictionary of observed counts from the quantum circuit.
    m_anc (np.array): Measurement matrix for the ancilla qubits.

    Returns:
    dict: A dictionary of corrected counts after applying conditional readout error mitigation.
    '''
    def get_bin_str(i, zfill=4):
        # Helper function to convert an integer to a binary string of a specified length.
        return bin(i)[2:].zfill(zfill)
    
    p = counts
    # Inverse of the Kronecker product of measurement matrices for system and ancilla qubits.
    M_inv = np.kron(np.linalg.inv(m_anc),
                    np.kron(np.linalg.inv(m_anc),
                            np.kron(np.linalg.inv(m_anc),
                                    np.linalg.inv(m_anc))))
    
    # Initializing probability arrays for conditional cases
    P = [[0 for i in range(16)] for j in range(8)]
    for j in range(8):
        for i in range(16):
            if get_bin_str(i) in p[i%8^j].keys():
                P[j][i] = p[i%8^j][get_bin_str(i)]

    # Applying inverse measurement matrices.
    Q = [M_inv@P[i] for i in range(8)]
    q = {}
    for i in range(16):
        q[get_bin_str(i)] = Q[i%8][i]

    return q