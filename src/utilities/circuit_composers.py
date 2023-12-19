from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

def cx_depth(circuit):
    return circuit.depth(lambda x: x[0].num_qubits == 2)

def cx_depth_4d_sic_binary(circuits):
    '''
    Calculate the average CNOT gate depth for a 4D SIC-POVM binary circuit.

    Parameters:
    circuits (dict): A dictionary of quantum circuits keyed by string identifiers.

    Returns:
    int: The calculated average CNOT gate depth.
    '''
    keys = ['0','00','10','000','010','100','110','0000','0010','0100','0110','1000','1010','1100','1110']
    cx_depth = 0
    for key in keys[0]:
        cx_depth += circuits[key].count_ops()['cx']
    for key in keys[1:3]:
        cx_depth += circuits[key].count_ops()['cx']/2
    for key in keys[3:7]:
        cx_depth += circuits[key].count_ops()['cx']/4
    for key in keys[7:15]:
        cx_depth += circuits[key].count_ops()['cx']/8
    return cx_depth

def cx_depth_4d_sic_btshort(circuits):
    '''
    Calculate the average CNOT gate depth for a 4D SIC-POVM hybrid circuit.

    Parameters:
    circuits (dict): A dictionary of quantum circuits keyed by string identifiers.

    Returns:
    int: The calculated average CNOT gate depth.
    '''
    keys = ['0','00','10']
    cx_depth = 0
    for key in keys[0]:
        cx_depth += circuits[key].count_ops()['cx']
    for key in keys[1:3]:
        cx_depth += circuits[key].count_ops()['cx']/2
    return cx_depth

def cx_depth_2d_sic_binary(circuits):
    '''
    Calculate the average CNOT gate depth for a 2D SIC-POVM hybrid circuit.

    Parameters:
    circuits (dict): A dictionary of quantum circuits keyed by string identifiers.

    Returns:
    int: The calculated average CNOT gate depth.
    '''
    keys = ['0','00','10']
    cx_depth = 0
    for key in keys[0]:
        cx_depth += circuits[key].count_ops()['cx']
    for key in keys[1:3]:
        cx_depth += circuits[key].count_ops()['cx']/2
    return cx_depth

def compose_binary_circuit_4D(circuits, cc=0):
    '''
    Composes a 3-qubit binary search tree circuit from given unitary circuits for each node.

    Parameters:
    circuits (dict): A dictionary of unitary circuits for each node.
    cc (int): Conditional calibration parameter.

    Returns:
    QuantumCircuit: The composed 3-qubit binary search tree circuit.
    '''
    # Create quantum and classical registers
    sys = QuantumRegister(3, 'q')
    cr = ClassicalRegister(4, 'cr')
    qc = QuantumCircuit(sys, cr)
    qc.compose(circuits['0'], sys[:], inplace=True)
    qc.measure(sys[-1], cr[0])
    
    level_0 = ['0']
    level_1 = ['00','10']
    level_2 = ['000','010','100','110']
    level_3 = ['0000','0010','0100','0110',
               '1000','1010','1100','1110']
    levels = [level_0, level_1, level_2, level_3]
    
    current_level = levels[1]
    for i in range(1,4):
        for key in current_level:
            #XOR node key with conditional calibration to invert required bits (conditions)
            cr_state_init = int(key[:-1],2)
            cr_state = cr_state_init^(cc%(2**i))
            with qc.if_test((cr, cr_state)):
                if cr_state >= 2**(i-1):
                    qc.x(sys[-1])
                qc.compose(circuits[key], sys[:], inplace=True)
        qc.measure(sys[-1], cr[i])
        if i < 3: current_level = levels[i+1]
        
    return qc

def compose_binary_circuit_2D(circuits, cc=False):
    '''
    Composes a 2-qubit binary search tree circuit from given unitary circuits for each node.

    Parameters:
    circuits (dict): A dictionary of unitary circuits for each node.
    cc (bool): Flag for conditional calibration.

    Returns:
    QuantumCircuit: The composed 2-qubit binary search tree circuit.
    '''
    # Create quantum and classical registers
    sys = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'cr')
    circuit = QuantumCircuit(sys, cr)

    # First level of the binary search tree
    circuit.compose(circuits['0'], sys[:], inplace=True)
    circuit.measure(sys[1], cr[0])

    # Second level of the binary search tree
    if not cc:
        with circuit.if_test((cr,0)) as else_:
            circuit.compose(circuits['00'], sys[:], inplace=True)
        with else_:
            circuit.x(sys[1])
            circuit.compose(circuits['10'], sys[:], inplace=True)
    else:
        with circuit.if_test((cr,1)) as else_:
            circuit.x(sys[1])
            circuit.compose(circuits['00'], sys[:], inplace=True)
        with else_:  
            circuit.compose(circuits['10'], sys[:], inplace=True)        
    circuit.measure(sys[1],cr[1])
    return circuit


def compose_hybrid_circuit_4D(circuits, cc=False, num_qubits=3):
    '''
    Composes a 3-qubit hybrid circuit from given unitary circuits for each node.

    Parameters:
    circuits (dict): A dictionary of unitary circuits for each node.
    cc (bool): Flag for conditional calibration.
    num_qubits (int): Number of qubits in the circuit.

    Returns:
    QuantumCircuit: The composed 3-qubit binary short circuit.
    '''
    # Create quantum and classical registers
    sys = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(4, 'cr')
    circuit = QuantumCircuit(sys, cr)

    circuit.compose(circuits['0'], sys[:], inplace=True)
    circuit.measure(sys[2], cr[0])
    if not cc:
        with circuit.if_test((cr,0)) as else_:
            circuit.compose(circuits['00'], sys[:], inplace=True)
        with else_:
            circuit.x(sys[2])
            circuit.compose(circuits['10'], sys[:], inplace=True)
    else:
        with circuit.if_test((cr,1)) as else_:
            circuit.x(sys[2])
            circuit.compose(circuits['00'], sys[:], inplace=True)
        with else_:
            circuit.compose(circuits['10'], sys[:], inplace=True)
    circuit.measure(sys[2], cr[1])
    circuit.measure(sys[1], cr[2])
    circuit.measure(sys[0], cr[3])
    return circuit