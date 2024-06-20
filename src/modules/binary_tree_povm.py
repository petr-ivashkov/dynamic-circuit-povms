import numpy as np
import scipy

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
# from qiskit.extensions import UnitaryGate

# Helper functions section:
def get_measurement_op(povm, start, end):
    """
    Returns a cumulative measurement operator by grouping together
    POVM elements from povm[start] to povm[end-1]
    """
    return np.sum(povm[start:end],axis=0).round(5)

def get_diagonalization(povm, start, end):
    """
    Returns: 
        Kraus operator <M>, diagonal <D> and the modal matrix <V> for a given 
        measurement operator by diagonalizing the measurement operator in the form:
        M = V@D@Vh such that M@M = E
    """
    E = get_measurement_op(povm, start, end)
    d2,V = np.linalg.eig(E)
    D = np.real(np.sqrt(np.diag(d2)))
    M = V@D@np.linalg.inv(V)
    return M,D,V

def get_next_level_binary_kraus_ops(povm, start, end):
    """
    Computes two next level binary Kraus operators 
    Args:
        povm: numpy array of POVM elements
        start/end: indices which define the cumulative POVM element
    Returns: 
        Two binary Kraus operators b0 and b1 which take from a higher to lower branch
    * <M> is the Kraus operator corresponding to the current level in binary tree
    * <M0> (<M1>) is the Kraus operators corresponding to the left (right) branch
    * <Q> asserts the completeness condition: b0@b0.T.conj() +  b1@b1.T.conj() = I
    * <M_psinv> is the Moore-Penrose pseudo-inverse of <M>
    """
    mid = int(start + (end-start)/2)
    # computing <M>
    M,D,V = get_diagonalization(povm, start, end)
    # computing the null space of <M>
    P = np.sign(D.round(5))
    Pc = np.eye(len(M))-P
    Q = V@Pc@V.T.conj()
    # computing <M_psinv>
    D_inv = np.linalg.pinv(D)
    M_psinv = V@D_inv@V.T.conj()
    # computing <M0> and <M1>
    M0,_,_ = get_diagonalization(povm, start, mid)
    M1,_,_ = get_diagonalization(povm, mid, end)
    # computing <b0> and <b1>
    b0 = M0@M_psinv + Q/np.sqrt(2)
    b1 = M1@M_psinv + Q/np.sqrt(2)
    return b0, b1

def closest_unitary(A):
    """ Calculates the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix A.
    """
    V, _, Wh = scipy.linalg.svd(A)
    U = np.matrix(V.dot(Wh))
    return U

def closest_unitary_schur(A):
    T, Z = scipy.linalg.schur(A, output='complex')
    return Z @ np.diag(np.diag(T)/abs(np.diag(T))) @ Z.T.conj()

def extend_to_unitary(b0, b1):
    """ Creates a coupling unitary between the system and ancilla.
        The condition for unitary is: <0|U|0> = b0 and <1|U|0> = b1,
        whereby the ancilla is projected onto states |0> and |1>.
        A two-column matrix A, with its upper left block given by b0, 
        and bottom left block by b1, is extended to unitary U by appending  
        a basis of the null space of A.
    """ 
    A = np.concatenate((b0.conj(),b1.conj()))
    u, _, _ = scipy.linalg.svd(A)
    y = u[:, len(A[0]):]
    U = np.hstack((A, y))
    # verify U is close to unitary
    assert np.allclose(U.T.conj()@U, np.eye(len(A)),atol=1e-03), "Failed to construct U"
    return closest_unitary(U)

# Class definitions
class POVM:
    """Base class that holds an arbitrary POVM <povm> as a list of <N> POVM elements.
    """
    def __init__(self, povm):
        """
        Constructor asserts that the given POVM is valid.
        """
        self.povm = povm
        self.N = len(povm)
        self.depth = int(np.ceil(np.log2(self.N))) # required depth of the binary tree
        self.povm_dim = len(povm[0]) # dimension of the POVM operators
        self.n_qubits = int(np.log2(self.povm_dim)) # number of system qubits
        assert self.is_valid()
    def is_valid(self):
        """Verifies the hermiticity, positivity of the POVM and that
        the POVM resolves the identity.
        Returns:
            True: if all conditions are satisfied
        Raises:
            Assertion Error: if one of conditions is not satisfied
        """
        for E in self.povm:
            assert np.allclose(E.conj().T, E), "Some POVM elements are not hermitian"
            assert np.all(np.linalg.eigvals(E).round(3) >= 0), "Some POVM elements are not positive semi-definite"
        assert np.allclose(sum(self.povm), np.eye(self.povm_dim)), "POVM does not resolve the identity"
        return True

class BinaryTreePOVM(POVM):
    """Class which implements the binary tree approach as described in https://journals.aps.org/pra/abstract/10.1103/PhysRevA.77.052104 
    to contruct a POVM measurement tree. 
    """
    def __init__(self, povm, conditional_calibration=False):
        """Creates a binary stree structure with BinaryMeasurementNode objects
        stored in <nodes> dictionary. The keys in the dictionary are the bitstrings 
        corresponding to the states of the classical register at the point when the
        corresponding node has been "reached".
        Args:
            povm: list of POVM elements
        """
        super().__init__(povm)
        # pad with zero operators if necessary
        while np.log2(self.N)-self.depth != 0:
            self.povm.append(np.zeros_like(self.povm[0]))
            self.N += 1
             
        self.nodes = {}
        self.conditional_calibration = conditional_calibration

        self.create_binary_tree(key="0", start=0, end=self.N)
        self.qc = self.construct_measurement_circuit()
    def create_binary_tree(self, key, start, end):
        """Recursive method to build the measurement tree.
        Terminates when the fine-grain level corresponding to the single POVM
        elements is reached.
        <start> and <end> are the first and (last-1) indices of POVM elements
        which were grouped together to obtain a cumulative coarse-grain operator. 
        The range [start, end) corresponds to the possible outcomes which "sit" in 
        the branches below.
        """
        if start >= (end-1):
            return
        new_node = BinaryMeasurementNode(self.povm, key=key, start=start, end=end)
        self.nodes[key] = new_node
        mid = int(start + (end-start)/2)
        self.create_binary_tree(new_node.left, start=start, end=mid)
        self.create_binary_tree(new_node.right, start=mid, end=end)        
    def construct_measurement_circuit(self):
        """Contructs a quantum circuit <qc> for a given POVM by sequentially appending
        coupling unitaries <U> and measurements conditioned on the state of the
        classical register <cr>. The method uses BFS traversal of the precomputed 
        binary measurement tree, i.e. the measurement nodes are visited in level-order.
        
        * Traversal terminates when the fine-grain level was reached.
        * Ancilla qubit is reset before each level.
        * The root node has the key "0".
        * The <if_test> instruction is applied to the entire classical register <cr>,
          whereby the value is the key of the corresponding node - padded with zeros 
          from right to the length of the <cr> register - and interpreted as an integer.
          
          Example:
            At the first level the two nodes have keys:
                left = "00" and right = "01"
            If the <cr> is 3 bits long, then the left/right unitary is applied if 
            the state of <cr> is int("000",2) = 0 / int("010",2) = 2
        """
        qr = QuantumRegister(self.n_qubits+1, name='system+anc')
        cr = ClassicalRegister(self.depth)
        if self.conditional_calibration:
            n_circuits = 2**(self.depth-1)
        else: n_circuits = 1
        qc = []
        for cc in range(n_circuits):
            qc.append(QuantumCircuit(qr, cr, name="measurement-circuit"))

        root = self.nodes["0"]

        # U_gate = UnitaryGate(root.U, label=root.key)

        for cc in range(len(qc)):

            # qc[cc].append(U_gate, qr)
            qc[cc].unitary(root.U, qr, label=root.key)

            qc[cc].measure(qr[-1], cr[0])
            # invert post-measurement state for conditional calibration
            if cc&1:
                qc[cc].x(qr[-1])
            if self.depth != 1:
                if cc&1:
                    qc[cc].x(qr[-1]).c_if(cr[0],0)
                else: qc[cc].x(qr[-1]).c_if(cr[0],1)
        if self.depth == 1: return qc
        
        current_level = [self.nodes["00"],self.nodes["10"]]

        for i in range(1,self.depth):
            next_level = []
            for node in current_level:
                # U_gate = UnitaryGate(node.U, label=node.key)

                #XOR node key with conditional calibration to invert required bits (conditions)
                for cc in range(len(qc)):
                    cr_state = int(node.key[:-1],2)^(cc%(2**i))
                    with qc[cc].if_test((cr, cr_state)):
                        # qc[cc].append(U_gate, qr)
                        qc[cc].unitary(node.U, qr, label=node.key)
                if node.left in self.nodes: next_level.append(self.nodes[node.left])
                if node.right in self.nodes: next_level.append(self.nodes[node.right])
            current_level = next_level
            for cc in range(len(qc)): 
                qc[cc].measure(qr[-1], cr[i])
                # invert the post-measurement state according to conditional calibration (check with bitwise AND)
                post_measurement_inversion = bool(cc&(2**i))
                if post_measurement_inversion:
                    qc[cc].x(qr[-1])
                # don't reset ancilla after the last measurement
                if i == self.depth-1: continue
                # instead of resetting apply conditional X gate
                # dependent on conditional calibration
                if post_measurement_inversion:
                    qc[cc].x(qr[-1]).c_if(cr[i],0)
                else: qc[cc].x(qr[-1]).c_if(cr[i],1) 
            
        return qc

class BinaryMeasurementNode(POVM):
    """A BinaryMeasurementNode object is a node in the BinaryTreePOVM.
    It contains:
        1. Its <key> in the <nodes> dictionary.
        2. <start> and <end>: the first and (last-1) indices of the accumulated 
        POVM elements, corresponding this node.
        3. Coupling unitary <U>.
        4. Keys <left> and <right> of the two children nodes.
        5. Attributes of the POVM class: <M>, <N>, <M_dim>.
        6. Its level <level> in the binary tree, where level of the root node is 0.
    """
    def __init__(self, povm, key, start, end):
        super().__init__(povm)
        self.key = key
        self.level = len(self.key)-1
        self.start = start
        self.end = end
        self.left = "0" + self.key
        self.right = "1" + self.key
        b0, b1 = get_next_level_binary_kraus_ops(self.povm, self.start, self.end)
        self.U = extend_to_unitary(b0, b1)
    def __str__(self):
        line1 = 'Node with the key {} at level {}\n'.format(self.key, self.level)
        line2 = 'Cumulative operator = [{},{})'.format(self.start, self.end)
        line3 = 'left = {}, right = {}\n'.format(self.left, self.right)
        line4 = 'U = \n{}\n'.format(self.U)
        return line1+line2+line3+line4