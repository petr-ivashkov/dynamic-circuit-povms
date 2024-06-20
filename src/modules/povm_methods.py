import numpy as np
import scipy

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# Helper functions section:
def closest_unitary(A):
    """ Calculates the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix A.
    """
    V, _, Wh = scipy.linalg.svd(A)
    U = np.matrix(V.dot(Wh))
    return U

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
        self.assert_valid()

    def assert_valid(self):
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
        assert np.allclose(sum(self.povm), np.eye(self.povm_dim), atol=1e-04), "POVM does not resolve the identity"

        # TODO: extend to arbitrary N by padding with zero operators and possibly rearranging
        assert (self.N & (self.N - 1)) == 0, "Number of POVM elements is not a power of 2"
    
    def get_measurement_op(self, start, end):
        """
        Returns a cumulative measurement operator by grouping together
        POVM elements from povm[start] to povm[end-1]
        """
        return np.sum(self.povm[start:end],axis=0).round(5)

    def get_diagonalization(self, start, end):
        """
        Returns: 
            Kraus operator <M>, diagonal <D> and the modal matrix <V> for a given 
            measurement operator by diagonalizing the measurement operator in the form:
            M = V@D@Vh such that Mh@M = E
        """
        E = self.get_measurement_op(start, end)
        d2,V = np.linalg.eig(E)
        D = np.diag(np.real(np.sqrt(d2)))
        M = V@D@np.linalg.inv(V)
        return M,D,V

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
            conditional_calibration: if True, returns a list of conditional calibration
                                     for conditional readout error mitigation.
        """
        super().__init__(povm)
        self.permute_povm()

        # pad with zero operators if necessary
        while np.log2(self.N)-self.depth != 0:
            self.povm.append(np.zeros_like(self.povm[0]))
            self.N += 1
             
        self.nodes = {}
        self.conditional_calibration = conditional_calibration

        self.create_binary_tree(key="0", start=0, end=self.N)
        self.qc = self.construct_measurement_circuit()

    def permute_povm(self):
        """
        Necessary index permutation due to the order of measurements and binary conversion.
        """
        permuted_povm = [None] * self.N
        for i in range(self.N):
            reversed_bin_i = np.binary_repr(i, self.depth)[::-1]
            new_index = int(reversed_bin_i, 2)
            permuted_povm[new_index] = self.povm[i]

        self.povm = permuted_povm
        self.assert_valid()

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

        * cc stands for conditional calibration. The number of conditional calibration
          circuits is 2^(number of mid-circuit measurements). Refer to https://arxiv.org/abs/2312.14087
          for further details.

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
        for cc in range(len(qc)):
            qc[cc].unitary(root.U, qr, label=root.key)
            qc[cc].measure(qr[-1], cr[0])
            # invert post-measurement state for conditional calibration
            if cc&1:
                qc[cc].x(qr[-1])
            if self.depth != 1:
                if cc&1:
                    qc[cc].x(qr[-1]).c_if(cr[0],0)
                else: qc[cc].x(qr[-1]).c_if(cr[0],1)
        if self.depth == 1:
            if not self.conditional_calibration:
                return qc[0]
            else:
                return qc 
                    
        current_level = [self.nodes["00"],self.nodes["10"]]

        for i in range(1,self.depth):
            next_level = []
            for node in current_level:
                #XOR node key with conditional calibration to invert required bits (conditions)
                for cc in range(len(qc)):
                    cr_state = int(node.key[:-1],2)^(cc%(2**i))
                    with qc[cc].if_test((cr, cr_state)):
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

        if not self.conditional_calibration:
            return qc[0]
        else:
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
        b0, b1 = self.get_next_level_binary_kraus_ops(self.start, self.end)
        self.U = self.extend_to_unitary(b0, b1)

    def __str__(self):
        line1 = 'Node with the key {} at level {}\n'.format(self.key, self.level)
        line2 = 'Cumulative operator = [{},{})'.format(self.start, self.end)
        line3 = 'left = {}, right = {}\n'.format(self.left, self.right)
        line4 = 'U = \n{}\n'.format(self.U)
        return line1+line2+line3+line4

    def extend_to_unitary(self, b0, b1):
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

    def get_next_level_binary_kraus_ops(self, start, end):
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
        M,D,V = self.get_diagonalization(start, end)
        # computing the null space of <M>
        P = np.sign(D.round(5))
        Pc = np.eye(len(M))-P
        Q = V@Pc@V.T.conj()
        # computing <M_psinv>
        D_inv = np.linalg.pinv(D, rcond=1e-4)
        M_psinv = V@D_inv@V.T.conj()
        # computing <M0> and <M1>
        M0,_,_ = self.get_diagonalization(start, mid)
        M1,_,_ = self.get_diagonalization(mid, end)
        # computing <b0> and <b1>
        b0 = M0@M_psinv + Q/np.sqrt(2)
        b1 = M1@M_psinv + Q/np.sqrt(2)
        return b0, b1

class NaimarkMeasurementNode(POVM):
    """
    A node representing a Naimark measurement, possibly as a node in a tree structure.
    The node contains the information and operations needed to perform Naimark's dilation.
    """
    def __init__(self, povm, key):
        """
        Initialize a NaimarkMeasurementNode with given POVM elements and node key.
        Verifies that the POVM is rank-1 and constructs the Naimark unitary.

        Args:
            povm: List of POVM elements.
            key: The key for the node in the tree.
            start: The starting index of the POVM elements for this node.
            end: The ending index of the POVM elements for this node.
        """
        super().__init__(povm)
        self.assert_rank_1()

        self.key = key
        self.level = len(self.key)-1

        self.vectors = [self.get_vector(E) for E in self.povm]
        self.U = self.get_naimark_unitary_from_vectors()

    def assert_rank_1(self):
        """Check if the given matrix is rank-1."""
        for E in self.povm: assert np.linalg.matrix_rank(E) == 1, "POVM is not rank 1."

    def get_vector(self, E):
        """Find the vector psi such that E = psi.T.conj() @ psi."""
        U, S, Vh = np.linalg.svd(E)
        psi = np.sqrt(S[0]) * U[:, 0]
        return psi

    def get_naimark_unitary_from_vectors(self, assert_unitarty=True, atol=1e-04):
        '''
        Construct a Naimark unitary matrix from a set of vectors.

        Parameters:
            assert_unitarty (bool): If True, asserts that the constructed matrix is unitary.
            atol (float): Absolute tolerance for the unitarity assertion.

        Returns:
            np.matrix: The constructed Naimark unitary matrix.
        '''
        A = np.concatenate(([psi.reshape(1, -1) for psi in self.vectors]), axis=0)
        y = scipy.linalg.null_space(A.T.conj())
        U = np.concatenate((A,y),axis=1)
        if assert_unitarty:
            assert np.allclose(U.T.conj()@U, np.eye(len(self.vectors)),atol=atol), "Failed to construct U"
        return closest_unitary(U)

    def __str__(self):
        line1 = 'Node with the key {} at level {}\n'.format(self.key, self.level)
        line2 = 'U = \n{}\n'.format(self.U)
        return line1+line2
    

class NaimarkPOVM(POVM):
    """
    A class that constructs a POVM measurement using the Naimark's dilation method.
    """
    def __init__(self, povm):
        """
        Initialize a NaimarkPOVM with given POVM elements.
        Pads the POVM if necessary and constructs the measurement circuit.

        Args:
            povm: List of POVM elements.
        """
        super().__init__(povm)
        # pad with zero operators if necessary
        while np.log2(self.N)-self.depth != 0:
            self.povm.append(np.zeros_like(self.povm[0]))
            self.N += 1
            
        self.nodes = {"0" : NaimarkMeasurementNode(self.povm, "0")}
        self.U = self.nodes["0"].U
        self.qc = self.construct_measurement_circuit()

    def construct_measurement_circuit(self):
        """
        Construct a quantum circuit for the POVM measurement using the Naimark unitary.

        Returns:
            qc: The constructed quantum measurement circuit.
        """
        n_sys_qubits = np.log2(self.povm_dim)
        L = np.log2(self.N)

        system = QuantumRegister(n_sys_qubits, name='system')
        anc = QuantumRegister(L-n_sys_qubits,name='ancilla')
        qc = QuantumCircuit(system, anc, name="measurement-circuit")

        qc.unitary(self.U, system[:] + anc[:], label='0')

        qc.measure_all()
        return qc
    
class HybridTreePOVM(POVM):
    """A class implementing a hybrid tree approach for constructing a POVM measurement tree.

    For details refer to: https://arxiv.org/abs/2312.14087

    Attributes:
        m (int): Number of binary search steps.
        
        * Other attributes are similar to BinaryTreePOVM
    """
    def __init__(self, povm, conditional_calibration=False):
        """
        Initializes a HybridTreePOVM object.

        Args:
            povm (list of np.ndarray): List of POVM elements.
            conditional_calibration (bool): If True, enables conditional calibration.
        """
        super().__init__(povm)
        self.permute_povm()
        # pad with zero operators if necessary
        while np.log2(self.N)-self.depth != 0:
            self.povm.append(np.zeros_like(self.povm[0]))
            self.N += 1
        
        self.m = int(np.log2(self.N / (2*self.povm_dim)))
        self.nodes = {}
        self.conditional_calibration = conditional_calibration

        self.create_hybrid_tree(key="0", start=0, end=self.N)
        self.qc = self.construct_measurement_circuit()

    def permute_povm(self):
        """
        Necessary index permutation due to the order of measurements and binary conversion.
        """
        permuted_povm = [None] * self.N
        for i in range(self.N):
            reversed_bin_i = np.binary_repr(i, self.depth)[::-1]
            new_index = int(reversed_bin_i, 2)
            permuted_povm[new_index] = self.povm[i]

        self.povm = permuted_povm
        self.assert_valid()

    def create_hybrid_tree(self, key, start, end):
        """
        Recursively constructs a hybrid measurement tree.

        Terminates when the number of remainig outcomes coresponds to 2d.
        """
        if (end - start) <= 2*self.povm_dim:
            M = self.get_measurement_op(start, end)
            m = scipy.linalg.sqrtm(M)
            m_inv = np.linalg.inv(m)

            naimark_povm = []
            for i in range(start, end):
                E_tilde = m_inv.T.conj()@self.povm[i]@m_inv
                naimark_povm.append(E_tilde)

            new_node = NaimarkMeasurementNode(naimark_povm, key)
            self.nodes[key] = new_node
            return 

        new_node = BinaryMeasurementNode(self.povm, key=key, start=start, end=end)
        self.nodes[key] = new_node
        mid = int(start + (end-start)/2)
        self.create_hybrid_tree(new_node.left, start=start, end=mid)
        self.create_hybrid_tree(new_node.right, start=mid, end=end)    

    def construct_measurement_circuit(self):
        """
        Constructs a quantum measurement circuit for the hybrid tree.
        """
        qr = QuantumRegister(self.n_qubits+1, name='system+anc')
        cr = ClassicalRegister(self.depth)
        if self.conditional_calibration:
            n_circuits = 2**self.m
        else: n_circuits = 1
        qc = []
        for cc in range(n_circuits):
            qc.append(QuantumCircuit(qr, cr, name="measurement-circuit"))

        root = self.nodes["0"]
        for cc in range(len(qc)):
            qc[cc].unitary(root.U, qr, label=root.key)
            if self.m == 1:
                qc[cc].measure(qr[-1], cr[0])
                # invert post-measurement state for conditional calibration
                if cc&1:
                    qc[cc].x(qr[-1])
                if self.depth != 1:
                    if cc&1:
                        qc[cc].x(qr[-1]).c_if(cr[0],0)
                    else: qc[cc].x(qr[-1]).c_if(cr[0],1)
            elif self.m > 1:
                for j in range(i, self.depth):
                    qc[cc].measure(qr[j], cr[j])

        if self.m == 0:
            if not self.conditional_calibration:
                return qc[0]
            else:
                return qc 
                    
        current_level = [self.nodes["00"],self.nodes["10"]]

        for i in range(1,self.m+1):
            next_level = []
            for node in current_level:
                #XOR node key with conditional calibration to invert required bits (conditions)
                for cc in range(len(qc)):
                    cr_state = int(node.key[:-1],2)^(cc%(2**i))
                    with qc[cc].if_test((cr, cr_state)):
                        qc[cc].unitary(node.U, qr, label=node.key)
                # NOTE: NaimarkMeasurementNode doesn't have <left> and <right> attributes
                if hasattr(node, "left"):
                    if node.left in self.nodes: next_level.append(self.nodes[node.left])
                if hasattr(node, "right"):
                    if node.right in self.nodes: next_level.append(self.nodes[node.right])
            current_level = next_level
            for cc in range(len(qc)): 
                if i < self.m:
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
                elif i == self.m:
                    for j in range(i, self.depth):
                        qc[cc].measure(qr[self.n_qubits-(j-i)], cr[j])

        if not self.conditional_calibration:
            return qc[0]
        else:
            return qc