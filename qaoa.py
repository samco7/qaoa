from qiskit import *
from qiskit.circuit import Parameter
import networkx as nx

class maxcut:

    def __init__(self, G, n_layers, shots):
        """
            Constructor. Accepts a graph G (undirected and weighted), an n_layers
            parameter, and a shots parameter
        """
        self.total_time = 0
        self.G = G
        self.n_layers = n_layers
        self.shots = shots

    def maxcut_obj(self, x):
        """
            Accepts a bitstring x and returns the cost function evaluated for
            that bitstring.
        """
        obj = 0
        w = nx.get_edge_attributes(self.G, "weight")
        n = len(self.G.nodes())
        for edge in self.G.edges():
            start = int(x[edge[0]])
            end = int(x[edge[1]])
            weight = w[edge]
            obj -= weight * (start*(1 - end) + end*(1 - start))
        return obj

    def compute_expectation(self, counts):
        """
            Accepts a counts dictionary that maps bitstrings to their sampled
            frequency. Returns the expected cost.
        """
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.maxcut_obj(bitstring)
            avg += obj * count
            sum_count += count

        return avg/sum_count

        # create the qaoa circuit according to the qaoa algorithm
    def create_qaoa_circ(self, theta):
        """
        Creates and returens a parametrized qaoa circuit. Accepts the parameter
        theta, which holds the beta values followed by the gamma values. Length
        of theta should be equal to 2 * n_layers.

        """
        nqubits = len(self.G.nodes())
        if len(theta)//2 != self.n_layers:
            raise Exception('Check parameter array size.')
        qc = QuantumCircuit(nqubits)

        beta = theta[:self.n_layers]
        gamma = theta[self.n_layers:]
        w = nx.get_edge_attributes(self.G, "weight")

        # initial_state
        for i in range(nqubits):
            qc.h(i)
        qc.barrier()

        for irep in range(0, self.n_layers):
            # problem unitary
            for pair in list(self.G.edges()):
                qc.cx(pair[0], pair[1])
                qc.rz(w[pair]*gamma[irep], pair[1])
                qc.cx(pair[0], pair[1])
                qc.barrier()
            # mixer unitary
            for i in range(0, nqubits):
                qc.rx(2*beta[irep], i)

        qc.measure_all()
        return qc

    # now we write a function that executes the circuit on the chosen backend
    def get_expectation(self):
        backend = Aer.get_backend('qasm_simulator')
        backend.shots = self.shots

        def execute_circ(theta, return_result=False):
            qc = self.create_qaoa_circ(theta)
            qobj = transpile(qc, backend=backend)
            result = backend.run(qobj, nshots=self.shots).result()
            self.total_time += result.time_taken #increment our count of total estimated computation time each time the circuit is executed
            counts = result.get_counts()
            if return_result:
                return self.compute_expectation(counts), result
            else:
                return self.compute_expectation(counts)

        return execute_circ
