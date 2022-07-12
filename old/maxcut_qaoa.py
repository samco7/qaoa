from qiskit import *
from qiskit.circuit import Parameter

def maxcut_obj(x, G):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.
    Args:
        x: str
           solution bitstring

        G: networkx graph
    Returns:
        obj: float
             Objective
    """
    obj = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1

    return obj

def compute_expectation(counts, G):
    """
    Computes expectation value based on measurement results
    Args:
        counts: dict
                key as bitstring, val as count

        G: networkx graph
    Returns:
        avg: float
             expectation value
    """
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():

        obj = maxcut_obj(bitstring, G)
        avg += obj * count
        sum_count += count

    return avg/sum_count

    # create the qaoa circuit according to the qaoa algorithm
def create_qaoa_circ(G, theta):
    """
    Creates a parametrized qaoa circuit
    Args:
        G: networkx graph
        theta: list
               unitary parameters
    Returns:
        qc: qiskit circuit
    """
    nqubits = len(G.nodes())
    p = len(theta)//2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)

    beta = theta[:p]
    gamma = theta[p:]

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for irep in range(0, p):
        # problem unitary
        for pair in set(G.edges()):
            qc.rzz(gamma[irep], pair[0], pair[1])
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)

    qc.measure_all()
    return qc

# now we write a function that executes the circuit on the chosen backend
def get_expectation(G, p, shots=1024):
    """
    Runs parametrized circuit
    Args:
        G: networkx graph
        p: int,
           Number of repetitions of unitaries
    """
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta, return_counts=False):
        qc = create_qaoa_circ(G, theta)
        counts = backend.run(qc, seed_simulator=10,
                             nshots=shots).result().get_counts()
        if return_counts:
            return compute_expectation(counts, G), counts
        else:
            return compute_expectation(counts, G)

    return execute_circ
