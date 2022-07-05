from qiskit import *
from qiskit.circuit import Parameter
import networkx as nx

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
    w = nx.get_edge_attributes(G, "weight")
    n = len(G.nodes())
    for edge in G.edges():
        start = edge[0]
        end = edge[1]
        weight = w[edge]
        obj -= weight * int(x[start]) * (1 - int(x[end]))
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
    w = nx.get_edge_attributes(G, "weight")

    # initial_state
    for i in range(nqubits):
        qc.h(i)

    for irep in range(0, p):
        # problem unitary
        for pair in list(G.edges()):
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
def get_expectation(G, p, shots=512):
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
