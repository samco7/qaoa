from classical_solvers import *
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from qiskit import *
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
import cvxpy as cp


class QAOAResult:
    def __init__(self):
        self.bitstring = None
        self.relaxed_bitstring = None
        self.obj = None
        self.relaxed_obj = None
        self.bmz_angles = None
        self.counts = None
        self.expectation = None
        self.unique_samples = None
        self.qaoa_hyperparameters = dict()

    def __str__(self):
        width = len('unique_samples: ')
        out = 'bitstring: '.rjust(width) + str(self.bitstring)
        out += '\n' + 'obj: '.rjust(width) + str(self.obj)
        if self.relaxed_obj is not None:
            out += '\n' + 'relaxed_obj: '.rjust(width) + str(self.relaxed_obj)
        out += '\n' + 'expectation: '.rjust(width) + str(self.expectation)
        out += '\n' + 'unique_samples: '.rjust(width) + str(self.unique_samples)
        return out

    def __repr__(self):
        return self.__str__()


class QAOASolver:
    def __init__(self, n_layers, warm_start_method=None, epsilon=None, backend=None, shots=1024):
        self.__n_layers = n_layers
        allowed_methods = [None, 'GW Rounded', 'BMZ Rounded', 'BMZ']
        check = warm_start_method in allowed_methods
        assert check, 'warm_start_method must be None, \'GW Rounded\', \'BMZ Rounded\', or \'BMZ\'.'
        self.__warm_start_method = warm_start_method
        if epsilon is None:
            assert warm_start_method is None, 'Choose a value for epsilon, cannot be None for warm-starting.'
        self.__epsilon = epsilon
        if backend is None:
            self.__backend = AerSimulator(method='statevector', device='CPU', precision='single')
        else:
            self.__backend = backend
        self.__shots = shots
        self.__graph = None
        self.__n_qubits = None
        self.__relaxed_bits = None
        self.__relaxed_obj = None
        self.__bmz_angles = None
        self.__thetas = None
        self.__circ = None

    def __solve_relaxed(self):
        if self.__warm_start_method == 'BMZ Rounded' or self.__warm_start_method == 'BMZ':
            self.__relaxed_bits, self.__relaxed_obj, self.__bmz_angles = BMZ(self.__graph)
        elif self.__warm_start_method == 'GW Rounded':
            self.__relaxed_bits, self.__relaxed_obj = GW(self.__graph)
        elif self.__warm_start_method is None:
            return
        else:
            raise Exception('warm_start_method must be None, \'GW Rounded\', \'BMZ Rounded\', or \'BMZ\'.')

    def __set_thetas(self):
        if self.__warm_start_method == 'BMZ':
            angles = np.copy(self.__bmz_angles)
            angles[np.where(np.abs(self.__bmz_angles) < self.__epsilon)] = self.__epsilon
            angles[np.where(np.abs(self.__bmz_angles - np.pi) < self.__epsilon)] = np.pi + self.__epsilon
            self.__thetas = angles[::-1]
        elif self.__warm_start_method == 'BMZ Rounded' or self.__warm_start_method == 'GW Rounded':
            c_stars = np.zeros(self.__n_qubits)
            c_stars[np.where(self.__relaxed_bits == 1)] = 1 - self.__epsilon
            c_stars[np.where(self.__relaxed_bits == 0)] = self.__epsilon
            self.__thetas = [2*np.arcsin(np.sqrt(c_star)) for c_star in c_stars][::-1]
        elif self.__warm_start_method is None:
            self.__thetas = np.pi/2*np.ones(self.__n_qubits)
            return
        else:
            raise Exception('warm_start_method must be None, \'GW Rounded\', \'BMZ Rounded\', or \'BMZ\'.')

    def __construct_circuit(self):
        circ = QuantumCircuit(self.__n_qubits)

        # subroutine to create one layer of the cost operator
        def get_cost_circuit(parameter):
            w = nx.get_edge_attributes(self.__graph, "weight")
            qc = QuantumCircuit(self.__n_qubits)
            for pair in list(self.__graph.edges()):
                qc.cx(pair[0], pair[1])
                qc.rz(parameter, pair[1])
                qc.cx(pair[0], pair[1])
            return qc

        # subroutine to create one layer of the mixer operator
        if self.__warm_start_method in ['GW Rounded', 'BMZ Rounded']:
            def get_mixer_circuit(parameter):
                qc = QuantumCircuit(self.__n_qubits)
                for i in self.__graph.nodes():
                    qc.ry(self.__thetas[i], i)
                    qc.rz(-2*parameter, i)
                    qc.ry(-self.__thetas[i], i)
                return qc

        else:
            def get_mixer_circuit(parameter):
                qc = QuantumCircuit(self.__n_qubits)
                for i in self.__graph.nodes():
                    qc.ry(-self.__thetas[i], i)
                    qc.rz(-2*parameter, i)
                    qc.ry(self.__thetas[i], i)
                return qc
        # initial_state
        for i in range(self.__n_qubits):
            circ.ry(self.__thetas[i], i)

        betas = []
        gammas = []
        # n_layers layers of alternating operators
        for i in range(0, self.__n_layers):
            beta = Parameter(f'beta_{i}')
            gamma = Parameter(f'gamma_{i}')
            betas.append(beta)
            gammas.append(gamma)
            circ = circ.compose(get_cost_circuit(gamma))
            circ = circ.compose(get_mixer_circuit(beta))
        circ.measure_all()

        circ = transpile(circ, self.__backend, optimization_level=0)
        self.__circ = circ
        self.__circ_parameters = {'betas':betas, 'gammas':gammas}

    def __execute_circuit(self, betas, gammas):
        bound_values = dict()
        for beta, i in zip(self.__circ_parameters['betas'], range(self.__n_layers)):
            bound_values[beta] = betas[i]
        for gamma, j in zip(self.__circ_parameters['gammas'], range(self.__n_layers)):
            bound_values[gamma] = gammas[j]
        bound_circuit = self.__circ.bind_parameters(bound_values)
        return self.__backend.run(bound_circuit, shots=self.__shots).result().get_counts()

    def __compute_expectation(self, counts):
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.__objective(bitstring)
            avg += obj * count
            sum_count += count
        avg = avg/sum_count
        return avg

    def __objective(self, x):
        A = nx.adjacency_matrix(self.__graph).toarray()
        mu = -np.ones(self.__n_qubits)@A
        sigma = A + np.diag(mu)
        x_arr = np.array([float(entry) for entry in x])
        return x_arr@sigma@x_arr

    def set_shots(self, shots):
        self.__shots = shots

    def set_warm_start_method(self, method=None):
        self.__warm_start_method = method

    def set_epsilon(self, epsilon):
        self.__epsilon = epsilon

    def solve(self, graph, relaxed_solution=None):
        self.__n_qubits = len(graph.nodes())
        self.__graph = graph
        if self.__warm_start_method is None:
            pass
        elif relaxed_solution is None:
            self.__solve_relaxed()
        else:
            self.__relaxed_bits = relaxed_solution[0]
            self.__relaxed_obj = relaxed_solution[1]
            if len(relaxed_solution) > 2:
                self.__bmz_angles = relaxed_solution[2]

        self.__set_thetas()
        self.__construct_circuit()

        initial_beta = np.random.uniform(0, np.pi, self.__n_layers)
        initial_gamma = np.random.uniform(0, 2*np.pi, self.__n_layers)
        initial_params = np.concatenate((initial_beta, initial_gamma))

        def qaoa_objective(parameters):
            betas = parameters[:self.__n_layers]
            gammas = parameters[self.__n_layers:]
            counts = self.__execute_circuit(betas, gammas)
            expectation = self.__compute_expectation(counts)
            return expectation

        optimized_params = minimize(qaoa_objective, initial_params, method='COBYLA').x
        optimal_betas = optimized_params[:self.__n_layers]
        optimal_gammas = optimized_params[self.__n_layers:]
        counts = self.__execute_circuit(optimal_betas, optimal_gammas)

        return_string = None
        best_val = np.inf
        for key in counts.keys():
            val = self.__objective(str(key))
            if val < best_val:
                best_val = val
                return_string = str(key)

        res = QAOAResult()
        res.bitstring = return_string
        res.relaxed_bitstring = self.__relaxed_bits
        res.obj = best_val
        res.relaxed_obj = self.__relaxed_obj
        res.bmz_angles = self.__bmz_angles
        res.counts = counts
        res.expectation = self.__compute_expectation(counts)
        res.unique_samples = len(counts)
        hyperparameters = dict()
        hyperparameters['epsilon'] = self.__epsilon
        hyperparameters['n_qubits'] = self.__n_qubits
        hyperparameters['n_layers'] = self.__n_layers
        hyperparameters['shots'] = self.__shots
        hyperparameters['warm_start_method'] = self.__warm_start_method
        res.qaoa_hyperparameters = hyperparameters
        return res
