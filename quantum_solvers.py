from classical_solvers import *
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from qiskit import *
from qiskit.quantum_info.operators import Operator
from qiskit.opflow.gradients import Gradient
from qiskit.opflow import CircuitStateFn
from qiskit.algorithms.optimizers import ADAM
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
        self.n_regularized = None
        self.counts = None
        self.expectation = None
        self.unique_samples = None
        self.optimizer = None
        self.opt_terminated = None
        self.qaoa_hyperparameters = dict()

    def __str__(self):
        width = len('opt_terminated: ')
        out = 'bitstring: '.rjust(width) + str(self.bitstring)
        out += '\n' + 'obj: '.rjust(width) + str(self.obj)
        if self.relaxed_obj is not None:
            out += '\n' + 'relaxed_obj: '.rjust(width) + str(self.relaxed_obj)
        if self.n_regularized is not None:
            out += '\n' + 'n_regularized: '.rjust(width) + str(self.n_regularized)
        out += '\n' + 'expectation: '.rjust(width) + str(self.expectation)
        out += '\n' + 'unique_samples: '.rjust(width) + str(self.unique_samples)
        out += '\n' + 'optimizer: '.rjust(width) + str(self.optimizer)
        if self.optimizer == 'ADAM':
            out += '\n' + 'opt_terminated: '.rjust(width) + str(self.opt_terminated)
        return out

    def __repr__(self):
        return self.str()


class QAOASolver:
    def __init__(self, n_layers, warm_start_method=None, epsilon=None, adjust_mixer=True, backend=None, shots=512):
        self.n_layers = n_layers
        allowed_methods = [None, 'GW Rounded', 'BMZ Rounded', 'BMZ']
        check = warm_start_method in allowed_methods
        assert check, 'warm_start_method must be None, \'GW Rounded\', \'BMZ Rounded\', or \'BMZ\'.'
        if warm_start_method is None:
            self.label = 'Standard QAOA'
        elif warm_start_method == 'GW Rounded':
            self.label = 'Rounded GW-WS-QAOA'
        elif warm_start_method == 'BMZ Rounded':
            self.label = 'Rounded BMZ-WS-QAOA'
        else:
            if adjust_mixer:
                self.label = 'BMZ-WS-QAOA'
            else:
                self.label = 'BMZ-WS-QAOA State Only'
        self.warm_start_method = warm_start_method
        if epsilon is None:
            epsilon = 0
        self.epsilon = epsilon
        self.adjust_mixer = adjust_mixer
        if warm_start_method is None:
            self.adjust_mixer = False
        if backend is None:
            self.backend = AerSimulator(method='statevector', device='CPU', precision='single')
        else:
            self.backend = backend
        self.shots = shots
        self.graph = None
        self.n_qubits = None
        self.relaxed_bits = None
        self.relaxed_obj = None
        self.bmz_angles = None
        self.n_regularized = None
        self.thetas = None
        self.circ = None
        self.circ_parameters = None
        self.optimizer = None
        self.opt_ended_early = False

    def solve_relaxed(self):
        if self.warm_start_method == 'BMZ Rounded' or self.warm_start_method == 'BMZ':
            self.relaxed_bits, self.relaxed_obj, self.bmz_angles = BMZ(self.graph)
        elif self.warm_start_method == 'GW Rounded':
            self.relaxed_bits, self.relaxed_obj = GW(self.graph)
        elif self.warm_start_method is None:
            return
        else:
            raise Exception('warm_start_method must be None, \'GW Rounded\', \'BMZ Rounded\', or \'BMZ\'.')
        return self.relaxed_bits, self.relaxed_obj, self.bmz_angles

    def set_thetas(self):
        if self.warm_start_method == 'BMZ':
            angles = np.copy(self.bmz_angles)
            indices_0 = np.where(np.abs(self.bmz_angles) < self.epsilon)
            angles[indices_0] = self.epsilon
            indices_1 = np.where(np.abs(self.bmz_angles - np.pi) < self.epsilon)
            angles[indices_1] = np.pi + self.epsilon
            self.thetas = angles[::-1]
            self.n_regularized = len(indices_0[0]) + len(indices_1[0])
        elif self.warm_start_method == 'BMZ Rounded' or self.warm_start_method == 'GW Rounded':
            c_stars = np.zeros(self.n_qubits)
            indices_0 = np.where(self.relaxed_bits == 1)
            c_stars[indices_0] = 1 - self.epsilon
            indices_1 = np.where(self.relaxed_bits == 0)
            c_stars[indices_1] = self.epsilon
            self.thetas = [2*np.arcsin(np.sqrt(c_star)) for c_star in c_stars][::-1]
            self.n_regularized = len(indices_0[0]) + len(indices_1[0])
        elif self.warm_start_method is None:
            self.thetas = np.pi/2*np.ones(self.n_qubits)
            return
        else:
            raise Exception('warm_start_method must be None, \'GW Rounded\', \'BMZ Rounded\', or \'BMZ\'.')

    def construct_circuit(self):
        circ = QuantumCircuit(self.n_qubits)

        # subroutine to create one layer of the cost operator
        def get_cost_circuit(parameter):
            w = nx.get_edge_attributes(self.graph, "weight")
            qc = QuantumCircuit(self.n_qubits)
            for pair in list(self.graph.edges()):
                qc.cx(pair[0], pair[1])
                qc.rz(2*parameter, pair[1])
                qc.cx(pair[0], pair[1])
            return qc

        # subroutine to create one layer of the mixer operator
        if self.adjust_mixer == False:
            def get_mixer_circuit(parameter):
                qc = QuantumCircuit(self.n_qubits)
                for i in self.graph.nodes():
                    qc.rx(2*parameter, i)
                return qc
        elif self.warm_start_method in ['GW Rounded', 'BMZ Rounded']:
            def get_mixer_circuit(parameter):
                qc = QuantumCircuit(self.n_qubits)
                for i in self.graph.nodes():
                    qc.ry(self.thetas[i], i)
                    qc.rz(-2*parameter, i)
                    qc.ry(-self.thetas[i], i)
                return qc
        else:
            def get_mixer_circuit(parameter):
                qc = QuantumCircuit(self.n_qubits)
                for i in self.graph.nodes():
                    qc.ry(-self.thetas[i], i)
                    qc.rz(-2*parameter, i)
                    qc.ry(self.thetas[i], i)
                return qc

        # initial_state
        for i in range(self.n_qubits):
            circ.ry(self.thetas[i], i)

        betas = []
        gammas = []
        # n_layers layers of alternating operators
        for i in range(0, self.n_layers):
            beta = Parameter(f'beta_{i}')
            gamma = Parameter(f'gamma_{i}')
            betas.append(beta)
            gammas.append(gamma)
            circ = circ.compose(get_cost_circuit(gamma))
            circ = circ.compose(get_mixer_circuit(beta))
        circ.measure_all()

        circ = transpile(circ, self.backend, optimization_level=0)
        self.circ = circ
        self.circ_parameters = {'betas':betas, 'gammas':gammas}

    def execute_circuit(self, betas, gammas):
        bound_values = dict()
        for beta, i in zip(self.circ_parameters['betas'], range(self.n_layers)):
            bound_values[beta] = betas[i]
        for gamma, j in zip(self.circ_parameters['gammas'], range(self.n_layers)):
            bound_values[gamma] = gammas[j]
        bound_circuit = self.circ.bind_parameters(bound_values)
        return self.backend.run(bound_circuit, shots=self.shots).result().get_counts()

    def compute_expectation(self, counts):
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.objective(bitstring)
            avg += obj * count
            sum_count += count
        avg = avg/sum_count
        return avg

    def objective(self, x):
        A = nx.adjacency_matrix(self.graph).toarray()
        mu = -np.ones(self.n_qubits)@A
        sigma = A + np.diag(mu)
        x_arr = np.array([float(entry) for entry in x])
        return x_arr@sigma@x_arr

    def set_shots(self, shots):
        self.shots = shots

    def set_warm_start_method(self, method=None):
        self.warm_start_method = method

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def initialize_problem(self, graph, relaxed_solution=None):
        self.n_qubits = len(graph.nodes())
        self.graph = graph
        if self.warm_start_method is None:
            pass
        elif relaxed_solution is None:
            self.solve_relaxed()
        else:
            self.relaxed_bits = relaxed_solution[0]
            self.relaxed_obj = relaxed_solution[1]
            if len(relaxed_solution) > 2:
                self.bmz_angles = relaxed_solution[2]

        self.set_thetas()
        self.construct_circuit()

    def solve(self, graph, optimizer='COBYLA', relaxed_solution=None):
        self.initialize_problem(graph, relaxed_solution)

        initial_beta = np.random.uniform(-.001, .001, self.n_layers)
        initial_gamma = np.random.uniform(-.001, .001, self.n_layers)
        initial_params = np.concatenate((initial_beta, initial_gamma))

        def qaoa_objective(parameters):
            betas = parameters[:self.n_layers]
            gammas = parameters[self.n_layers:]
            counts = self.execute_circuit(betas, gammas)
            expectation = self.compute_expectation(counts)
            return expectation

        self.optimizer = optimizer
        if optimizer == 'COBYLA':
            optimized_params = minimize(qaoa_objective, initial_params, method='COBYLA').x
        elif optimizer == 'ADAM':
            maxiter = 1000
            adam = ADAM(maxiter=maxiter, amsgrad=True, tol=1e-2, lr=1e-1)
            # operator = Operator(self.circ)
            # params = [self.circ_parameters['betas'], self.circ_parameters['gammas']]
            # operator = CircuitStateFn(primitive=self.circ)
            # grad = Gradient(grad_method='lin_comb').convert(operator=operator, params=params)
            res = adam.minimize(qaoa_objective, initial_params)
            optimized_params = res.x
            # print(res.nfev)
            if res.nfev == maxiter:
                self.opt_ended_early = True
                print('Optimization terminated early.')
        else:
            raise(Exception('Accepted optimizers are \'ADAM\' and \'COBYLA\'.'))

        optimal_betas = optimized_params[:self.n_layers]
        optimal_gammas = optimized_params[self.n_layers:]
        counts = self.execute_circuit(optimal_betas, optimal_gammas)

        return_string = None
        best_val = np.inf
        for key in counts.keys():
            val = self.objective(str(key))
            if val < best_val:
                best_val = val
                return_string = str(key)

        res = QAOAResult()
        res.bitstring = return_string
        res.relaxed_bitstring = self.relaxed_bits
        res.obj = best_val
        res.relaxed_obj = self.relaxed_obj
        res.bmz_angles = self.bmz_angles
        res.n_regularized = self.n_regularized
        res.counts = counts
        res.expectation = self.compute_expectation(counts)
        res.unique_samples = len(counts)
        res.optimizer = optimizer
        res.opt_terminated = not self.opt_ended_early
        hyperparameters = dict()
        hyperparameters['epsilon'] = self.epsilon
        hyperparameters['n_qubits'] = self.n_qubits
        hyperparameters['n_layers'] = self.n_layers
        hyperparameters['shots'] = self.shots
        hyperparameters['warm_start_method'] = self.warm_start_method
        res.qaoa_hyperparameters = hyperparameters
        return res
