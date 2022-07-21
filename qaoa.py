#import packages
from qiskit import *
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
import networkx as nx
import numpy as np
from os.path import exists
from scipy.optimize import fmin_cobyla as cobyla
from networkx.algorithms.approximation import maxcut as nxmaxcut
from time import time
from matplotlib import pyplot as plt
plt.style.use('seaborn')

class maxcut:

    def __init__(self, n_layers=5, shots=2**10):
        """
            Constructor. Accepts n_layers the number of cost and mixer layers,
            shots the numer of shots to do on the simulated quantum device
        """
        self.G = None
        self.nodes_points = None
        self.total_time = 0
        self.n_layers = n_layers
        self.shots = shots
        self.qc = 0

    def reset_time(self):
        self.total_time = 0

    def maxcut_obj(self, x):
        """
            Accepts a bitstring x and returns the cost function evaluated for
            that bitstring.
        """
        obj = 0
        if self.G is None:
            self.create_graph(draw=False)
        w = nx.get_edge_attributes(self.G, "weight")
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
        if self.G is None:
            self.create_graph(draw=False)
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
        # qc.barrier()

        for irep in range(0, self.n_layers):
            # problem unitary
            for pair in list(self.G.edges()):
                qc.cx(pair[0], pair[1])
                qc.rz(2*w[pair]*gamma[irep], pair[1])
                qc.cx(pair[0], pair[1])
                # qc.rzz(2*w[pair]*gamma[irep], pair[0], pair[1])
                # qc.barrier()

            # mixer unitary
            for i in self.G.nodes():
                # qc.h(i)
                # qc.rz(2 * beta[irep], i)
                # qc.h(i)
                qc.rx(2*beta[irep], i)
                # qc.barrier()

        qc.measure_all()
        self.qc = qc
        return qc

    # now we write a function to execute the circuit on the chosen backend
    def get_expectation(self):
        backend = Aer.get_backend('qasm_simulator')
        backend_opts = {'method':"matrix_product_state" , 'mps_sample_measure_qubits_opt':12}
        # backend = AerSimulator(
        #     method="statevector",
        #     device="GPU")

        def execute_circ(theta, return_result=False):
            qc = self.create_qaoa_circ(theta)
            result = backend.run(qc, seed_simulator=10, shots=self.shots, backend_options=backend_opts).result()
            # result = execute(qc, backend, shots=self.shots, backend_options=backend_opts).result()
            self.total_time += result.time_taken #increment our count of total estimated computation time each time the circuit is executed
            counts = result.get_counts()
            if return_result:
                return self.compute_expectation(counts), result
            else:
                return self.compute_expectation(counts)

        return execute_circ

    def create_distribution(self, points=None, n1=5, n2=5, shuffle=True, plot=False):
        '''
            Set up a random distribution with clusters centered at (10, 10) and
            (-10, -10).
            Parameters: cluster sizes n1 and n2, booleans indicating whether to
            shuffle/plot.
            Save as an attribute a dictionary that maps nodes to points (nodes are
            integer keys).
        '''
        # set up two random distributions with n_points total points and plot
        if points is None:
            n_points = n1 + n2
            set1  = np.random.normal(-10, 1, (2, n1))
            set2 = np.random.normal(10, 1, (2, n2))
            points = np.hstack((set1, set2))
            if shuffle:
                ind = np.arange(n_points)
                np.random.shuffle(ind)
                points = points[:, ind]
        nodes_points = dict()
        counter = 0
        for column in range(points.shape[1]):
            nodes_points[counter] = points[:, column]
            counter += 1
        if plot:
            plt.scatter(points[0, :], points[1, :], color='teal')
            plt.axis('equal')
            plt.show()
        self.nodes_points = nodes_points

    def create_graph(self, draw=False):
        '''
            Create a graph corresponding to a given point distribution.
            Parameters: nodes_points, a dictionary mapping integer nodes to n-dimensional
            points, and draw a boolean indicating whether to draw the graph we create.
            Returns the graph.
        '''
        # now set up edges with weights corresponding to distances between nodes/points
        G = nx.Graph()
        if self.nodes_points is None:
            self.create_distribution(plot=False)
        nodes = self.nodes_points.keys()
        for node1 in nodes:
            point1 = self.nodes_points[node1]
            for node2 in nodes:
                if node2 == node1:
                    continue
                point2 = self.nodes_points[node2]
                weight = np.linalg.norm(point1 - point2)
                G.add_edge(node1, node2, weight=weight)

        # draw the graph (note that weight is not reflected here as of now, that gets messy)
        if draw:
            nx.draw(G)
        self.G = G

    def optimize(self, verbose=False, compare=False):
        self.reset_time()
        return_info = dict()
        sig_figs = 4

        # get the expectation function from qaoa.py
        expectation = self.get_expectation()

        # load starting params if we've done this circuit size before
        n_points = len(self.nodes_points.keys())
        file_path = f'./saved_params/{n_points}_points_{self.n_layers}_layers_params.npy'
        if exists(file_path):
            initial_params = np.load(file_path)
        else:
            if verbose:
                print('No saved parameters. Starting with random values.')
            initial_beta = np.random.uniform(0, np.pi, self.n_layers)
            initial_gamma = np.random.uniform(0, 2*np.pi, self.n_layers)
            initial_params = np.concatenate((initial_beta, initial_gamma))

        # minimize the expectation by running the circuit and time the result
        start = time()
        sol = cobyla(expectation, initial_params,
                     (lambda x: x[:self.n_layers], lambda x: np.pi - x[:self.n_layers],
                     lambda x: x[self.n_layers:], lambda x: 2*np.pi - x[self.n_layers:]))
        result = expectation(sol, return_result=True)[1]
        counts = result.get_counts()
        sim_time = time() - start
        if verbose:
            print('Optimized Beta Parameters:\n', np.round(sol[:self.n_layers], sig_figs))
            print('Optimized Gamma Parameters:\n', np.round(sol[self.n_layers:], sig_figs))

        # get the optimal bitstring we found and the cut value it produces
        found_bitstring = list(counts.keys())[np.argmax(list(counts.values()))]
        found_min = self.maxcut_obj(found_bitstring)

        # save results in dictionary to return
        return_info['found_bitstring'] = found_bitstring
        return_info['found_min'] = found_min
        return_info['device_time'] = self.total_time
        return_info['sim_time'] = sim_time
        return_info['counts'] = counts

        if compare:
            # use a classical method to get the true optimal bitstring and time it
            start = time()
            true_sol = nxmaxcut.one_exchange(self.G, weight='weight')[1]
            classical_time = time() - start
            true_bitstring = ['0' for x in range(n_points)]
            for point in true_sol[0]:
                true_bitstring[point] = '1'
            true_bitstring = ''.join(true_bitstring)

            # save parameters if our result is somewhat decent
            true_min = self.maxcut_obj(true_bitstring)
            approx_ratio = np.abs(found_min/true_min)
            if approx_ratio >= .55:
                np.save(file_path, sol)

            #save results in dictionary to returns
            return_info['true_bitstring'] = true_bitstring
            return_info['true_min'] = true_min
            return_info['classical_time'] = classical_time
            return_info['approx_ratio'] = approx_ratio

        if verbose:
            if compare:
                print('\nTrue Optimal Bitstring: ' + true_bitstring)
            print('Found Bitstring: ' + found_bitstring)
            if compare:
                print('\nTrue Minimum: ' + str(round(true_min, sig_figs)))
            print('Found Minimum: ' + str(round(found_min, sig_figs)))
            if compare:
                print('Approximation Ratio: ' + str(round(approx_ratio, sig_figs)))
            print('\nSimulation Runtime: ' + str(round(sim_time, sig_figs)) + ' seconds')
            print('Estimated Device Time: ' + str(round(self.total_time, sig_figs)) + ' seconds')
            if compare:
                print('Classical Algorithm Time: ' + str(round(classical_time, sig_figs)) + ' seconds')

        return return_info

    def plot_clusters(self, found_bitstring, true_bitstring):
        # convert the result back to points separated by the cut we found above
        cluster0, cluster1 = [], []
        counter = 0
        for char in found_bitstring:
            if int(char) == 0:
                cluster0.append(self.nodes_points[counter])
            else:
                cluster1.append(self.nodes_points[counter])
            counter += 1

        # do the same for the true ideal clustering
        true_cluster0, true_cluster1 = [], []
        counter = 0
        for char in true_bitstring:
            if int(char) == 0:
                true_cluster0.append(self.nodes_points[counter])
            else:
                true_cluster1.append(self.nodes_points[counter])
            counter += 1

        # plot our points/clusters
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        cluster0 = np.vstack((cluster0)).T
        ax[0].scatter(cluster0[0, :], cluster0[1, :])
        cluster1 = np.vstack((cluster1)).T
        ax[0].scatter(cluster1[0, :], cluster1[1, :])
        ax[0].axis('equal')
        ax[0].set_title('Found Clusters', fontsize=18)
        true_cluster0 = np.vstack((true_cluster0)).T
        ax[1].scatter(true_cluster0[0, :], true_cluster0[1, :])
        true_cluster1 = np.vstack((true_cluster1)).T
        ax[1].scatter(true_cluster1[0, :], true_cluster1[1, :])
        ax[1].axis('equal')
        ax[1].set_title('Ideal Clusters', fontsize=18)
        plt.show()
