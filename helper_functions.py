from classical_solvers import *
import numpy as np
import networkx as nx
from docplex.mp.model import Model
import numpy as np
import networkx as nx
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from matplotlib import pyplot as plt
plt.style.use('seaborn-v0_8')


def objective(graph, x):
        A = nx.adjacency_matrix(graph).toarray()
        mu = -np.ones(len(graph.nodes()))@A
        sigma = A + np.diag(mu)
        x_arr = np.array([float(entry) for entry in x])
        return x_arr@sigma@x_arr


def perturb_relaxed(perturbation_ratio, graph, relaxed_solution):
    if len(relaxed_solution) > 2:
        relaxed_bits, relaxed_obj, bmz_angles = relaxed_solution
    else:
        relaxed_bits, relaxed_obj = relaxed_solution
        bmz_angles = None

    exact_obj = akmaxsat(graph)[1]
    while relaxed_obj/exact_obj > perturbation_ratio:
        idx = np.random.choice(list(range(len(relaxed_bits))), len(relaxed_bits))[0]
        relaxed_bits[idx] = (relaxed_bits[idx] + 1)%2
        relaxed_obj = objective(graph, relaxed_bits)
        if bmz_angles is not None:
            bmz_angles[idx] = np.mod(bmz_angles[idx] + np.pi, 2*np.pi)

    if bmz_angles is not None:
        return relaxed_bits, relaxed_obj, bmz_angles
    else:
        return relaxed_bits, relaxed_obj


def random_graph(n_nodes, a=0, b=10):
    G = nx.Graph()
    nodes = list(range(n_nodes))
    weights = np.random.uniform(a, b, n_nodes*(n_nodes - 1))
    counter = 0
    for node1 in nodes:
        for node2 in nodes:
            if node1 != node2:
                w = np.random.uniform(a, b)
                G.add_edge(node1, node2, weight=weights[counter])
                counter += 1
    return G


def create_clusters(points=None, c1=(0, 1), c2=(0, -1), s1=(1, 2), s2=(1, 2), n1=5, n2=5):
    """
        If an array of points is passed in, create a dictionary mapping integer nodes to cartesian points.
        Otherwise, set up a random distribution with clusters centered at c1 and c2.

        Parameters:
            c1 (tuple): center of the first cluster
            c2 (tuple): center of the second cluster
            s1 (tuple): standard deviation of the first cluster
            s2 (tuple): standard deviation of the second cluster
            n1 (int): number of points in first cluster
            n2 (int): number of points in second cluster

        Returns:
            nodes_points (dict): a dictionary mapping integer nodes to cartesian points
    """
    # set up two random distributions with n_points total points and plot
    if points is None:
        set1  = np.random.normal(c1, s1, (n1, 2))
        set2  = np.random.normal(c2, s2, (n2, 2))
        points = np.vstack((set1, set2)).T
        ind = list(range(points.shape[1]))
        np.random.shuffle(ind)
        points = points[:, ind]
    nodes_points = dict()
    counter = 0
    for column in range(points.shape[1]):
        nodes_points[counter] = points[:, column]
        counter += 1
    return nodes_points


def dist_to_graph(nodes_points):
    """
        Creates a graph corresponding to a given point distribution.

        Parameters:
            nodes_points (dict): a dictionary mapping integer nodes to n-dimensional points. Should be
                of the form returned by the create_distribution function above.

        Returns:
            G (networkx graph): the graph corresponding to the distribution passed in above.
    """
    # now set up edges with weights corresponding to distances between nodes/points
    G = nx.Graph()
    nodes = nodes_points.keys()
    for node1 in nodes:
        point1 = nodes_points[node1]
        for node2 in nodes:
            if node2 == node1:
                continue
            point2 = nodes_points[node2]
            weight = np.linalg.norm(point1 - point2)
            G.add_edge(node1, node2, weight=weight)
    return G


def get_ratio_counts(counts, graph):
    exact_string, exact_val = akmaxsat(graph)
    ratio_counts = dict()
    for count in counts.keys():
        val = objective(graph, count)
        ratio = round(val/exact_val, 4)
        if ratio in ratio_counts.keys():
            ratio_counts[ratio] += counts[count]
        else:
            ratio_counts[ratio] = counts[count]
    return ratio_counts


def plot_clusters(bitstring_1, bitstring_2, nodes_points, title_1, title_2):
    """
        Plots the found clusters from qaoa alongside the actual solution clusters.

        Parameters:
            bitstring_1 (str): first bitstring to plot
            bitstring_2 (str): second bitstring to plot
                nodes_points (dict): a dictionary mapping graph nodes to euclidean points (this is of the form
                created by the create_distribution function in the file initialize_graph.py in this repository.)
            title_1 (str): title of first subplot
            title_2 (str): title of second subplot
    """
    # convert the result back to points separated by the cut we found above
    cluster0, cluster1, orig_data = [], [], []
    counter = 0
    for char in bitstring_1:
        orig_data.append(nodes_points[counter])
        if int(char) == 0:
            cluster0.append(nodes_points[counter])
        else:
            cluster1.append(nodes_points[counter])
        counter += 1

    # do the same for the gw clustering
    true_cluster0, true_cluster1 = [], []
    counter = 0
    for char in bitstring_2:
        if int(char) == 0:
            true_cluster0.append(nodes_points[counter])
        else:
            true_cluster1.append(nodes_points[counter])
        counter += 1

    # plot our points/clusters
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    orig_data = np.vstack((orig_data)).T
    ax[0].scatter(orig_data[0, :], orig_data[1, :], c='k')
    ax[0].axis('equal')
    ax[0].set_title('Origial Data Points', fontsize=18)
    cluster0 = np.vstack((cluster0)).T
    ax[1].scatter(cluster0[0, :], cluster0[1, :])
    cluster1 = np.vstack((cluster1)).T
    ax[1].scatter(cluster1[0, :], cluster1[1, :])
    ax[1].axis('equal')
    ax[1].set_title(title_1, fontsize=18)
    true_cluster0 = np.vstack((true_cluster0)).T
    ax[2].scatter(true_cluster0[0, :], true_cluster0[1, :])
    true_cluster1 = np.vstack((true_cluster1)).T
    ax[2].scatter(true_cluster1[0, :], true_cluster1[1, :])
    ax[2].axis('equal')
    ax[2].set_title(title_2, fontsize=18)
    return fig
