import sys, os
import cvxpy as cp
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from pyakmaxsat import AKMaxSATSolver
from pyqubo import Array


def BMZ(graph):
    W = nx.adjacency_matrix(graph).toarray()
    N = len(W)

    def T(theta):
        matrix = np.empty((N, N))
        for i in range(N):
            for j in range(N):
                matrix[i, j] = theta[i] - theta[j]
        return matrix

    def f(theta):
        return 1/2*np.sum(W*np.cos(T(theta)))

    def df(theta):
        return (W*np.sin(T(theta))).T@np.ones(N)

    # def d2f(theta):
    #     return W*np.cos(T(theta)) - np.diag((W*np.cos(T(theta)))@np.ones(N))

    def generate_cut(theta, alpha):
        theta = np.mod(theta - alpha, 2*np.pi)
        x = np.ones(N)
        for i in range(N):
            if theta[i] >= np.pi/2 and theta[i] < 3*np.pi/2:
                x[i] = -1
        return x

    def objective(x):
        obj = 0
        for i in range(N):
            for j in range(N):
                obj += W[i, j]*(1 - x[i]*x[j])
        return obj/4

    init_theta = np.random.uniform(0, 2*np.pi, N)
    theta = minimize(f, init_theta, method='L-BFGS-B', jac=df).x
    theta = np.mod(theta, 2*np.pi)
    theta_sorted = np.sort(theta)
    alpha = 0
    gamma = -np.inf
    i = 1
    j_list = np.where(theta_sorted > np.pi)[0]
    if len(j_list) == 0:
        j = N
    else:
        j = j_list[0]
    theta_sorted = np.concatenate((theta_sorted, [2*np.pi]))
    while alpha <= np.pi and j <= N:
        x = generate_cut(theta, alpha)
        obj = objective(x)
        if obj > gamma:
            gamma = obj
            x_star = x
            alpha_star = alpha
        if theta_sorted[i] <= theta_sorted[j] - np.pi:
            alpha = theta_sorted[i]
            i += 1
        else:
            alpha = theta_sorted[j] - np.pi
            j += 1
    bitstring = np.ones(N)
    bitstring[np.where(x_star == -1)] = 0
    return bitstring, -gamma, np.mod(theta - alpha_star, 2*np.pi),


def GW(graph):
    """
        Solves the relaxed optimization problem using Goemans-Williamson.
    """
    # solve the semidefinite programming problem from the book
    A = nx.adjacency_matrix(graph).toarray()
    mu = -np.ones(len(graph.nodes())) @ A
    sigma = A + np.diag(mu)
    N = len(sigma)
    Y = cp.Variable((N, N), symmetric=True)
    objective = cp.Maximize(cp.trace(-sigma @ Y))
    constraints = [cp.constraints.psd.PSD(Y), cp.diag(Y) == np.ones(N)]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    Y_sol = Y.value

    # convert to the form from Egger, Marecek, and Woerner appendix B (and the book mentioned in the comment above) using eigendecomposition
    d, Q = np.linalg.eig(Y_sol)
    root_d = np.sqrt(np.diag(d) @ np.diag(d >= 0))
    V = Q @ root_d

    # now get the actual relaxed solution using hyperplane rounding as in Goemans-Williamson
    relaxed_sol = np.empty(N)
    r = np.random.uniform(-1, 1, N)
    r /= np.linalg.norm(r, 2)
    for i in range(N):
        relaxed_sol[i] = (r @ V[i, :])
    # relaxed_sol = relaxed_sol[::-1]

    # the above is for a setup where partitions correspond to x_i either 1 or -1, and the relaxed variables take values between -1 and 1. We now round to get binary variables 0 or 1 in each spot, completing the GW algorithm.
    new_relaxed_sol = np.zeros(N)
    new_relaxed_sol[np.where(relaxed_sol > 0)] = 1
    return new_relaxed_sol, new_relaxed_sol@sigma@new_relaxed_sol


class suppress_output(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def akmaxsat(graph):
    n = len(graph.nodes())
    A = nx.adjacency_matrix(graph).toarray()
    mu = -np.ones(n) @ A
    sigma = A + np.diag(mu)
    x = Array.create('x', n, 'BINARY')
    qubo = sum([sigma[i, j] * x[i] * x[j] for i in range(n) for j in range(n)])
    model = qubo.compile().to_bqm()
    solver = AKMaxSATSolver()
    with suppress_output():
        sampleset = solver.sample(model)
    string = ''
    for i in range(n):
        bit = list(sampleset.samples())[0][f'x[{i}]']
        string += str(bit)
    vec = np.array([float(entry) for entry in string])
    val = vec@sigma@vec
    return string, val


def naive_exact_maxcut(graph):
    """
        Gets the true optimal maximum cut naively by trying all possible cuts and taking the best one.
    """
    A = nx.adjacency_matrix(graph).toarray()
    mu = -np.ones(len(graph.nodes())) @ A
    sigma = A + np.diag(mu)

    def objective(x):
        x = np.array([float(entry) for entry in x])
        return x@sigma@x

    N = len(sigma)
    # function to generate all binary strings of length n
    saved = []
    def get_all_bitstrings(n, arr, i=0):
        if i == n:
            bitstring = ''.join(str(x) for x in arr)
            saved.append(bitstring)
            return

        # first assign "0" at ith position and try for all other permutations for remaining positions
        arr[i] = 0
        get_all_bitstrings(n, arr, i + 1)

        # and then assign "1" at ith position and try for all other permutations for remaining positions
        arr[i] = 1
        get_all_bitstrings(n, arr, i + 1)

    # get all bitstrings and initialize min and opt variables
    get_all_bitstrings(N, [None]*N)
    minimum = np.inf
    best_string = None

    # iterate through all the options, saving any improvements, then return the optimal bitstring
    for bitstring in saved:
        cut_val = objective(bitstring)
        if cut_val < minimum:
            minimum = cut_val
            best_string = bitstring
    return best_string, minimum
