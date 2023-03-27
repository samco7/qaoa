from time import time
from quantum_solvers import *
from helper_functions import *
from save_results import *
from tqdm import tqdm
from qiskit.providers.aer import AerSimulator


def experiment_1(n_qubits, n_layers, n_trials, epsilons, backend=None, shots=512, save=True):
    expectations = np.zeros((n_trials, len(epsilons)))
    vals = np.copy(expectations)
    progress = tqdm(total=n_trials*len(epsilons))
    for i in range(len(epsilons)):
        counter = 0
        for j in range(n_trials):
            graph = random_graph(n_qubits)
            exact_val = akmaxsat(graph)[1]
            qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ', epsilon=epsilons[i], backend=backend, shots=shots)
            res = qaoa.solve(graph)
            expectations[counter, i] = res.expectation/exact_val
            vals[counter, i] = res.obj/exact_val
            counter += 1
            progress.update(1)
    progress.close()

    res = {'expectations':expectations, 'vals':vals, 'shots':shots, 'n_layers':n_layers, 'n_qubits':n_qubits, 'n_trials':n_trials, 'epsilons':epsilons}
    if save:
        save_result(res, 'experiment_1')
    return res


def plot_experiment_1(res, save=True):
    expectations, vals = res['expectations'], res['vals']
    epsilons = res['epsilons']
    epsilon_strings = [str(round(val, 4)) for val in epsilons]

    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    ax[0].boxplot(expectations, sym='.', positions=range(len(epsilons)),
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    ax[0].set_xticks(range(len(epsilons)), epsilon_strings)
    ax[0].set_xlabel(r'$\epsilon$ value', fontsize=12)
    ax[0].set_ylabel('Circuit expectation', fontsize=12)
    plt.grid(False, axis='x')

    ax[1].boxplot(vals, sym='.', positions = range(len(epsilons)),
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    ax[1].set_xticks(range(len(epsilons)), epsilon_strings)
    ax[1].set_xlabel(r'$\epsilon$ value', fontsize=12)
    ax[1].set_ylabel('Best sampled cut', fontsize=12)
    plt.grid(False, axis='x')
    plt.tight_layout()
    plt.suptitle(f'Experiment 1', fontsize=14, y=1.02)

    if save:
        save_plot('experiment_1')
    plt.show()
