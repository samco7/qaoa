from time import time
from quantum_solvers import *
from helper_functions import *
from save_results import *
from tqdm import tqdm
from qiskit.providers.aer import AerSimulator


def experiment_5(n_qubits, n_trials, layer_vals, backend=None, shots=512, save=True):
    expectations = np.zeros((n_trials, len(layer_vals)*4))
    vals = np.copy(expectations)
    progress = tqdm(total=len(layer_vals)*n_trials*4)
    for i in range(len(layer_vals)):
        for j in range(n_trials):
            graph = random_graph(n_qubits)
            exact_val = akmaxsat(graph)[1]

            standard_qaoa = QAOASolver(layer_vals[i], warm_start_method=None, epsilon=None, shots=shots, backend=backend)
            gw_rounded_qaoa = QAOASolver(layer_vals[i], warm_start_method='GW Rounded', epsilon=.25, shots=shots, backend=backend)
            bmz_rounded_qaoa = QAOASolver(layer_vals[i], warm_start_method='BMZ Rounded', epsilon=.25, shots=shots, backend=backend)
            bmz_qaoa = QAOASolver(layer_vals[i], warm_start_method='BMZ', epsilon=.2, shots=shots, backend=backend)

            standard_res = standard_qaoa.solve(graph)
            progress.update(1)
            gw_rounded_res = gw_rounded_qaoa.solve(graph)
            progress.update(1)

            bmz_relaxed = BMZ(graph)

            bmz_rounded_res = bmz_rounded_qaoa.solve(graph, relaxed_solution=bmz_relaxed)
            progress.update(1)
            bmz_res = bmz_qaoa.solve(graph, relaxed_solution=bmz_relaxed)
            progress.update(1)

            expectations[j, i] = standard_res.expectation/exact_val
            expectations[j, len(layer_vals) + i] = gw_rounded_res.expectation/exact_val
            expectations[j, 2*len(layer_vals) + i] = bmz_rounded_res.expectation/exact_val
            expectations[j, 3*len(layer_vals) + i] = bmz_res.expectation/exact_val

            vals[j, i] = standard_res.obj/exact_val
            vals[j, len(layer_vals) + i] = gw_rounded_res.obj/exact_val
            vals[j, 2*len(layer_vals) + i] = bmz_rounded_res.obj/exact_val
            vals[j, 3*len(layer_vals) + i] = bmz_res.obj/exact_val
    progress.close()

    res = {'expectations':expectations, 'vals':vals, 'layer_vals':layer_vals, 'shots':shots, 'n_qubits':n_qubits, 'n_trials':n_trials}
    info = {'n_qubits':n_qubits, 'layer_vals':layer_vals,'n_trials':n_trials, 'shots':shots}
    if save:
        save_result(res, 'experiment_5')
        save_info(info, 'experiment_5')
    return res


def plot_experiment_5(res, save=True):
    layer_vals = res['layer_vals']
    N = len(layer_vals)
    expectations = res['expectations']
    vals = res['vals']

    fig, ax = plt.subplots(2, 1, figsize=(14, 5))
    grouping = np.arange(1, N + 1, 1)

    positions = np.concatenate((grouping, grouping + N + 1, grouping + 2*(N + 1), grouping + 3* (N + 1)))
    algorithms = ['Standard QAOA', 'GW-Rounded-WS-QAOA', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA']
    labels = []
    #inelegantbutworks
    counter = 0
    for i in range(len(positions)):
        idx = i%len(layer_vals)
        if idx == len(layer_vals)//2:
            labels.append(str(layer_vals[idx]) + ' layers\n\n' + str(algorithms[counter]))
            counter += 1
        else:
            labels.append(str(layer_vals[idx]) + ' layers')

    ax[0].boxplot(expectations, positions=positions, sym='.',
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    ax[0].set_xticks(positions, labels)
    ax[0].set_ylabel('Circuit expectation', fontsize=12)
    plt.grid(False, axis='x')

    ax[1].boxplot(vals, positions=positions, sym='.',
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    ax[1].set_xticks(positions, labels)
    ax[1].set_ylabel('Best sampled cut', fontsize=12)
    plt.grid(False, axis='x')

    plt.suptitle(f'Experiment 5', fontsize=14, y=1.02)
    plt.tight_layout()
    if save:
        save_plot('experiment_5')
    plt.show()
