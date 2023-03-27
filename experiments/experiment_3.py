from time import time
from quantum_solvers import *
from helper_functions import *
from save_results import *
from tqdm import tqdm
from qiskit.providers.aer import AerSimulator


def experiment_3(n_qubits, n_layers, n_trials, shot_vals, backend=None, save=True):
    vals = np.zeros((n_trials, len(shot_vals)*4))
    progress = tqdm(total=len(shot_vals)*n_trials*4)
    for i in range(len(shot_vals)):
        for j in range(n_trials):
            graph = random_graph(n_qubits)
            exact_val = akmaxsat(graph)[1]

            standard_qaoa = QAOASolver(n_layers, warm_start_method=None, epsilon=None, shots=shot_vals[i], backend=backend)
            gw_rounded_qaoa = QAOASolver(n_layers, warm_start_method='GW Rounded', epsilon=.25, shots=shot_vals[i], backend=backend)
            bmz_rounded_qaoa = QAOASolver(n_layers, warm_start_method='BMZ Rounded', epsilon=.25, shots=shot_vals[i], backend=backend)
            bmz_qaoa = QAOASolver(n_layers, warm_start_method='BMZ', epsilon=.2, shots=shot_vals[i], backend=backend)

            standard_res = standard_qaoa.solve(graph)
            progress.update(1)
            gw_rounded_res = gw_rounded_qaoa.solve(graph)
            progress.update(1)

            bmz_relaxed = BMZ(graph)

            bmz_rounded_res = bmz_rounded_qaoa.solve(graph, relaxed_solution=bmz_relaxed)
            progress.update(1)
            bmz_res = bmz_qaoa.solve(graph, relaxed_solution=bmz_relaxed)
            progress.update(1)

            vals[j, i] = standard_res.obj/exact_val
            vals[j, len(shot_vals) + i] = gw_rounded_res.obj/exact_val
            vals[j, 2*len(shot_vals) + i] = bmz_rounded_res.obj/exact_val
            vals[j, 3*len(shot_vals) + i] = bmz_res.obj/exact_val
    progress.close()

    res = {'vals':vals, 'shot_vals':shot_vals, 'n_layers':n_layers, 'n_qubits':n_qubits, 'n_trials':n_trials}
    if save:
        save_result(res, 'experiment_3')
    return res


def plot_experiment_3(res, save=True):
    shot_vals = res['shot_vals']
    vals = res['vals']

    fig = plt.figure(figsize=(14, 5))
    algorithms = ['Standard QAOA', 'GW-Rounded-WS-QAOA', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA']
    positions = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    labels = []
    #inelegantbutworks
    counter = 0
    for i in range(len(positions)):
        idx = i%len(shot_vals)
        if idx == 1:
            labels.append(str(shot_vals[idx]) + ' shots\n\n' + str(algorithms[counter]))
            counter += 1
        else:
            labels.append(str(shot_vals[idx]) + ' shots')

    plt.boxplot(vals, positions=positions, sym='.',
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    plt.xticks(positions, labels)
    plt.ylabel('Best sampled cut', fontsize=12)
    plt.grid(False, axis='x')
    plt.title(f'Experiment 3', fontsize=14, y=1.02)

    if save:
        save_plot('experiment_3')
    plt.show()
