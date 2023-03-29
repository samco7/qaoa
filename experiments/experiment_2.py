from time import time
from quantum_solvers import *
from helper_functions import *
from save_results import *
from tqdm import tqdm
from qiskit.providers.aer import AerSimulator


def experiment_2(n_qubits, n_layers, n_trials, backend=None, shots=512, save=True):
    standard_expectations = []
    standard_vals = []
    standard_results = []

    gw_rounded_expectations = []
    gw_rounded_vals = []
    gw_rounded_results = []

    bmz_rounded_expectations = []
    bmz_rounded_vals = []
    bmz_rounded_results = []

    bmz_expectations = []
    bmz_vals = []
    bmz_results = []

    progress = tqdm(total=n_trials*4)
    for i in range(n_trials):
        graph = random_graph(n_qubits)
        exact_val = akmaxsat(graph)[1]

        standard_qaoa = QAOASolver(n_layers=n_layers, warm_start_method=None, epsilon=None, backend=backend, shots=shots)
        gw_rounded_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='GW Rounded', epsilon=.25, backend=backend, shots=shots)
        bmz_rounded_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ Rounded', epsilon=.25, backend=backend, shots=shots)
        bmz_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ', epsilon=.2, backend=backend, shots=shots)

        standard_res = standard_qaoa.solve(graph)
        standard_expectations.append(standard_res.expectation/exact_val)
        standard_vals.append(standard_res.obj/exact_val)
        standard_results.append(standard_res)
        progress.update(1)

        gw_rounded_res = gw_rounded_qaoa.solve(graph)
        gw_rounded_expectations.append(gw_rounded_res.expectation/exact_val)
        gw_rounded_vals.append(gw_rounded_res.obj/exact_val)
        gw_rounded_results.append(gw_rounded_res)
        progress.update(1)

        bmz_relaxed = BMZ(graph)

        bmz_rounded_res = bmz_rounded_qaoa.solve(graph, relaxed_solution=bmz_relaxed)
        bmz_rounded_expectations.append(bmz_rounded_res.expectation/exact_val)
        bmz_rounded_vals.append(bmz_rounded_res.obj/exact_val)
        bmz_rounded_results.append(bmz_rounded_res)
        progress.update(1)

        bmz_res = bmz_qaoa.solve(graph, relaxed_solution=bmz_relaxed)
        bmz_expectations.append(bmz_res.expectation/exact_val)
        bmz_vals.append(bmz_res.obj/exact_val)
        bmz_results.append(bmz_res)
        progress.update(1)
    progress.close()

    res = {'standard_expectations':standard_expectations,
        'gw_rounded_expectations':gw_rounded_expectations,
        'bmz_rounded_expectations':bmz_rounded_expectations,
        'bmz_expectations':bmz_expectations,
        'standard_vals':standard_vals,
        'gw_rounded_vals':gw_rounded_vals,
        'bmz_rounded_vals':bmz_rounded_vals,
        'bmz_vals':bmz_vals,
        'standard_results':standard_results,
        'gw_rounded_results':gw_rounded_results,
        'bmz_rounded_results':bmz_rounded_results,
        'bmz_results':bmz_results,
        'shots':shots, 'n_layers':n_layers, 'n_qubits':n_qubits, 'n_trials':n_trials}
    info = {'n_qubits':n_qubits, 'n_layers':n_layers,'n_trials':n_trials, 'shots':shots}
    if save:
        save_result(res, 'experiment_2')
        save_info(info, 'experiment_2')
    return res


def plot_experiment_2(res, save=True):
    standard_expectations = res['standard_expectations']
    gw_rounded_expectations = res['gw_rounded_expectations']
    bmz_rounded_expectations = res['bmz_rounded_expectations']
    bmz_expectations = res['bmz_expectations']
    standard_vals = res['standard_vals']
    gw_rounded_vals = res['gw_rounded_vals']
    bmz_rounded_vals = res['bmz_rounded_vals']
    bmz_vals = res['bmz_vals']

    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    positions = [0, 2, 4, 6]

    ax[0].boxplot([standard_expectations, gw_rounded_expectations, bmz_rounded_expectations, bmz_expectations], positions=positions, sym='.',
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    ax[0].set_xticks(positions, ['Standard QAOA', 'GW-Rounded-WS-QAOA', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA'])
    ax[0].set_ylabel('Circuit expectation', fontsize=12)
    plt.grid(False, axis='x')

    ax[1].boxplot([standard_vals, gw_rounded_vals, bmz_rounded_vals, bmz_vals], positions=positions, sym='.',
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    ax[1].set_xticks(positions, ['Standard QAOA', 'GW-Rounded-WS-QAOA', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA'])
    ax[1].set_ylabel('Best sampled cut', fontsize=12)
    plt.grid(False, axis='x')
    plt.suptitle(f'Experiment 2', fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        save_plot('experiment_2')
    plt.show()
