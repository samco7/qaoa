from time import time
from quantum_solvers import *
from helper_functions import *
from save_results import *
from tqdm import tqdm
from qiskit.providers.aer import AerSimulator


def experiment_4(n_qubits, n_layers, n_trials, perturbation_ratio, backend=None, shots=512, save=True):
    standard_expectations = []
    standard_vals = []

    state_only_expectations = []
    state_only_vals = []
    state_only_relaxed = []
    state_only_improvement = []

    bmz_rounded_expectations = []
    bmz_rounded_vals = []
    bmz_rounded_relaxed = []
    bmz_rounded_improvement = []

    bmz_expectations = []
    bmz_vals = []
    bmz_relaxed = []
    bmz_improvement = []

    exact_vals = []
    standard_results = []
    state_only_results = []
    bmz_rounded_results = []
    bmz_results = []

    progress = tqdm(total=n_trials*3)
    weights_set = list(range(1, 11))
    for i in range(n_trials):
        graph = random_graph(n_qubits, weights_set)
        exact_val = akmaxsat(graph)[1]
        exact_vals.append(exact_val)
        relaxed_solution = perturb_relaxed(perturbation_ratio, graph, BMZ(graph))

        standard_qaoa = QAOASolver(n_layers=n_layers, warm_start_method=None, epsilon=None, backend=backend, shots=shots)
        standard_res = standard_qaoa.solve(graph)
        standard_expectations.append(standard_res.expectation/exact_val)
        standard_vals.append(standard_res.obj/exact_val)
        standard_results.append(standard_res)
        progress.update(1)

        state_only_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ', epsilon=.2, backend=backend, shots=shots, adjust_mixer=False)
        state_only_res = state_only_qaoa.solve(graph, relaxed_solution=relaxed_solution)
        state_only_expectations.append(state_only_res.expectation/exact_val)
        state_only_vals.append(state_only_res.obj/exact_val)
        state_only_relaxed.append(state_only_res.relaxed_obj/exact_val)
        state_only_improvement.append((state_only_res.obj - state_only_res.relaxed_obj)/exact_val)
        state_only_results.append(state_only_res)

        bmz_rounded_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ Rounded', epsilon=.25, backend=backend, shots=shots)
        bmz_rounded_res = bmz_rounded_qaoa.solve(graph, relaxed_solution=relaxed_solution)
        bmz_rounded_expectations.append(bmz_rounded_res.expectation/exact_val)
        bmz_rounded_vals.append(bmz_rounded_res.obj/exact_val)
        bmz_rounded_relaxed.append(bmz_rounded_res.relaxed_obj/exact_val)
        bmz_rounded_improvement.append((bmz_rounded_res.obj - bmz_rounded_res.relaxed_obj)/exact_val)
        bmz_rounded_results.append(bmz_rounded_res)
        progress.update(1)

        bmz_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ', epsilon=.2, backend=backend, shots=shots)
        bmz_res = bmz_qaoa.solve(graph, relaxed_solution=relaxed_solution)
        bmz_expectations.append(bmz_res.expectation/exact_val)
        bmz_vals.append(bmz_res.obj/exact_val)
        bmz_relaxed.append(bmz_res.relaxed_obj/exact_val)
        bmz_improvement.append((bmz_res.obj - bmz_res.relaxed_obj)/exact_val)
        bmz_results.append(bmz_res)
        progress.update(1)
    progress.close()

    res = {'standard_results':standard_results,
    'state_only_results':state_only_results,
    'bmz_rounded_results':bmz_rounded_results,
    'bmz_results':bmz_results,
    'exact_vals':exact_vals,
    'standard_expectations':standard_expectations,
    'state_only_expectations':state_only_expectations,
    'bmz_rounded_expectations':bmz_rounded_expectations,
    'bmz_expectations':bmz_expectations,
    'standard_vals':standard_vals,
    'state_only_vals':state_only_vals,
    'bmz_rounded_vals':bmz_rounded_vals,
    'bmz_vals':bmz_vals,
    'state_only_relaxed':state_only_relaxed,
    'bmz_rounded_relaxed':bmz_rounded_relaxed,
    'bmz_relaxed':bmz_relaxed,
    'state_only_improvement':state_only_improvement,
    'bmz_rounded_improvement':bmz_rounded_improvement,
    'bmz_improvement':bmz_improvement,
    'shots':shots, 'n_layers':n_layers, 'n_qubits':n_qubits, 'n_trials':n_trials,
    'perturbation ratio':perturbation_ratio}
    info = {'n_qubits':n_qubits, 'n_layers':n_layers,'n_trials':n_trials, 'perturbation_ratio':perturbation_ratio, 'shots':shots, 'weights_set':weights_set}
    if save:
        save_result(res, 'experiment_4')
        save_info(info, 'experiment_4')
    return res

def plot_experiment_4(res, save=True):
    standard_expectations = res['standard_expectations']
    state_only_expectations = res['state_only_expectations']
    bmz_rounded_expectations = res['bmz_rounded_expectations']
    bmz_expectations = res ['bmz_expectations']
    standard_vals = res['standard_vals']
    state_only_vals = res['state_only_vals']
    bmz_rounded_vals = res['bmz_rounded_vals']
    bmz_vals = res['bmz_vals']
    state_only_relaxed = res['state_only_relaxed']
    bmz_rounded_relaxed = res['bmz_rounded_relaxed']
    bmz_relaxed = res['bmz_relaxed']
    state_only_improvement = res['state_only_improvement']
    bmz_rounded_improvement = res['bmz_rounded_improvement']
    bmz_improvement = res['bmz_improvement']

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    ax = ax.ravel()
    positions = [0, 2, 4, 6]

    ax[0].boxplot([standard_expectations, state_only_expectations, bmz_rounded_expectations, bmz_expectations], positions=positions, sym='.',
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    ax[0].set_xticks(positions, ['Standard QAOA', 'BMZ State Only', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA'])
    ax[0].set_ylabel('Circuit expectation', fontsize=12)
    ax[0].grid(False, axis='x')

    ax[1].boxplot([standard_vals, state_only_vals, bmz_rounded_vals, bmz_vals], positions=positions, sym='.',
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    ax[1].set_xticks(positions, ['Standard QAOA', 'BMZ State Only', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA'])
    ax[1].set_ylabel('Best sampled cut', fontsize=12)
    ax[1].grid(False, axis='x')

    ax[2].boxplot([state_only_relaxed, bmz_rounded_relaxed, bmz_relaxed], positions=positions[:-1], sym='.',
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    ax[2].set_xticks(positions[:-1], ['BMZ State Only', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA'])
    ax[2].set_ylabel('Relaxed accuracy (after perturbation)', fontsize=12)
    ax[2].grid(False, axis='x')

    ax[3].boxplot([state_only_improvement, bmz_rounded_improvement, bmz_improvement], positions=positions[:-1], sym='.',
    flierprops=dict(markeredgecolor='k'),
    medianprops=dict(color='steelblue'),
    boxprops=dict(color='k'),
    capprops=dict(color='k'),
    whiskerprops=dict(color='k'))
    ax[3].set_xticks(positions[:-1], ['BMZ State Only', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA'])
    ax[3].set_ylabel('Improvement', fontsize=12)
    ax[3].grid(False, axis='x')

    plt.tight_layout()
    plt.suptitle(f'Experiment 4', fontsize=14, y=1.02)
    if save:
        save_plot('experiment_4')
    plt.show()
