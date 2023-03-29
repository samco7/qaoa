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


def plot_experiment_2(res, n_bins=20, save=True):
    standard_expectations = res['standard_expectations']
    gw_rounded_expectations = res['gw_rounded_expectations']
    bmz_rounded_expectations = res['bmz_rounded_expectations']
    bmz_expectations = res['bmz_expectations']
    standard_vals = res['standard_vals']
    gw_rounded_vals = res['gw_rounded_vals']
    bmz_rounded_vals = res['bmz_rounded_vals']
    bmz_vals = res['bmz_vals']
    n_trials = res['n_trials']
    labels = ['Standard QAOA', 'GW-Rounded-WS-QAOA', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA']

    vals = np.round(np.vstack((standard_vals, gw_rounded_vals, bmz_rounded_vals, bmz_vals)).T, 6)
    expectations = np.vstack((standard_expectations, gw_rounded_expectations, bmz_rounded_expectations, bmz_expectations)).T
    min_val = np.round(np.min(vals), 2)
    min_expectation = np.round(np.min(expectations), 2)

    other_vals = np.copy(vals)
    perfect_vals = np.full((n_trials, 4), np.nan)
    perfect_vals_index = np.where(vals == 1)
    other_vals[perfect_vals_index] = np.nan
    perfect_vals[perfect_vals_index] = 1

    other_expectations = np.copy(expectations)
    perfect_expectations = np.full((n_trials, 4), np.nan)
    perfect_expectation_index = np.where(expectations == 1)
    other_expectations[perfect_expectation_index] = np.nan
    perfect_expectations[perfect_expectation_index] = 1

    bin_width = (1 - min_val)/n_bins
    bins = np.arange(min_val, 1 + bin_width, bin_width)
    width_ratio = bin_width/(1 - min_val)
    fig, ax = plt.subplots(1, 2, width_ratios=[1 - width_ratio, width_ratio], figsize=(9, 3))
    ax = ax.ravel()

    other_val_hist = ax[0].hist(other_vals, bins=bins, align='mid')
    ax[0].set_xticks(np.linspace(0, 1, 51))
    ax[0].set_xlim(min_val, .9999)
    ax[0].set_ylim(0, 2*np.max(other_val_hist[0]))
    ax[0].grid(False, axis='x')
    ax[0].set_xlabel('Cut size (% of maximum cut)', fontsize=12)
    ax[0].set_ylabel('Counts', fontsize=12)
    ax[0].legend(labels, fontsize=12, loc='upper center')
    ax[0].set_title('(a)', fontsize=12, x=.02, y=.9, backgroundcolor='white')

    perfect_val_hist = ax[1].hist(perfect_vals, bins=[1 - bin_width/2, 1 + bin_width/2], align='mid')
    ax[1].set_xlim(1 - bin_width/2, 1 + bin_width/2)
    ax[1].set_xticks([1], ['1.0'])
    ax[1].yaxis.tick_right()
    ax[1].grid(False, axis='x')
    ax[1].set_ylabel('Max cut counts', fontsize=12)
    ax[1].yaxis.set_label_position('right')

    plt.tight_layout()
    if save:
        save_plot('experiment_2', 'best_measurement')
    plt.show()

    bin_width = (1 - min_expectation)/n_bins
    bins = np.arange(min_expectation, 1 + bin_width, bin_width)
    width_ratio = bin_width/(1 - min_expectation)
    fig, ax = plt.subplots(1, 2, width_ratios=[1 - width_ratio, width_ratio], figsize=(9, 3))
    ax = ax.ravel()

    other_expectation_hist = ax[0].hist(other_expectations, bins=bins, align='mid')
    ax[0].set_xticks(np.linspace(0, 1, 51))
    ax[0].set_xlim(min_expectation, .9999)
    ax[0].set_ylim(0, 2*np.max(other_expectation_hist[0]))
    ax[0].grid(False, axis='x')
    ax[0].set_xlabel('Cut size (% of maximum cut)', fontsize=12)
    ax[0].set_ylabel('Counts', fontsize=12)
    ax[0].legend(labels, fontsize=12, loc='upper center')
    ax[0].set_title('(b)', fontsize=12, x=.02, y=.9, backgroundcolor='white')

    perfect_expectation_hist = ax[1].hist(perfect_expectations, bins=[1 - bin_width/2, 1 + bin_width/2], align='mid')
    ax[1].set_xlim(1 - bin_width/2, 1 + bin_width/2)
    ax[1].set_xticks([1], ['1.0'])
    ax[1].yaxis.tick_right()
    ax[1].grid(False, axis='x')
    ax[1].set_ylabel('Max cut counts', fontsize=12)
    ax[1].yaxis.set_label_position('right')

    plt.tight_layout()
    if save:
        save_plot('experiment_2', 'circuit_expectation')
    plt.show()

# def plot_experiment_2(res, save=True):
#     standard_expectations = res['standard_expectations']
#     gw_rounded_expectations = res['gw_rounded_expectations']
#     bmz_rounded_expectations = res['bmz_rounded_expectations']
#     bmz_expectations = res['bmz_expectations']
#     standard_vals = res['standard_vals']
#     gw_rounded_vals = res['gw_rounded_vals']
#     bmz_rounded_vals = res['bmz_rounded_vals']
#     bmz_vals = res['bmz_vals']

#     fig, ax = plt.subplots(2, 1, figsize=(8, 10))
#     positions = [0, 2, 4, 6]

#     ax[0].boxplot([standard_expectations, gw_rounded_expectations, bmz_rounded_expectations, bmz_expectations], positions=positions, sym='.',
#     flierprops=dict(markeredgecolor='k'),
#     medianprops=dict(color='steelblue'),
#     boxprops=dict(color='k'),
#     capprops=dict(color='k'),
#     whiskerprops=dict(color='k'))
#     ax[0].set_xticks(positions, ['Standard QAOA', 'GW-Rounded-WS-QAOA', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA'])
#     ax[0].set_ylabel('Circuit expectation', fontsize=12)
#     plt.grid(False, axis='x')

#     ax[1].boxplot([standard_vals, gw_rounded_vals, bmz_rounded_vals, bmz_vals], positions=positions, sym='.',
#     flierprops=dict(markeredgecolor='k'),
#     medianprops=dict(color='steelblue'),
#     boxprops=dict(color='k'),
#     capprops=dict(color='k'),
#     whiskerprops=dict(color='k'))
#     ax[1].set_xticks(positions, ['Standard QAOA', 'GW-Rounded-WS-QAOA', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA'])
#     ax[1].set_ylabel('Best sampled cut', fontsize=12)
#     plt.grid(False, axis='x')
#     plt.suptitle(f'Experiment 2', fontsize=14, y=1.02)
#     plt.tight_layout()

#     if save:
#         save_plot('experiment_2')
#     plt.show()
