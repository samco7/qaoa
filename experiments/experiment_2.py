from time import time
from quantum_solvers import *
from helper_functions import *
from save_results import *
from tqdm import tqdm
from seaborn import color_palette as cp
from qiskit.providers.aer import AerSimulator


def experiment_2(n_qubits, n_layers, n_trials, optimizer='COBYLA', backend=None, shots=512, save=True):
    standard_expectations = []
    standard_vals = []
    standard_results = []

    gw_rounded_expectations = []
    gw_rounded_vals = []
    gw_rounded_results = []

    state_only_expectations = []
    state_only_vals = []
    state_only_results = []

    bmz_rounded_expectations = []
    bmz_rounded_vals = []
    bmz_rounded_results = []

    bmz_expectations = []
    bmz_vals = []
    bmz_results = []

    weights_set = list(range(1, 11))
    progress = tqdm(total=n_trials*5)
    for i in range(n_trials):
        graph = random_graph(n_qubits, weights_set)
        exact_val = akmaxsat(graph)[1]

        standard_qaoa = QAOASolver(n_layers=n_layers, warm_start_method=None, epsilon=None, backend=backend, shots=shots)
        gw_rounded_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='GW Rounded', epsilon=.25, backend=backend, shots=shots)
        state_only_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ', epsilon=.2, backend=backend, shots=shots, adjust_mixer=False)
        bmz_rounded_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ Rounded', epsilon=.25, backend=backend, shots=shots)
        bmz_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ', epsilon=.2, backend=backend, shots=shots)

        standard_res = standard_qaoa.solve(graph, optimizer=optimizer)
        standard_expectations.append(standard_res.expectation/exact_val)
        standard_vals.append(standard_res.obj/exact_val)
        standard_results.append(standard_res)
        progress.update(1)

        gw_rounded_res = gw_rounded_qaoa.solve(graph, optimizer=optimizer)
        gw_rounded_expectations.append(gw_rounded_res.expectation/exact_val)
        gw_rounded_vals.append(gw_rounded_res.obj/exact_val)
        gw_rounded_results.append(gw_rounded_res)
        progress.update(1)

        bmz_relaxed = BMZ(graph)

        state_only_res = state_only_qaoa.solve(graph, relaxed_solution=bmz_relaxed, optimizer=optimizer)
        state_only_expectations.append(state_only_res.expectation/exact_val)
        state_only_vals.append(state_only_res.obj/exact_val)
        state_only_results.append(state_only_res)
        progress.update(1)

        bmz_rounded_res = bmz_rounded_qaoa.solve(graph, relaxed_solution=bmz_relaxed, optimizer=optimizer)
        bmz_rounded_expectations.append(bmz_rounded_res.expectation/exact_val)
        bmz_rounded_vals.append(bmz_rounded_res.obj/exact_val)
        bmz_rounded_results.append(bmz_rounded_res)
        progress.update(1)

        bmz_res = bmz_qaoa.solve(graph, relaxed_solution=bmz_relaxed, optimizer=optimizer)
        bmz_expectations.append(bmz_res.expectation/exact_val)
        bmz_vals.append(bmz_res.obj/exact_val)
        bmz_results.append(bmz_res)
        progress.update(1)
    progress.close()

    res = {'standard_expectations':standard_expectations,
        'gw_rounded_expectations':gw_rounded_expectations,
        'state_only_expectations':state_only_expectations,
        'bmz_rounded_expectations':bmz_rounded_expectations,
        'bmz_expectations':bmz_expectations,
        'standard_vals':standard_vals,
        'gw_rounded_vals':gw_rounded_vals,
        'state_only_vals':state_only_vals,
        'bmz_rounded_vals':bmz_rounded_vals,
        'bmz_vals':bmz_vals,
        'standard_results':standard_results,
        'gw_rounded_results':gw_rounded_results,
        'state_only_results':state_only_results,
        'bmz_rounded_results':bmz_rounded_results,
        'bmz_results':bmz_results,
        'shots':shots, 'n_layers':n_layers, 'n_qubits':n_qubits, 'n_trials':n_trials, 'optimizer':optimizer}
    info = {'n_qubits':n_qubits, 'n_layers':n_layers,'n_trials':n_trials, 'shots':shots, 'weights_set':weights_set, 'optimizer':optimizer}
    if save:
        save_result(res, 'experiment_2')
        save_info(info, 'experiment_2')
    return res


def plot_experiment_2(res, n_bins=12, save=True, date=None):
    standard_expectations = res['standard_expectations']
    gw_rounded_expectations = res['gw_rounded_expectations']
    state_only_expectations = res['state_only_expectations']
    bmz_rounded_expectations = res['bmz_rounded_expectations']
    bmz_expectations = res['bmz_expectations']
    standard_vals = res['standard_vals']
    gw_rounded_vals = res['gw_rounded_vals']
    state_only_vals = res['state_only_vals']
    bmz_rounded_vals = res['bmz_rounded_vals']
    bmz_vals = res['bmz_vals']
    n_trials = res['n_trials']
    labels = ['Standard QAOA', 'Rounded GW-WS-QAOA', 'BMZ-WS-QAOA State Only', 'Rounded BMZ-WS-QAOA', 'BMZ-WS-QAOA']
    colors = np.vstack((cp('deep')[:1], cp('deep')[2:3], cp('deep')[4:5], cp('pastel')[1:2], cp('deep')[3:4]))

    vals = 100*np.round(np.vstack((standard_vals, gw_rounded_vals, state_only_vals, bmz_rounded_vals, bmz_vals)).T, 6)
    expectations = 100*np.vstack((standard_expectations, gw_rounded_expectations, state_only_expectations, bmz_rounded_expectations, bmz_expectations)).T
    min_val = np.round(np.min(vals), 2)
    if min_val == 100:
        min_val = 95
    min_expectation = np.round(np.min(expectations), 2)

    other_vals = np.copy(vals)
    perfect_vals = np.full((n_trials, 5), np.nan)
    perfect_vals_index = np.where(vals == 100)
    other_vals[perfect_vals_index] = np.nan
    perfect_vals[perfect_vals_index] = 100

    bin_width = (100 - min_val)/n_bins
    bins = np.arange(min_val, 100 + bin_width, bin_width)
    width_ratio = bin_width/100/(1 - min_val/100)
    fig, ax = plt.subplots(1, 2, width_ratios=[1 - width_ratio, width_ratio], figsize=(5.5, 2.5))
    ax = ax.ravel()

    other_val_hist = ax[0].hist(other_vals, bins=bins, align='mid', color=colors, rwidth=1)
    ax[0].set_xticks(np.linspace(0, 100, 51))
    ax[0].tick_params(axis='both', which='both', labelsize=10)
    ax[0].set_xlim(min_val, 99.9999)
    ax[0].set_ylim(0, 1.2*np.max(other_val_hist[0]))
    ax[0].grid(False, axis='x')
    ax[0].set_xlabel('Cut size (% of maximum cut)', fontsize=12)
    ax[0].set_ylabel('Counts', fontsize=12)
    ax[0].set_title('(a)', fontsize=12, x=.03, y=.9, backgroundcolor='white')

    perfect_val_hist = ax[1].hist(perfect_vals, bins=[100 - bin_width/2, 100 + bin_width/2], align='mid', color=colors, rwidth=1)
    ax[1].set_xlim(100 - bin_width/2, 100 + bin_width/2)
    ax[1].set_ylim(0, np.max(perfect_val_hist[0]) + 5)
    ax[1].set_xticks([100], ['100'])
    ax[1].tick_params(axis='both', which='both', labelsize=10)
    ax[1].yaxis.tick_right()
    ax[1].grid(False, axis='x')
    ax[1].grid(True, axis='y', color='white')
    ax[1].set_ylabel('Max cut counts', fontsize=12)
    ax[1].yaxis.set_label_position('right')
    ax[1].set_facecolor('lightgrey')

    plt.tight_layout()
    fig.legend(labels, fontsize=10, loc='upper center', ncols=2, bbox_to_anchor=(.5, 1.3))
    if save:
        save_plot('experiment_2', 'best_measurement', date=date)
    plt.show()

    bin_width = (100 - min_expectation)/n_bins
    bins = np.arange(min_expectation - bin_width, 100 + 2*bin_width, bin_width)
    fig = plt.figure(figsize=(5, 2.5))
    expectation_hist = plt.hist(expectations, bins=bins, align='left', color=colors, rwidth=1)
    plt.xticks(np.linspace(0, 100, 26))
    plt.tick_params(axis='both', which='both', labelsize=10)
    plt.xlim(min_expectation - 2*bin_width, 100 + bin_width)
    plt.ylim(0, 1.2*np.max(expectation_hist[0]))
    plt.grid(False, axis='x')
    plt.xlabel('Cut size (% of maximum cut)', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title('(b)', fontsize=12, x=.025, y=.9, backgroundcolor='white')

    plt.tight_layout()
    if save:
        save_plot('experiment_2', suffix='circuit_expectation', date=date)
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
#     plt.suptitle(f'Experiment 2', fontsize=12, y=1.02)
#     plt.tight_layout()

#     if save:
#         save_plot('experiment_2')
#     plt.show()
