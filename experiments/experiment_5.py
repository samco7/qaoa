import sys
sys.path.insert(0, './../')
from time import time
from quantum_solvers import *
from helper_functions import *
from save_results import *
from tqdm import tqdm
from qiskit.providers.aer import AerSimulator


def experiment_5(n_qubits, n_trials, layer_vals, backend=None, shots=512, save=True, max_circ_evals=None):
    weights_set = list(range(1, 11))
    expectations = np.zeros((n_trials, len(layer_vals)*5))
    vals = np.copy(expectations)
    opt_terminated_early = 0
    progress = tqdm(total=len(layer_vals)*n_trials*5)
    for j in range(n_trials):
        graph = random_graph(n_qubits, weights_set)
        exact_val = akmaxsat(graph)[1]
        for i in range(len(layer_vals)):
            standard_qaoa = QAOASolver(layer_vals[i], warm_start_method=None, epsilon=None, shots=shots, backend=backend, max_circ_evals=max_circ_evals)
            gw_rounded_qaoa = QAOASolver(layer_vals[i], warm_start_method='GW Rounded', epsilon=.25, shots=shots, backend=backend, max_circ_evals=max_circ_evals)
            bmz_rounded_qaoa = QAOASolver(layer_vals[i], warm_start_method='BMZ Rounded', epsilon=.25, shots=shots, backend=backend, max_circ_evals=max_circ_evals)
            state_only_qaoa = QAOASolver(layer_vals[i], warm_start_method='BMZ', epsilon=.1, backend=backend, shots=shots, adjust_mixer=False, max_circ_evals=max_circ_evals)
            bmz_qaoa = QAOASolver(layer_vals[i], warm_start_method='BMZ', epsilon=.1, shots=shots, backend=backend, max_circ_evals=max_circ_evals)

            standard_res = standard_qaoa.solve(graph)
            progress.update(1)
            gw_rounded_res = gw_rounded_qaoa.solve(graph)
            progress.update(1)

            bmz_relaxed = BMZ(graph)

            bmz_rounded_res = bmz_rounded_qaoa.solve(graph, relaxed_solution=bmz_relaxed)
            progress.update(1)
            state_only_res = state_only_qaoa.solve(graph, relaxed_solution=bmz_relaxed)
            progress.update(1)
            bmz_res = bmz_qaoa.solve(graph, relaxed_solution=bmz_relaxed)
            progress.update(1)

            for res in [standard_res, gw_rounded_res, bmz_rounded_res, state_only_res, bmz_res]:
                if not res.opt_terminated: opt_terminated_early += 1

            vals[j, len(layer_vals) + i] = gw_rounded_res.obj/exact_val
            vals[j, 2*len(layer_vals) + i] = bmz_rounded_res.obj/exact_val
            vals[j, 3*len(layer_vals) + i] = state_only_res.obj/exact_val
            vals[j, 4*len(layer_vals) + i] = bmz_res.obj/exact_val

            if layer_vals[i] == 0:
                vals[j, i] = None
                expectations[j, i] = None
                expectations[j, len(layer_vals) + i] = None
                expectations[j, 2*len(layer_vals) + i] = None
                expectations[j, 3*len(layer_vals) + i] = None
                expectations[j, 4*len(layer_vals) + i] = None
            else:
                vals[j, i] = standard_res.obj/exact_val
                expectations[j, i] = standard_res.expectation/exact_val
                expectations[j, len(layer_vals) + i] = gw_rounded_res.expectation/exact_val
                expectations[j, 2*len(layer_vals) + i] = bmz_rounded_res.expectation/exact_val
                expectations[j, 3*len(layer_vals) + i] = state_only_res.expectation/exact_val
                expectations[j, 4*len(layer_vals) + i] = bmz_res.expectation/exact_val
    progress.close()
    if opt_terminated_early > 0:
        print(opt_terminated_early, 'QAOA optimization routines terminated early.')

    res = {'expectations':expectations, 'vals':vals, 'layer_vals':layer_vals, 'shots':shots, 'n_qubits':n_qubits, 'n_trials':n_trials}
    info = {'n_qubits':n_qubits, 'layer_vals':layer_vals,'n_trials':n_trials, 'shots':shots, 'weights_set':weights_set}
    if save:
        save_result(res, 'experiment_5')
        save_info(info, 'experiment_5')
    return res


def plot_experiment_5(res, save=True):
    layer_vals = res['layer_vals']
    N = len(layer_vals)
    expectations = res['expectations']
    vals = res['vals']
    min_expect = np.nanmin(expectations)
    min_val = np.nanmin(vals)
    algorithms = ['Standard QAOA', 'GW-Rounded-WS-QAOA', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA State Only', 'BMZ-WS-QAOA']

    for i in range(len(algorithms)):
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))

        mean_expectations = np.mean(expectations[:, i*N:(i + 1)*N], axis=0)
        best_expectations = np.max(expectations[:, i*N:(i + 1)*N], axis=0)
        worst_expectations = np.min(expectations[:, i*N:(i + 1)*N], axis=0)

        mean_vals = np.mean(vals[:, i*N:(i + 1)*N], axis=0)
        best_vals = np.max(vals[:, i*N:(i + 1)*N], axis=0)
        worst_vals = np.min(vals[:, i*N:(i + 1)*N], axis=0)

        ax[0].plot(layer_vals, mean_expectations)
        ax[0].fill_between(layer_vals, worst_expectations, best_expectations, alpha=0.2)
        ax[0].set_title('Expected Cut Value')
        ax[0].set_xlabel('n_layers')
        ax[0].set_ylabel('Approximation ratio')
        ax[0].set_ylim(min_expect, 1)
        ax[1].plot(layer_vals, mean_vals)
        ax[1].fill_between(layer_vals, worst_vals, best_vals, alpha=0.2)
        ax[1].set_title('Best Cut Value')
        ax[1].set_xlabel('n_layers')
        ax[1].set_ylabel('Approximation ratio')
        ax[1].set_ylim(min_val, 1)

        sorted_layers = np.sort(layer_vals)
        ax[0].set_xlim(sorted_layers[1] - .1, sorted_layers[-1] + .1)
        ax[0].set_xticks(sorted_layers[1:])
        if algorithms[i] == 'Standard QAOA':
            ax[1].set_xlim(sorted_layers[1] - .1, sorted_layers[-1] + .1)
            ax[1].set_xticks(sorted_layers[1:])
        else:
            ax[1].set_xlim(sorted_layers[0] - .1, sorted_layers[-1] + .1)
            ax[1].set_xticks(sorted_layers)
        plt.suptitle(algorithms[i], y=1)
        plt.tight_layout()
        if save:
            save_plot('experiment_5', suffix=algorithms[i].replace(' ', '-').lower())
        plt.show()


# def plot_experiment_5(res, save=True):
#     layer_vals = res['layer_vals']
#     N = len(layer_vals)
#     expectations = res['expectations']
#     vals = res['vals']

#     fig, ax = plt.subplots(2, 1, figsize=(14, 5))
#     grouping = np.arange(1, N + 1, 1)

#     positions = np.concatenate((grouping, grouping + N + 1, grouping + 2*(N + 1), grouping + 3* (N + 1)))
#     algorithms = ['Standard QAOA', 'GW-Rounded-WS-QAOA', 'BMZ-Rounded-WS-QAOA', 'BMZ-WS-QAOA State Only', 'BMZ-WS-QAOA']
#     labels = []
#     #inelegantbutworks
#     counter = 0
#     for i in range(len(positions)):
#         idx = i%len(layer_vals)
#         if idx == len(layer_vals)//2:
#             labels.append(str(layer_vals[idx]) + ' layers\n\n' + str(algorithms[counter]))
#             counter += 1
#         else:
#             labels.append(str(layer_vals[idx]) + ' layers')

#     ax[0].boxplot(expectations, positions=positions, sym='.',
#     flierprops=dict(markeredgecolor='k'),
#     medianprops=dict(color='steelblue'),
#     boxprops=dict(color='k'),
#     capprops=dict(color='k'),
#     whiskerprops=dict(color='k'))
#     ax[0].set_xticks(positions, labels)
#     ax[0].set_ylabel('Circuit expectation', fontsize=12)
#     plt.grid(False, axis='x')

#     ax[1].boxplot(vals, positions=positions, sym='.',
#     flierprops=dict(markeredgecolor='k'),
#     medianprops=dict(color='steelblue'),
#     boxprops=dict(color='k'),
#     capprops=dict(color='k'),
#     whiskerprops=dict(color='k'))
#     ax[1].set_xticks(positions, labels)
#     ax[1].set_ylabel('Best sampled cut', fontsize=12)
#     plt.grid(False, axis='x')

#     plt.suptitle(f'Experiment 5', fontsize=14, y=1.02)
#     plt.tight_layout()
#     if save:
#         save_plot('experiment_5')
#     plt.show()
