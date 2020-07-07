import os.path
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import recurrent_network.network as rn

rng = np.random.RandomState(1234)


def activation(t):
    # for trajectory comparison
    # if 0.0 <= t <= 50.0:
    # for visualization
    if 600.0 <= t <= 650.0:
        return np.ones(800)
    else:
        return np.zeros(800)


def _activate_neurons(network, fraction):
    """
    activate a random fraction of neurons in the network;
    could be used to mimick electrical or optogenetic stimulation.
    for simplicity, set their external input weights to 5
    and external input to them to 5 for a brief period (5 ms)
    :param network: Network instance
    :param fraction: number between 0 and 1; fraction of neurons to be activated
    :return: external input function that allows this activation
    """
    nr_activate = int(network.N * fraction)
    activate_ids = rng.choice(range(network.N), nr_activate, replace=False)
    network.Win[activate_ids] = 0.008
    network.activate_ids = activate_ids
    # network.Win[activate_ids] = 0.016 * rng.rand()
    # print('Activate IDs:')
    # print(str(activate_ids))

    # def activation(t):
    #     drive = np.zeros(network.N)
    #     if 0.0 <= t <= 20.0:
    #         drive[network.activate_ids] = 1.0
    #     return drive
    #
    # network.ext_inp = activation
    # return activation


def _remove_neurons(network, fraction):
    """
    remove neurons from network; equivalent to setting their input and output synapses to 0,
    so that's what we will do
    :param network: Network instance
    :param fraction: number between 0 and 1; fraction of neurons to be removed
    :return: nothing
    """
    nr_remove = int(network.N * fraction)
    remove_ids = rng.choice(range(network.N), nr_remove, replace=False)
    for id in remove_ids:
        network.Wrec[id, :] = 0.0
        network.Wrec[:, id] = 0.0
        network.Win[id] = 0.0


def _remove_synapses(network, fraction):
    """
    remove (recurrent) synapses from network by setting them to zero
    :param network: Network instance
    :param fraction: number between 0 and 1; fraction of synapses to be removed
    :return: nothing
    """
    existing_synapses = np.nonzero(network.Wrec)
    # hack: proportion is the same for zero and non-zero synapses
    # so we just create a matrix of the full size and set a fraction of all weights to zero
    tmp = rng.rand(network.Wrec.shape[0], network.Wrec.shape[1])
    remove_synapses = tmp < fraction
    network.Wrec[remove_synapses] = 0.0


def manipulate_network(manipulation, mode):
    """
    main function to manipulate neurons/synapses in single recurrent network
    loads previously fitted output weights
    computes correlation between neural trajectories as a similarity measure
    :return: nothing
    """
    # load output weights
    out_dir = '/Users/robert/project_src/cooling/single_cpg_manipulation/weights'
    # weight_suffix1 = 'outunit1_weights.npy'
    # weight_suffix2 = 'outunit2_weights.npy'
    weight_suffix1 = 'outunit1_weights_force.npy'
    weight_suffix2 = 'outunit2_weights_force.npy'
    weight_suffix3 = 'Wrec_weights_force.npy'
    # weight_suffix1 = 'outunit_weights_twotimescales_force.npy'
    # weight_suffix2 = 'Wrec_weights_twotimescales_force.npy'
    # weight_suffix1 = 'outunit1_weights_circle.npy'
    # weight_suffix2 = 'outunit2_weights_circle.npy'
    Wout1 = np.load(os.path.join(out_dir, weight_suffix1))
    Wout2 = np.load(os.path.join(out_dir, weight_suffix2))
    Wrec = np.load(os.path.join(out_dir, weight_suffix3))

    # create network with same parameters
    t_max = 2000.0
    dt = 1.0
    # nw = rn.Network(N=800, g=1.5, pc=0.1)
    nw = rn.Network(N=800, g=1.5, pc=1.0)
    # nw = rn.Network(N=1000, g=1.5, pc=1.0)
    nw.Wrec = Wrec
    # nw = Network(N=500, g=1.2, pc=0.5)
    # nw = rn.Network(N=50, g=0.5/np.sqrt(0.2), pc=1.0)
    # Laje and Buonomano: 50 ms step
    # def ext_inp(t):
    #     if 0.0 <= t <= 50.0:
    #         return 5.0 * np.ones(nw.N)
    #     else:
    #         return np.zeros(nw.N)
    def ext_inp(t):
        return np.zeros(nw.N)

    # run dynamics at reference temperature and compute neural/behavioral trajectory
    ref_t, ref_rates = nw.simulate_network(T=t_max, dt=dt, external_input=ext_inp)
    ref_mean_ = np.mean(ref_rates, axis=1)
    ref_mean = ref_mean_.transpose()
    # ref trajectory: first three PCs (sufficient?)
    pcs, ref_trajectory = rn.compute_neural_trajectory(ref_rates)
    neuron_out1 = np.dot(Wout1, ref_rates)
    neuron_out2 = np.dot(Wout2, ref_rates)
    ref_behavior = neuron_out1 * neuron_out2
    # ref_behavior = np.array([neuron_out1, neuron_out2])
    # ref_behavior = np.array(neuron_out)

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.plot(ref_trajectory[0, :], ref_trajectory[1, :], 'k', linewidth=0.5, label='ref')
    ax2 = fig1.add_subplot(2, 1, 2)
    # ax2.plot(neuron_out1, neuron_out2, 'k', linewidth=0.5, label='ref')
    ax2.plot(ref_t, ref_behavior, 'k', linewidth=0.5, label='ref')

    # run dynamics at different fractions of manipulated neurons
    # and compute neural/behavioral trajectories
    if manipulation == 'activation':
        if mode == 'sweep':
            fractions = [5.e-3, 1.e-2, 2.e-2, 5.e-2, 8.e-2, 1.e-1, 1.5e-1, 2.e-1]
        elif mode == 'vis':
            fractions = [1.e-2, 5.e-2, 1.e-1]
    elif manipulation == 'inactivation':
        if mode == 'sweep':
            fractions = [1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2, 5.e-2, 8.e-2, 1.e-1, 1.5e-1, 2.e-1]
        elif mode == 'vis':
            fractions = [2.e-3, 1.e-2, 5.e-2]

    fig2 = plt.figure(2)
    ax3 = fig2.add_subplot(len(fractions) + 1, 1, 1)
    for i in range(10):
        ax3.plot(ref_t, ref_rates[i, :], linewidth=0.5)

    n_repetitions = 10 # simulate multiple synapse removals
    manipulated_trajectories = []
    manipulated_behaviors = []
    for i, fraction in enumerate(fractions):
        # manipulated_nw = rn.Network(N=800, g=1.5, pc=0.1)
        # manipulated_nw = Network(N=500, g=1.2, pc=0.5)
        fraction_behaviors = []
        fraction_trajectories = []
        inputs = []
        zero_active = []
        networks = []
        # drives = []
        fraction_rates = np.zeros((n_repetitions, nw.N, int(t_max / dt + 0.5)))
        for j in range(n_repetitions):
            # manipulated_nw = rn.Network(N=800, g=1.5, pc=0.1)
            manipulated_nw = rn.Network(N=800, g=1.5, pc=1.0)
            # manipulated_nw = rn.Network(N=1000, g=1.5, pc=1.0)
            manipulated_nw.Wrec = np.copy(Wrec)
            # manipulated_nw = rn.Network(N=50, g=0.5/np.sqrt(0.2), pc=1.0)

            if manipulation == 'activation':
                nr_activate = int(manipulated_nw.N * fraction)
                activate_ids_ = rng.choice(range(manipulated_nw.N), nr_activate, replace=False)
                if 0 in activate_ids_:
                    zero_active.append(1)
                else:
                    zero_active.append(0)
                activate_ids = np.array([1 if i in activate_ids_ else 0 for i in range(manipulated_nw.N)], dtype=bool)
                # activate_ids = np.array([i % (j + 2) for i in range(manipulated_nw.N)], dtype=bool)
                manipulated_nw.Win = np.zeros(manipulated_nw.N)
                manipulated_nw.Win[activate_ids] = 0.016 * rng.rand()
                manipulated_nw.Win[~activate_ids] = 0.0
                manipulated_nw.activate_ids = activate_ids

                inputs.append(manipulated_nw.activate_ids)
                # manipulated_t, manipulated_rates = manipulated_nw.simulate_network(T=t_max, dt=dt,
                #                                                                    external_input=activation_func)
                # func = lambda t: activation(t, manipulated_nw)
                manipulated_t, manipulated_rates = manipulated_nw.simulate_network(T=t_max, dt=dt,
                                                                                   external_input=activation)
                fraction_rates[j, :, :] = manipulated_rates[:, :]
                behavior = np.dot(Wout1, manipulated_rates) * np.dot(Wout2, manipulated_rates)

            # drives.append(activation)

            elif manipulation == 'inactivation':
                # for visualization, run 600 ms of regular sim, then inactivate from that state
                baseline = 600.0
                default_nw = rn.Network(N=800, g=1.5, pc=1.0)
                default_nw.Wrec = np.copy(Wrec)
                default_t, default_rates = default_nw.simulate_network(T=baseline, dt=dt)
                _remove_neurons(manipulated_nw, fraction)
                new_x0 = np.arctanh(default_rates[:, -1])
                manipulated_t_, manipulated_rates = manipulated_nw.simulate_network(T=t_max-baseline, dt=dt, x0=new_x0)
                manipulated_t = np.arange(0.0, t_max, dt)
                # stitch together our Frankenstein network output
                fraction_rates[j, :, :len(default_t)] = default_rates[:, :]
                fraction_rates[j, :, len(default_t):] = manipulated_rates[:, :]
                behavior = np.dot(Wout1, fraction_rates[j, :, :]) * np.dot(Wout2, fraction_rates[j, :, :])

            networks.append(manipulated_nw)
            fraction_behaviors.append(behavior)
            projected_rates = rn.project_neural_trajectory(fraction_rates[j, :, :], ref_mean, pcs)
            fraction_trajectories.append(projected_rates)
        manipulated_behaviors.append(fraction_behaviors)
        manipulated_trajectories.append(fraction_trajectories)

        # hack: simply plot last simulation
        plot_index = 7 # 1 (then 7) best for w1 = 200 / w2 = 2 w1
        label_str = 'fraction = %.3f' % fraction
        ax1.plot(fraction_trajectories[plot_index][0, :], fraction_trajectories[plot_index][1, :], linewidth=0.5, label=label_str)
        # ax2.plot(fraction_behaviors[-1][0], fraction_behaviors[-1][1], linewidth=0.5, label=label_str)
        ax2.plot(manipulated_t, fraction_behaviors[plot_index], linewidth=0.5, label=label_str)
        ax3 = fig2.add_subplot(len(fractions) + 1, 1, 2 + i)
        for j in range(10):
            ax3.plot(manipulated_t, fraction_rates[plot_index, j, :], linewidth=0.5)

    ax1.legend()
    ax2.legend()

    # measure similarity of neural/behavioral trajectories as a function of temperature
    trajectory_similarities = []
    trajectory_errors = []
    behavior_similarities = []
    behavior_errors = []
    for i in range(len(fractions)):
        tmp_similarities1 = []
        tmp_similarities2 = []
        for j in range(n_repetitions):
            similarity1 = rn.measure_trajectory_similarity(ref_trajectory, manipulated_trajectories[i][j])
            similarity2 = rn.measure_trajectory_similarity(ref_behavior, manipulated_behaviors[i][j])
            tmp_similarities1.append(similarity1)
            tmp_similarities2.append(similarity2)
        trajectory_similarities.append(np.mean(tmp_similarities1))
        trajectory_errors.append(np.std(tmp_similarities1)/np.sqrt(n_repetitions))
        behavior_similarities.append(np.mean(tmp_similarities2))
        behavior_errors.append(np.std(tmp_similarities2)/np.sqrt(n_repetitions))
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.errorbar(fractions, trajectory_similarities, yerr=trajectory_errors, fmt='ko-', label='neural activity')
    ax4.errorbar(fractions, behavior_similarities, yerr=behavior_errors, fmt='ro-', label='behavior')
    ax4.set_xlabel('Fraction of neurons inactivated')
    ax4.set_ylabel('Corr. coeff.')
    ax4.legend()

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--manipulation', required=True, choices=('activation', 'inactivation'),
                        help='activate or inactivate neurons')
    parser.add_argument('--mode', required=True, choices=('vis', 'sweep'),
                        help='vis: small parameter set for figures; sweep: large parameter set for graph')

    args = parser.parse_args()
    manipulate_network(args.manipulation, args.mode)