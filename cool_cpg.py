import os.path
import numpy as np
import matplotlib.pyplot as plt
import recurrent_network.network as rn
from scipy.signal import resample_poly


def _tau_q10(tau, dtemp):
    q10 = 2.0
    return tau / (q10 ** (dtemp / 10.0))


def _q(dtemp):
    q10 = 2.0
    return q10 ** (dtemp / 10.0)


def cool_single_cpg():
    """
    main function to cool single recurrent network
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


    # run dynamics at different temperatures using some Q10 for tau
    # and compute neural/behavioral trajectories
    dT_steps = [-0.2, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0]
    # dT_steps = [-1.0, -3.0, -5.0]
    # dT_steps = [-5.0, -10.0, -20.0]
    # dT_steps = [-0.5, -1.0, -2.0]
    fig2 = plt.figure(2)
    ax3 = fig2.add_subplot(len(dT_steps) + 1, 1, 1)
    for i in range(10):
        ax3.plot(ref_t, ref_rates[i, :], linewidth=0.5)
    cooled_trajectories = []
    cooled_behaviors = []
    for i, dT in enumerate(dT_steps):
        # cooled_tau = _tau_q10(nw.tau, dT)
        # cooled_nw = Network(N=800, g=1.5, pc=0.1, tau=cooled_tau)
        cooled_q = _q(dT)
        # cooled_nw = rn.Network(N=800, g=1.5, pc=0.1, q=cooled_q)
        cooled_nw = rn.Network(N=800, g=1.5, pc=1.0, q=cooled_q)
        # cooled_nw = rn.Network(N=1000, g=1.5, pc=1.0, q=cooled_q)
        cooled_nw.Wrec = Wrec
        # cooled_nw = Network(N=500, g=1.2, pc=0.5, q=cooled_q)
        # cooled_nw = rn.Network(N=50, g=0.5/np.sqrt(0.2), pc=1.0, q=cooled_q)
        cooled_t, cooled_rates = cooled_nw.simulate_network(T=t_max/cooled_q, dt=dt, external_input=ext_inp)

        # behavior = np.array([np.dot(Wout1, cooled_rates), np.dot(Wout2, cooled_rates)])
        behavior = np.dot(Wout1, cooled_rates) * np.dot(Wout2, cooled_rates)
        # behavior = np.array(np.dot(Wout, cooled_rates))
        cooled_behaviors.append(behavior)
        projected_rates = rn.project_neural_trajectory(cooled_rates, ref_mean, pcs)
        cooled_trajectories.append(projected_rates)

        label_str = 'dT = %.1f' % dT
        ax1.plot(projected_rates[0, :], projected_rates[1, :], linewidth=0.5, label=label_str)
        # ax2.plot(behavior[0], behavior[1], linewidth=0.5, label=label_str)
        ax2.plot(cooled_t, behavior, linewidth=0.5, label=label_str)
        ax3 = fig2.add_subplot(len(dT_steps) + 1, 1, 2 + i)
        for j in range(10):
            ax3.plot(cooled_t, cooled_rates[j, :], linewidth=0.5)

    ax1.legend()
    ax1.set_xlabel('PC 1 (a.u.)')
    ax1.set_ylabel('PC 2 (a.u.)')
    ax2.legend()
    ax2.set_xlabel('Output (a.u.)')

    # measure similarity of neural/behavioral trajectories as a function of temperature
    trajectory_similarities = []
    behavior_similarities = []
    for i in range(len(dT_steps)):
        similarity1 = rn.measure_trajectory_similarity(ref_trajectory, cooled_trajectories[i])
        similarity2 = rn.measure_trajectory_similarity(ref_behavior, cooled_behaviors[i])
        trajectory_similarities.append(similarity1)
        behavior_similarities.append(similarity2)
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.plot(dT_steps, trajectory_similarities, 'ko-', label='neural activity')
    ax4.plot(dT_steps, behavior_similarities, 'ro-', label='behavior')
    ax4.set_xlim(ax4.get_xlim()[::-1])
    ax4.set_xlabel('Temperature change')
    ax4.set_ylabel('Corr. coeff.')
    ax4.legend()

    plt.show()


def cool_distributed_cpgs():
    """
    main function to manipulate neurons/synapses in distributed recurrent networks
    loads previously fitted output weights
    computes correlation between neural trajectories as a similarity measure
    :return: nothing
    """
    # load output weights
    out_dir = '/Users/robert/project_src/cooling/single_cpg_manipulation'
    weight_suffix1 = 'outunit1_parallel_weights_force.npy'
    weight_suffix2 = 'outunit2_parallel_weights_force.npy'
    weight_suffix3 = 'Wrec1_parallel_weights_force.npy'
    weight_suffix4 = 'Wrec2_parallel_weights_force.npy'
    Wout1 = np.load(os.path.join(out_dir, weight_suffix1))
    Wout2 = np.load(os.path.join(out_dir, weight_suffix2))
    Wrec1 = np.load(os.path.join(out_dir, weight_suffix3))
    Wrec2 = np.load(os.path.join(out_dir, weight_suffix4))

    # create network with same parameters
    t_max = 2000.0
    dt = 1.0
    nw1 = rn.Network(N=800, g=1.5, pc=1.0)
    nw1.Wrec = Wrec1
    nw2 = rn.Network(N=800, g=1.5, pc=1.0, seed=5432)
    nw2.Wrec = Wrec2
    def ext_inp(t):
        return np.zeros(nw1.N)

    # run dynamics at reference temperature and compute neural/behavioral trajectory
    ref_t1, ref_rates1 = nw1.simulate_network(T=t_max, dt=dt, external_input=ext_inp)
    ref_t2, ref_rates2 = nw2.simulate_network(T=t_max, dt=dt, external_input=ext_inp)
    neuron_out1 = np.dot(Wout1, ref_rates1)
    neuron_out2 = np.dot(Wout2, ref_rates2)
    ref_behavior = np.array([neuron_out1, neuron_out2])

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(neuron_out1, neuron_out2, 'k', linewidth=0.5, label='ref')

    # run dynamics at different temperatures using some Q10 for tau
    # and compute neural/behavioral trajectories
    dT_steps = [-0.2, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0]
    # dT_steps = [-1.0, -3.0, -5.0]
    # dT_steps = [-0.5, -1.0, -2.0]
    cooled_behaviors = []
    for i, dT in enumerate(dT_steps):
        cooled_q = _q(dT)
        cooled_nw = rn.Network(N=800, g=1.5, pc=1.0, q=cooled_q)
        cooled_nw.Wrec = Wrec1
        cooled_t, cooled_rates = cooled_nw.simulate_network(T=t_max/cooled_q, dt=dt, external_input=ext_inp)

        cooled_behavior = np.dot(Wout1, cooled_rates)
        target_length = len(ref_behavior[1])
        original_length = len(cooled_behavior)
        cooled_behavior_resampled = resample_poly(cooled_behavior, target_length, original_length)
        cooled_behaviors.append(np.array([cooled_behavior_resampled, ref_behavior[1]]))

        label_str = 'dT = %.1f' % dT
        ax1.plot(cooled_behavior_resampled, ref_behavior[1], linewidth=0.5, label=label_str)

    ax1.legend()
    ax1.set_xlabel('Output (a.u.)')

    # measure similarity of neural/behavioral trajectories as a function of temperature
    behavior_similarities = []
    for i in range(len(dT_steps)):
        similarity = rn.measure_trajectory_similarity(ref_behavior, cooled_behaviors[i])
        behavior_similarities.append(similarity)
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(dT_steps, behavior_similarities, 'ro-', label='behavior')
    ax2.set_xlim(ax2.get_xlim()[::-1])
    ax2.set_xlabel('Temperature change')
    ax2.set_ylabel('Corr. coeff.')
    ax2.legend()

    plt.show()


if __name__ == '__main__':
    cool_single_cpg()
    # cool_distributed_cpgs()