import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import recurrent_network.network as rn


def run_output_fit_force():
    """
    learn weights onto 2 output units to generate target output trajectory
    using FORCE algorithm
    :return: nothing
    """
    t_max = 4000.0
    dt = 0.1
    nw = rn.Network(N=800, g=1.5, pc=1.0)

    # def ext_inp(t):
    #     return np.zeros(nw.N)

    def behavior(t):
        w = 2 * np.pi / 200.0 # chaotic dynamics
        phi = np.pi / 100.0
        target_neuron1 = 0.5*np.sin(w * t + phi)
        target_neuron2 = 0.5*np.sin(2 * w * t + phi)
        # target_neuron2 = 0.5*np.cos(w * t + phi)
        return target_neuron1, target_neuron2

    force_result = nw.simulate_learn_network(behavior, T=t_max)
    Wout1, Wout2, Wrec_new = force_result

    nw_test = rn.Network(N=800, g=1.5, pc=1.0)
    nw_test.Wrec = Wrec_new
    t, rates = nw_test.simulate_network(T=t_max, dt=dt)

    neuron_out1 = np.dot(Wout1, rates)
    neuron_out2 = np.dot(Wout2, rates)
    target_neuron1, target_neuron2 = behavior(t)
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(neuron_out1, neuron_out2, 'r-', linewidth=1, label='learned')
    ax2.plot(target_neuron1, target_neuron2, 'k--', linewidth=0.5, label='target')
    ax2.legend()
    ax2.set_xlabel('Output 1 activity')
    ax2.set_ylabel('Output 2 activity')
    ax2.set_title('Output')

    plt.show()

    out_dir = '/Users/robert/project_src/cooling/single_cpg_manipulation'
    out_suffix1 = 'outunit1_weights_force.npy'
    out_suffix2 = 'outunit2_weights_force.npy'
    out_suffix3 = 'Wrec_weights_force.npy'
    np.save(os.path.join(out_dir, out_suffix1), Wout1)
    np.save(os.path.join(out_dir, out_suffix2), Wout2)
    np.save(os.path.join(out_dir, out_suffix3), Wrec_new)


def run_output_fit():
    """
    fit weights onto 2 output units to generate target output trajectory
    :return: nothing
    """

    t_max = 2000.0
    dt = 1.0
    # nw = rn.Network(N=800, g=1.5, pc=0.1)
    # nw = Network(N=500, g=1.2, pc=0.5)
    nw = rn.Network(N=50, g=0.5/np.sqrt(0.2), pc=1.0)
    # def ext_inp(t):
    #     return 1.0 * np.sin(2 * np.pi / 220.0 * t + 0.1) * np.ones(nw.N)
    # Laje and Buonomano: 50 ms step
    # def ext_inp(t):
    #     if 0.0 <= t <= 50.0:
    #         return 5.0 * np.ones(nw.N)
    #     else:
    #         return np.zeros(nw.N)
    def ext_inp(t):
        return np.zeros(nw.N)
    t, rates = nw.simulate_network(T=t_max, dt=dt, external_input=ext_inp)

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(2, 1, 1)
    # for i in range(nw.N):
    for i in range(10):
        ax1.plot(t, rates[i, :], linewidth=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_title('Network activity')

    w = 2 * np.pi / 215.0 # CPG dynamics
    # w = 2 * np.pi / 2000.0 # chaotic dynamics
    phi = np.pi / 100.0
    target_neuron1 = 0.5*np.sin(w * t + phi)
    # target_neuron2 = 0.5*np.sin(2 * w * t + phi)
    target_neuron2 = 0.5*np.cos(w * t + phi)

    # target(t) =  y(t) * Wout
    # target: 1 x M; y: N x M; Wout: 1 x N; N << M
    tmp_solution = lstsq(rates.transpose(), target_neuron1, cond=1.0e-4)
    Wout1 = tmp_solution[0]
    tmp_solution = lstsq(rates.transpose(), target_neuron2, cond=1.0e-4)
    Wout2 = tmp_solution[0]

    neuron_out1 = np.dot(Wout1, rates)
    neuron_out2 = np.dot(Wout2, rates)
    # fig2 = plt.figure(2)
    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.plot(neuron_out1, neuron_out2, 'r-', linewidth=1, label='fit')
    ax2.plot(target_neuron1, target_neuron2, 'k--', linewidth=0.5, label='target')
    ax2.legend()
    ax2.set_xlabel('Output 1 activity')
    ax2.set_ylabel('Output 2 activity')
    ax2.set_title('Output')

    plt.show()

    out_dir = '/Users/robert/project_src/cooling/single_cpg_manipulation'
    out_suffix1 = 'outunit1_weights_circle.npy'
    out_suffix2 = 'outunit2_weights_circle.npy'
    np.save(os.path.join(out_dir, out_suffix1), Wout1)
    np.save(os.path.join(out_dir, out_suffix2), Wout2)


if __name__ == '__main__':
    # run_output_fit()
    run_output_fit_force()