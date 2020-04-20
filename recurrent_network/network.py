import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import resample_poly
import matplotlib.pyplot as plt


def compute_neural_trajectory(rates):
    """
    compute PCA of rates
    :param rates: neuron firing rates (shape N neurons (dims.) x M time points (obs.))
    :return: pcs, projection: first three PCs and projection of rates onto these (pcs: column vectors!)
    """
    # transpose such that we have M obs. x N dims
    rates_ = np.copy(rates)
    rates_ = np.transpose(rates_)
    # center firing rates
    rates_ -= np.mean(rates_, axis=0)
    rates_ /= np.std(rates_, axis=0)
    # eigenvectors - V^T; eigenvalues: sv**2
    U, sv, Vt = np.linalg.svd(rates_)
    pcs = np.transpose(Vt)
    projected_rates = np.dot(rates_, pcs)
    return pcs[:, :3], projected_rates[:, :3].transpose()


def project_neural_trajectory(rates, ref_mean, pcs):
    """
    simple projection of firing rates onto previously computed PCs
    :param rates: neuron firing rates (shape N neurons (dims.) x M time points (obs.))
    :param pcs: principal components used for projection
    :return: projection: projected rates
    """
    # transpose such that we have M obs. x N dims
    rates_ = np.copy(rates)
    rates_ = np.transpose(rates_)
    # center firing rates
    # rates -= np.mean(rates, axis=0)
    rates_ -= ref_mean
    rates_ /= np.std(rates_, axis=0)
    projected_rates = np.dot(rates_, pcs)
    return projected_rates.transpose()


def measure_trajectory_similarity(ref_trajectory, measure_trajectory):
    """
    compute correlation coefficient between two (multi-dimensional) trajectories
    :param ref_trajectory: reference trajectory
    :param measure_trajectory: trajectory to be compared
    :return:
    """
    if len(ref_trajectory.shape) != len(measure_trajectory.shape):
        e = 'Trajectories do not have the same dimensions'
        raise ValueError(e)

    # resample measure_trajectory to len(ref_trajectory)
    # find time axis (assume more samples than dimensions)
    time_axis = np.argmax(ref_trajectory.shape)
    target_length = ref_trajectory.shape[time_axis]
    original_length = measure_trajectory.shape[time_axis]
    if target_length != original_length:
        print("Re-sampling measurement trajectory...")
        measure_resampled = resample_poly(measure_trajectory, target_length, original_length, axis=time_axis)
        measure_ = measure_resampled.flatten()
    else:
        measure_ = measure_trajectory.flatten()
    ref_ = ref_trajectory.flatten()
    return np.corrcoef(ref_, measure_)[0, 1]


class Network(object):

    def __init__(self, N=50, tau=10, g=0.5, pc=0.2, i_noise=1.0e-3, q=1.0, seed=4223):
        self.rng = np.random.RandomState(seed)

        # network parameters
        # default parameters give oscillatory dynamics in absence of external input
        # for non-zero initial conditions
        self.N = N # number of neurons
        self.tau = tau # in ms
        self.g = g # synaptic strength param
        self.pc = pc # connection probability
        self.i_noise = i_noise
        self.q = q # temperature scaling parameter

        # full recurrent connectivity
        # self.Wrec = self.rng.randn(N, N) * g / np.sqrt(pc * N)
        # sparse recurrent connectivity
        Wtmp = self.rng.randn(N, N) * g / np.sqrt(pc * N)
        keep_synapses = self.rng.rand(N, N) < self.pc
        self.Wrec = Wtmp * keep_synapses
        # self.Win = self.rng.randn(N) * g / np.sqrt(pc * N)
        self.Win = self.rng.randn(N)
        self.ext_inp = None

    def _network_dynamics(self, t, x, ext_inp):
        """
        :param t: time point
        :param x: array of membrane potentials
        :param ext_inp: callable, accepts simulation time t as input and returns array of external input to each neuron
        :return: dxdt: array of derivatives of membrane potential for each unit
        """

        # dxdt = (1. / (self.tau) * (-x * self.q + np.dot(self.Wrec, np.tanh(x * self.q)) + np.dot(self.Win, ext_inp(t))
        #                           + self.i_noise * self.rng.randn(self.N)))
        dxdt = 1. / (self.tau) * (-x * self.q + np.dot(self.Wrec, np.tanh(x * self.q)) + np.dot(self.Win, ext_inp(t)))
        # dxdt = (1. / (self.tau / self.q) * (-x + np.dot(self.Wrec, np.tanh(x)) + np.dot(self.Win, ext_inp(t))
        #                           + self.i_noise * self.rng.randn(self.N)))
        # dxdt = 1. / (self.tau / self.q) * (-x + np.dot(self.Wrec, np.tanh(x)) + np.dot(self.Win, ext_inp(t)))
        return dxdt

    def simulate_network(self, external_input=None, x0=None, T=2000.0, dt=None):
        """
        main function to run recurrent network simulations
        :param external_input: callable, accepts simulation time t as input
        and returns array of external input to each neuron
        :param x0: array of initial values for each neuron (length self.N)
        :return: solution
        """

        if external_input is None and self.ext_inp is None:
            def external_input(t):
                return np.zeros(self.N)
        elif external_input is None and self.ext_inp is not None:
            external_input = self.ext_inp
        if x0 is None:
            x0 = np.tanh(self.rng.randn(self.N))

        if dt is None:
            t_eval = None
        else:
            t_eval = np.arange(0.0, T, dt)

        status_str = 'Simulating network activity for %.0f ms' % T
        print(status_str)
        # set up integration using anonymous function because scipy integrators
        # expect function to only have two arguments t, x
        solution = solve_ivp(lambda t, x: self._network_dynamics(t, x, external_input),
                             (0.0, T), x0, t_eval=t_eval)
        time_steps = solution.t
        rates = np.tanh(solution.y)
        print('Done')

        return time_steps, rates

    def simulate_learn_network(self, out_func, external_input=None, x0=None, T=2000.0):
        """
        main function to run recurrent network simulations and use the FORCE algorithm
        to update recurrent and output weights.
        Simple implementation for fully recurrent networks from Sussillo & Abbott 2009
        :param out_func: callable, output function to be learned
        :param external_input: callable, accepts simulation time t as input
        and returns array of external input to each neuron
        :param x0: array of initial values for each neuron (length self.N)
        :param T: learning duration
        :return: solution
        """
        if external_input is None:
            def external_input(t):
                return np.zeros(self.N)
        if x0 is None:
            x0 = np.tanh(self.rng.randn(self.N))

        dot = np.dot
        alpha = 1.0e-0
        nsecs = T # default: 1440
        dt = 0.1
        learn_every = 2
        time_steps = np.arange(0.0, nsecs, dt)

        wo_length = []
        z_steps = []
        out_steps = []

        wo = np.zeros(self.N)
        x = x0
        step = 0
        P = 1.0 / alpha * np.diag(np.ones(self.N))

        print('Starting simultaneous simulation and weight updates...')
        for t_step in time_steps:
            print('\rTime = %f' % t_step, end='', flush=True)
            step += 1
            # take Euler step
            dxdt = self._network_dynamics(t_step, x, external_input)
            x += dxdt * dt
            r = np.tanh(x)
            z = dot(wo, r)

            if step % learn_every == 0:
                # update inv. correlation matrix
                k = dot(P, r)
                rPr = dot(r, k)
                c = 1.0 / (1.0 + rPr)
                P -= c * np.outer(k, k)

                # update error, output and recurrent weights
                e = z - out_func(t_step)
                dw = -c * e * k # why c???
                wo += dw
                self.Wrec += dw.transpose()

            z_steps.append(z)
            out_steps.append(out_func(t_step))
            wo_length.append(np.sqrt(dot(wo, wo)))

        print('')
        print('Done!')
        z_steps = np.array(z_steps)
        out_steps = np.array(out_steps)
        diff = z_steps - out_steps
        fig = plt.figure(1)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.semilogy(time_steps, np.sqrt(diff * diff), 'k-', linewidth=0.5)
        ax1.set_ylabel('Error')
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.semilogy(time_steps, wo_length, 'k-', linewidth=0.5)
        ax2.set_ylabel('|wo|')
        ax2.set_xlabel('Time (ms)')
        plt.show()

        return wo, self.Wrec

    def simulate_learn_network_two_outputs(self, out_func, external_input=None, x0=None, T=2000.0, dt=None):
        """
        main function to run recurrent network simulations and use the FORCE algorithm
        to update recurrent and output weights.
        Simple implementation for fully recurrent networks from Sussillo & Abbott 2009
        :param out_func: callable, output function to be learned
        :param external_input: callable, accepts simulation time t as input
        and returns array of external input to each neuron
        :param x0: array of initial values for each neuron (length self.N)
        :return: solution
        """
        if external_input is None:
            def external_input(t):
                return np.zeros(self.N)
        if x0 is None:
            x0 = np.tanh(self.rng.randn(self.N))

        dot = np.dot
        alpha = 1.0e-0
        nsecs = T # default: 1440
        dt = 0.1
        learn_every = 2
        time_steps = np.arange(0.0, nsecs, dt)

        wo_length = []
        z_steps = []
        out_steps = []

        wo1 = np.zeros(self.N)
        wo2 = np.zeros(self.N)
        x = x0
        step = 0
        P = 1.0 / alpha * np.diag(np.ones(self.N))

        print('Starting simultaneous simulation and weight updates...')
        for t_step in time_steps:
            print('\rTime = %f' % t_step, end='', flush=True)
            step += 1
            # take Euler step
            dxdt = self._network_dynamics(t_step, x, external_input)
            x += dxdt * dt
            r = np.tanh(x)
            z1 = dot(wo1, r)
            z2 = dot(wo2, r)

            if step % learn_every == 0:
                # update inv. correlation matrix
                k = dot(P, r)
                rPr = dot(r, k)
                c = 1.0 / (1.0 + rPr)
                P -= c * np.outer(k, k)

                # update error, output and recurrent weights
                # implements alternating between outputs
                if step % (learn_every * 2) == 0:
                    e = z1 - out_func(t_step)[0]
                    dw = -c * e * k # why c???
                    wo1 += dw
                else:
                    e = z2 - out_func(t_step)[1]
                    dw = -c * e * k # why c???
                    wo2 += dw
                self.Wrec += dw.transpose()

            z_steps.append((z1, z2))
            out_steps.append(out_func(t_step))
            wo_length.append(np.sqrt(dot(wo1, wo1) + dot(wo2, wo2)))

        print('')
        print('Done!')
        z_steps = np.array(z_steps)
        out_steps = np.array(out_steps)
        diff_ = z_steps - out_steps
        diff = diff_[:, 0] * diff_[:, 0] + diff_[:, 1] * diff_[:, 1]
        fig = plt.figure(1)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.semilogy(time_steps, np.sqrt(diff), 'k-', linewidth=0.5)
        ax1.set_ylabel('Error')
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.semilogy(time_steps, wo_length, 'k-', linewidth=0.5)
        ax2.set_ylabel('|wo|')
        ax2.set_xlabel('Time (ms)')
        plt.show()

        return wo1, wo2, self.Wrec
