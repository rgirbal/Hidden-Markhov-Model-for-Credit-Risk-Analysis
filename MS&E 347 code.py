# inspired by aldengolab and his HMM implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import datetime as dt
import seaborn as sns

default_data = pd.ExcelFile('defaultdata.xlsx').parse()
default_data = default_data.groupby(['date'])['date'].count()

start_date = str(default_data.index[0].year) + '-01-01'
end_date = str(default_data.index[-1].year + 1) + '-01-01'

date_ranges = [pd.date_range(start=start_date, end=end_date, freq='1D'),
               pd.date_range(start=start_date, end=end_date, freq='1W'),
               pd.date_range(start=start_date, end=end_date, freq='1M'),
               pd.date_range(start=start_date, end=end_date, freq='3M'),
               pd.date_range(start=start_date, end=end_date, freq='1Y')]

default_data_daily = default_data.reindex(date_ranges[0]).fillna(0).cumsum()

datasets = []
for dr in date_ranges:
    default_data_dr = default_data_daily.reindex(dr)
    datasets.append(default_data_dr.diff().fillna(default_data_dr))


import scipy as sp
from scipy import stats
from scipy.special import logsumexp
from numpy import seterr

class PHMM:

    def __init__(self, init_delta, init_theta, init_lambdas, conv=5e-03):
        seterr(divide='ignore')
        self.nstates = len(init_delta)
        self.delta = np.log(init_delta)
        self.theta = np.log(init_theta)
        self.lambdas = np.array(init_lambdas)
        self.conv = conv * np.square(self.nstates)
        seterr(divide='warn')

    """
    Returns the transition probability matrix (NOT log probabilities)
    """
    def transition_matrix(self):
        return np.exp(self.theta)

    """
    A Poisson random variable generator.
    """
    def poisson_var_gen(self, mean):
        if mean == -1:
            return -1
        else:
            return stats.poisson(mean).rvs()

    """
    A Poisson log probability mass function.

    """
    def poisson_log_mass(self, mean, val):
        if mean == -1:
            if val == -1:
                return 0
            else:
                return -np.inf
        elif mean >= 0:
            if val == -1:
                return -np.inf
            else:
                return stats.poisson(mean).logpmf(val)

    """
    Generates a random sequence of integers following the class's current
    PHMM specification.

    """
    def sequence_gen(self, n=100):
        out_seq = [] # Eventual output sequence
        states = []  # True hidden states
        state = np.random.choice(a=self.nstates, p=np.exp(self.delta))
        out_seq.append(self.poisson_var_gen(self.lambdas[state]))
        states.append(state)
        # Our stop condition, which differs based on whether we have a
        # stop state
        def condition():
            if self.lambdas[-1] == -1:
                return out_seq[-2] != -1
            else:
                return len(out_seq) < n
        while condition():
            state = np.random.choice(a=self.nstates, p=np.exp(self.theta[state]))
            out_seq.append(self.poisson_var_gen(self.lambdas[state]))
            states.append(state)
        return out_seq, states

    """
    Calculates the forward state log probabilities of a PHMM observation sequence.

    """
    def forward_lprobs(self, seq):
        seterr(divide='ignore')
        g_1 = [self.poisson_log_mass(self.lambdas[i], seq[0]) for i in range(self.nstates)]
        g_1 = np.add(self.delta, g_1)
        glst = [g_1]
        for i in range(1, len(seq)):
            g_i = []
            for j in range(self.nstates):
                prev = np.add(glst[-1], self.theta[::, j])
                prev = logsumexp(prev)
                g_ij = prev + self.poisson_log_mass(self.lambdas[j], seq[i])
                g_i.append(g_ij)
            glst.append(g_i)
        g_n = glst[-1]
        seterr(divide='warn')
        return np.array(glst)

    """
    Calculates the forward log probability of observing a given sequence. Should
    be equal to the backward log probability of the same sequence.

    """
    def forward_lprob(self, seq):
        glst = self.forward_lprobs(seq)
        return logsumexp(glst[-1])

    """
    Calculates the backward state log probabilities of a PHMM observation sequence.

    """
    def backward_lprobs(self, seq):
        seterr(divide='ignore')
        f_n = [self.poisson_log_mass(self.lambdas[i], seq[-1]) for i in range(self.nstates)]
        flst = [f_n]
        for i in range(len(seq) - 2, -1, -1):
            f_i = []
            for j in range(self.nstates):
                prev = np.add(self.theta[j], flst[-1])
                prev = logsumexp(prev)
                f_ij = self.poisson_log_mass(self.lambdas[j], seq[i]) + prev
                f_i.append(f_ij)
            flst.append(f_i)
        flst.reverse()
        seterr(divide='warn')
        return np.array(flst)

    """
    Calculates the backward log probability of observing a given sequence. Should
    be equal to the forward log probability of the same sequence.

    """
    def backward_lprob(self, seq):
        flst = self.backward_lprobs(seq)
        f_1 = np.add(self.delta, flst[0])
        return logsumexp(f_1)

    """
    Calculates the probability of being in each state for every element of
    an observation sequence using the forward-backward algorithm.

    """
    def forward_backward(self, seq):
        fprobs = self.forward_lprobs(seq)
        bprobs = self.backward_lprobs(seq)
        probs = np.add(fprobs, bprobs)
        probsums = list(map(logsumexp, probs))
        norm_probs = list(map(lambda lst, sum_: list(map(lambda x: x - sum_, lst)), probs, probsums))
        return np.exp(norm_probs)

    """
    Calculates the log-likelihood of the given HMM generating the
    provided sequences.

    """
    def log_likelihood(self, seqlst):
        probs = list(map(self.forward_lprob, seqlst))
        return np.sum(probs)

    """
    Calculates the most likely state path of an observation sequence.

    """
    def viterbi(self, seq):
        v_n = [0.0 for _ in range(self.nstates)]
        vlst = [v_n]
        wlst = []
        for i in range(len(seq) - 1, 0, -1):
            v_i = []
            w_i = []
            for j in range(self.nstates):
                all_v_ij = []
                for k in range(self.nstates):
                    temp = self.theta[j, k] + self.poisson_log_mass(self.lambdas[k], seq[i])
                    temp += vlst[-1][k]
                    all_v_ij.append(temp)
                v_i.append(max(all_v_ij))
                w_i.append(np.argmax(all_v_ij))
            vlst.append(v_i)
            wlst.append(w_i)
        wlst.reverse()
        first_prob = [self.poisson_log_mass(self.lambdas[i], seq[0]) for i in range(self.nstates)]
        first_prob = np.add(first_prob, self.delta)
        first_prob = np.add(first_prob, vlst[-1])
        h_1 = np.argmax(first_prob)
        statelst = [h_1]
        for i in range(len(wlst)):
            statelst.append(wlst[i][statelst[-1]])
        return statelst

    """
    Trains the PHMM on a set of observation sequences using the Baum-Welch
    algorithm, a special case of the EM algorithm.


    """
    def baum_welch(self, seqlst, max_iter=200):
        itr = 0
        transition_probs = self.theta
        lambdalast = self.lambdas
        prev_transition_probs = None
        prev_lambdalast = None

        # Convergence-checking function
        def assess_convergence(ll=False):
            if prev_transition_probs is None or prev_lambdalast is None:
                return False
            diff = []
            bools = []
            for i in range(len(transition_probs)):
                for j in range(len(transition_probs[i])):
                    if np.isneginf(transition_probs[i, j]) or np.isneginf(prev_transition_probs[i, j]) or prev_transition_probs[i, j] == 0:
                        pass
                    else:
                        d = abs((transition_probs[i, j] - prev_transition_probs[i, j]) / prev_transition_probs[i, j])
                        diff.append(d)
                        bools.append(d <= self.conv)
            for i in range(len(lambdalast)):
                d = abs((lambdalast[i] - prev_lambdalast[i]) / prev_lambdalast[i])
                diff.append(d)
                bools.append(d <= self.conv)
            # print("Difference: ", sum(diff))
            if ll:
                print("Log-Likelihood:", self.log_likelihood(seqlst))
            return all(bools)

        while not assess_convergence() and itr < max_iter:
            prev_transition_probs = transition_probs
            prev_lambdalast = lambdalast
            trans_lst = []
            seqs = [d for sub in seqlst for d in sub]
            rs = []
            for seq, k in zip(seqlst, range(len(seqlst))):
                flst = self.backward_lprobs(seq)
                rlst = []
                r_1_hat = np.add(self.delta, flst[0])
                r_1_sum = logsumexp(r_1_hat)
                r1 = list(map(lambda r: r - r_1_sum, r_1_hat))
                rlst.append(r1)
                tlst = []
                for i in range(1, len(seq)):
                    t_i_hat = []
                    # Indexed the same as transition matrix
                    for j in range(self.nstates):
                        t_i_hat.append(rlst[-1][j] + np.add(transition_probs[j], flst[i]))
                    t_i_sum = logsumexp(t_i_hat)
                    t_i = list(map(lambda lst: list(map(lambda t: t - t_i_sum, lst)), t_i_hat))
                    r_i = np.array(list(map(logsumexp, zip(*t_i))))
                    tlst.append(t_i)
                    rlst.append(r_i)
                expd_trans = []
                for j in range(len(transition_probs)):
                    expd_trans.append([])
                    for l in range(len(transition_probs[j])):
                        t_ij = [tlst[t][j][l] for t in range(len(tlst))]
                        expd_trans[j].append(logsumexp(t_ij))
                trans_lst.append(expd_trans)
                # Now we calculate expected values of the Poisson parameters (wow!)
                rs.append(rlst)
            # Update initial probs
            r1s = [d[0] for d in rs]
            new_delta = []
            for j in range(len(self.delta)):
                d_i = logsumexp([r1s[t][j] for t in range(len(r1s))])
                new_delta.append(d_i)
            del_sum = logsumexp(new_delta)
            new_delta = np.array(list(map(lambda x: x - del_sum, new_delta)))
            # Update Poisson parameters
            rs = [d for sub in rs for d in sub]
            seq_probs = np.exp(list(zip(*rs)))
            sums = np.array(list(map(sum, seq_probs)))
            scaled_vals = seqs * seq_probs
            expd_vals = np.array(list(map(sum, scaled_vals)))
            pmeans = expd_vals / sums
            lambdalast = pmeans
            # Update transition matrix
            expd_trans = []
            for j in range(len(transition_probs)):
                    expd_trans.append([])
                    for l in range(len(transition_probs[j])):
                        t_ij = [trans_lst[t][j][l] for t in range(len(trans_lst))]
                        expd_trans[j].append(logsumexp(t_ij))
            totals = list(map(logsumexp, expd_trans))
            new_trans = list(map(lambda tlst, t: list(map(lambda et: et - t, tlst)), expd_trans, totals))
            transition_probs = np.array(new_trans)
            # Apply updates
            self.delta = new_delta
            self.theta = transition_probs
            self.lambdas = lambdalast
            itr += 1


scaling = [365, 52, 12, 4, 1]
freq = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']

for K in range(2, 5):
    for i in reversed(range(1, len(datasets))):
        X = datasets[i].values

        init_delta = np.random.rand(K)
        init_delta /= np.sum(init_delta)

        init_theta = np.random.rand(K, K)
        init_theta /= np.sum(init_theta, axis=1)

        init_lambdas = np.linspace(X.min() + (X.max() - X.min()) / (K + 1.),
                                   X.max() - (X.max() - X.min()) / (K + 1.), K)

        model = PHMM(init_delta=init_delta,
                     init_theta=init_theta,
                     init_lambdas=init_lambdas)
        model.baum_welch([X])

        plt.figure()
        plt.bar(x=range(1970, 2013), height=datasets[-1].values, color='grey')
        plt.plot(np.linspace(1970, 2013, len(X)), scaling[i] * model.lambdas[model.viterbi(X)], color='black')
        plt.savefig('k' + str(K) + freq[i] + '.png', dpi=300)
        plt.show()

K = 3
X = datasets[-1].values

init_delta = np.random.rand(K)
init_delta /= np.sum(init_delta)

init_theta = np.random.rand(K, K)
init_theta /= np.sum(init_theta, axis=1)

init_lambdas = np.linspace(X.min() + (X.max() - X.min()) / (K + 1.),
                           X.max() - (X.max() - X.min()) / (K + 1.), K)

out_of_sample_lambdas = []
for limit in range(25, len(X)):
    model = PHMM(init_delta=init_delta,
                 init_theta=init_theta,
                 init_lambdas=init_lambdas)

    model.baum_welch([X[:limit]])
    in_sample_latent_state = model.viterbi(X[:limit])[-1]

    out_of_sample_lambda_dist = np.exp(model.theta[in_sample_latent_state])
    out_of_sample_lambdas.append(model.lambdas.dot(out_of_sample_lambda_dist))


violin_df = pd.DataFrame({'year': np.repeat(1996 + np.arange(18), 100), 'lambda': np.repeat(out_of_sample_lambdas, 100)})
violin_df['sample'] = np.random.poisson(violin_df['lambda'].values)
violin_df['realized'] = np.repeat(X[25:], 100)

fig, ax = plt.subplots(figsize=(15, 10))
g = sns.violinplot(x='year', y='sample', data=violin_df, color='skyblue')
sns.stripplot(x='year', y='realized', data=violin_df, ax=g, color='red')
plt.savefig('violink3yearly.png', dpi=300)

K = 4
X = datasets[-2].values

init_delta = np.random.rand(K)
init_delta /= np.sum(init_delta)

init_theta = np.random.rand(K, K)
init_theta /= np.sum(init_theta, axis=1)

init_lambdas = np.linspace(X.min() + (X.max() - X.min()) / (K + 1.),
                           X.max() - (X.max() - X.min()) / (K + 1.), K)

out_of_sample_lambdas = []
for limit in range(25 * 4, len(X), 4):
    model = PHMM(init_delta=init_delta,
                 init_theta=init_theta,
                 init_lambdas=init_lambdas)

    model.baum_welch([X[:limit]])
    in_sample_latent_state = model.viterbi(X[:limit])[-1]

    p = np.exp(model.theta)

    out_of_sample_lambda_dist0 = p[in_sample_latent_state]
    out_of_sample_lambda_dist1 = p.dot(p)[in_sample_latent_state]
    out_of_sample_lambda_dist2 = p.dot(p.dot(p))[in_sample_latent_state]
    out_of_sample_lambda_dist3 = p.dot(p).dot(p.dot(p))[in_sample_latent_state]

    out_of_sample_lambdas.append(model.lambdas.dot(out_of_sample_lambda_dist0 +
                                                   out_of_sample_lambda_dist1 +
                                                   out_of_sample_lambda_dist2 +
                                                   out_of_sample_lambda_dist3))

violin_df = pd.DataFrame({'year': np.repeat(1996 + np.arange(18), 100), 'lambda': np.repeat(out_of_sample_lambdas, 100)})
violin_df['sample'] = np.random.poisson(violin_df['lambda'].values)
violin_df['realized'] = np.repeat(datasets[-1].values[25:], 100)

fig, ax = plt.subplots(figsize=(15, 10))
g = sns.violinplot(x='year', y='sample', data=violin_df, color='skyblue')
sns.stripplot(x='year', y='realized', data=violin_df, ax=g, color='red')
plt.savefig('violink4quarterly.png', dpi=300)

chisquare_tests = np.zeros((3, len(datasets) - 1))
p_values = np.zeros((3, len(datasets) - 1))

for k in range(2, 5):
    for i in reversed(range(1, len(datasets))):
        X = datasets[i].values

        init_delta = np.random.rand(K)
        init_delta /= np.sum(init_delta)

        init_theta = np.random.rand(K, K)
        init_theta /= np.sum(init_theta, axis=1)

        init_lambdas = np.linspace(X.min() + (X.max() - X.min()) / (K + 1.),
                                   X.max() - (X.max() - X.min()) / (K + 1.), K)

        model = PHMM(init_delta=init_delta,
                     init_theta=init_theta,
                     init_lambdas=init_lambdas)
        model.baum_welch([X])
        chisquare_tests[k - 2][i - 1], p_values[k - 2][i - 1] = stats.chisquare(f_obs=X,
                                                                                f_exp=model.lambdas[model.viterbi(X)])
        print('k = ' + str(k) + ', i = ' + str(i) + ' chisq = ' + str(chisquare_tests[k - 2][i - 1]) + ' pval = ' + str(
            p_values[k - 2][i - 1]))

print(chisquare_tests)
print(p_values)