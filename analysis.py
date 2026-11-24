"""HW8 Problem 2: statistical analysis of PPO A (fixed lr) vs B (annealed lr).
Usage: python analysis.py path/to/ppo_AvsB_logs.pkl"""
import sys, json
import numpy as np, pandas as pd, scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = pd.read_pickle(sys.argv[1])
df, mt = data['optimization'], data['metadata']
df['config_name'] = df['exp_id'].map(lambda s: int(s.split('_')[1]))
df['run_id'] = df['exp_id'].map(lambda s: int(s.split('_')[-1]))
anneal = mt.groupby('config_name')['anneal_lr'].first().to_dict()
logs = {c: {r: np.concatenate(g['ep_rewards']) for r, g in cdf.groupby('run_id')}
        for c, cdf in df.groupby('config_name')}
# score per run = mean return over last 100 training episodes, ordered by run_id
scores = {c: np.array([logs[c][r][-100:].mean() for r in sorted(logs[c])]) for c in logs}
A, B = scores[0], scores[1]
out = {'anneal': {int(k): bool(v) for k, v in anneal.items()},
       'n_eps': int(len(logs[0][0])), 'nA': len(A), 'nB': len(B)}

iqm = lambda x, axis=-1: scipy.stats.trim_mean(x, 0.25, axis=axis)
METRICS = {'mean': lambda x, axis=-1: np.mean(x, axis=axis),
           'median': lambda x, axis=-1: np.median(x, axis=axis), 'iqm': iqm}

# Q1/Q3 full-data aggregates
out['full'] = {k: [float(f(A)), float(f(B))] for k, f in METRICS.items()}
out['describe'] = {'A': [float(v) for v in np.percentile(A, [0, 25, 50, 75, 100])],
                   'B': [float(v) for v in np.percentile(B, [0, 25, 50, 75, 100])],
                   'stdA': float(A.std(ddof=1)), 'stdB': float(B.std(ddof=1)),
                   'fracA_-200': float(np.mean(A <= -199.99)), 'fracB_-200': float(np.mean(B <= -199.99))}

# Q2/Q3 bootstrap: resample m runs (with replacement) from each variant, 10k times
rng = np.random.default_rng(0)
NB = 10_000
out['boot'] = {}
diffs = {}
for m in [30, 5, 1]:
    iA = rng.integers(0, len(A), (NB, m)); iB = rng.integers(0, len(B), (NB, m))
    for k, f in METRICS.items():
        sA, sB = f(A[iA], axis=1), f(B[iB], axis=1)
        d = sB - sA
        diffs[(k, m)] = d
        out['boot'][f'{k}_m{m}'] = {'P(B>A)': float(np.mean(d > 0)), 'P(B=A)': float(np.mean(d == 0)),
                                    'P(B<A)': float(np.mean(d < 0)),
                                    'mean_diff': float(d.mean()), 'sd_diff': float(d.std()),
                                    'ciA': [float(v) for v in np.quantile(sA, [.025, .975])],
                                    'ciB': [float(v) for v in np.quantile(sB, [.025, .975])],
                                    'ci_diff': [float(v) for v in np.quantile(d, [.025, .975])]}

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
for i, k in enumerate(METRICS):
    for j, m in enumerate([30, 5, 1]):
        ax = axes[i, j]; d = diffs[(k, m)]
        ax.hist(d, bins=60, color='tab:blue', alpha=.8)
        ax.axvline(0, color='k', lw=1)
        ax.set_title(f'{k}, m={m}:  P(B>A)={np.mean(d > 0):.2f}', fontsize=10)
        if i == 2: ax.set_xlabel('score(B) - score(A)')
fig.tight_layout(); fig.savefig('figures/bootstrap_hist.png', dpi=150)

fig, ax = plt.subplots(figsize=(6, 3.5))
bins = np.linspace(-200, min(A.min(), B.min()) * 0 + max(A.max(), B.max()) + 1, 30)
ax.hist(A, bins=bins, alpha=.6, label='A (fixed lr)'); ax.hist(B, bins=bins, alpha=.6, label='B (annealed lr)')
ax.set_xlabel('score (mean return, last 100 episodes)'); ax.set_ylabel('runs'); ax.legend()
fig.tight_layout(); fig.savefig('figures/scores_hist.png', dpi=150)

# Q4 confidence intervals without bootstrap (t-interval for the mean)
def t_ci(x):
    h = scipy.stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]
out['tci'] = {'A': t_ci(A), 'B': t_ci(B)}
w = scipy.stats.ttest_ind(B, A, equal_var=False)
se = np.sqrt(A.var(ddof=1) / len(A) + B.var(ddof=1) / len(B))
dfw = se**4 / ((A.var(ddof=1) / len(A))**2 / (len(A) - 1) + (B.var(ddof=1) / len(B))**2 / (len(B) - 1))
h = scipy.stats.t.ppf(.975, dfw) * se
out['tci']['diff'] = [float(B.mean() - A.mean() - h), float(B.mean() - A.mean() + h)]
# standard bootstrap with m = n (for reference)
iA = rng.integers(0, len(A), (NB, len(A))); iB = rng.integers(0, len(B), (NB, len(B)))
out['boot_n'] = {k: {'ciA': [float(v) for v in np.quantile(f(A[iA], axis=1), [.025, .975])],
                     'ciB': [float(v) for v in np.quantile(f(B[iB], axis=1), [.025, .975])],
                     'ci_diff': [float(v) for v in np.quantile(f(B[iB], axis=1) - f(A[iA], axis=1), [.025, .975])]}
                 for k, f in METRICS.items()}

# Q5 Welch t-test, all 50 and first 10/20/30 runs
out['welch'] = {}
for n in [10, 20, 30, 50]:
    r = scipy.stats.ttest_ind(B[:n], A[:n], equal_var=False)
    out['welch'][n] = {'t': float(r.statistic), 'p': float(r.pvalue),
                       'meanA': float(A[:n].mean()), 'meanB': float(B[:n].mean())}
print(json.dumps(out, indent=1))
