import numpy as np, json
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sm.datasets.elnino.load_pandas().data
X = d.iloc[:, 1:].to_numpy(); X = X - X.mean(axis=0, keepdims=True)
n, m = X.shape
grid = (np.arange(m)+0.5)/m

def smooth_cov(Xs, h):
    C = (Xs.T @ Xs)/len(Xs)
    D = grid[:,None]-grid[None,:]; D = np.minimum(np.abs(D), 1-np.abs(D))
    K = np.maximum(0, 1-(D/h)**2); W = K/K.sum(1, keepdims=True)
    return W @ C @ W.T

def Pq(C, q):
    s_ = m//q
    return np.mean([np.roll(np.roll(C,l*s_,0),l*s_,1) for l in range(q)], axis=0)

qs = [1,2,3,4,6,12]; rng = np.random.default_rng(0)
full, stat = {}, {}
for h in [0.10, 0.18, 0.30]:
    rf = {q: [] for q in qs}; rs = {q: [] for q in qs}
    for split in range(60):
        idx = rng.permutation(n); A,B = idx[:30], idx[30:]
        C_A = smooth_cov(X[A], h)
        CB = (X[B].T@X[B])/len(B)
        for q in qs:
            P = Pq(C_A,q)
            rf[q].append(np.mean((P-CB)**2))
            rs[q].append(np.mean((P-Pq(CB,12))**2))
    full[h] = [np.mean(rf[q]) for q in qs]; stat[h] = [np.mean(rs[q]) for q in qs]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
cols = {0.10:"tab:blue", 0.18:"tab:orange", 0.30:"tab:green"}
for h in full:
    ax[0].plot(qs, full[h], "o-", color=cols[h], label=f"$h={h}$")
    ax[1].plot(qs, stat[h], "o-", color=cols[h], label=f"$h={h}$")
ax[0].set_title("Target: full covariance $C$\n(selection refuses symmetry: $\\hat q=1$)")
ax[1].set_title("Target: stationary component $\\Pi_{12}C$\n(gains saturate at the resolution scale)")
for a in ax:
    a.set_xlabel("symmetry level $q$"); a.set_xscale("log"); a.set_xticks(qs); a.set_xticklabels(qs)
    a.legend(fontsize=8)
ax[0].set_ylabel("hold-out risk")
plt.tight_layout(); plt.savefig("fig_elnino.pdf", dpi=150)
OUT = {"full": {str(h): [round(v,4) for v in full[h]] for h in full},
       "stat": {str(h): [round(v,4) for v in stat[h]] for h in stat}}
json.dump(OUT, open("elnino_numbers.json","w"), indent=1)
print(json.dumps(OUT, indent=1))
