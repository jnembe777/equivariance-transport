import numpy as np
import statsmodels.api as sm

d = sm.datasets.elnino.load_pandas().data
X = d.iloc[:, 1:].to_numpy()
X = X - X.mean(axis=0, keepdims=True)
n, m = X.shape
grid = (np.arange(m)+0.5)/m

def smooth_cov(Xs, h):
    C_emp = (Xs.T @ Xs)/len(Xs)
    D = grid[:,None]-grid[None,:]
    D = np.minimum(np.abs(D), 1-np.abs(D))
    K = np.maximum(0, 1-(D/h)**2)
    W = K/K.sum(axis=1, keepdims=True)
    return W @ C_emp @ W.T

def Pq(C, q):
    s_ = m//q
    return np.mean([np.roll(np.roll(C,l*s_,0),l*s_,1) for l in range(q)], axis=0)

qs = [1,2,3,4,6,12]
rng = np.random.default_rng(0)
print("=== cible : composante stationnaire Pi_12 C  (risque hold-out, 60 splits) ===")
print("h      " + "".join(f"q={q:<8}" for q in qs) + " gain(q=12 vs 1)  q_sat(gain>=90%)")
for h in [0.09, 0.12, 0.18, 0.30]:
    risks = {q: [] for q in qs}
    for split in range(60):
        idx = rng.permutation(n); A, B = idx[:30], idx[30:]
        C_A = smooth_cov(X[A], h)
        target = Pq((X[B].T@X[B])/len(B), 12)      # composante stationnaire du temoin
        for q in qs:
            risks[q].append(np.mean((Pq(C_A,q)-target)**2))
    r = {q: np.mean(risks[q]) for q in qs}
    g12 = 1 - r[12]/r[1]
    # premier q atteignant 90% du gain total
    qsat = next(q for q in qs if (1-r[q]/r[1]) >= 0.9*g12)
    print(f"{h:.2f}   " + "".join(f"{r[q]:<10.4f}" for q in qs) + f" {100*g12:5.1f}%          q={qsat}")
print("\nprediction du cadre : q_sat ~ c_K/h -> DIMINUE quand h augmente")
