import numpy as np
rng = np.random.default_rng(1)

# gamma extremal Holder-beta : lambda_k = c k^{-(beta+1)}, deux-cotes
def make_spectrum(beta, Kmax=4096):
    lam = np.zeros(Kmax+1)
    lam[1:] = np.arange(1, Kmax+1, dtype=float)**(-(beta+1))
    lam[0] = 1.0
    lam /= (lam[0] + 2*lam[1:].sum())   # gamma(0)=1
    return lam

def gamma_of(lam, u):
    k = np.arange(1, len(lam))
    return lam[0] + 2*np.sum(lam[1:,None]*np.cos(2*np.pi*k[:,None]*u[None,:]), axis=0)

beta = 1.0
lam = make_spectrum(beta)
# --- Lemme de masse dyadique : somme de bloc <= L R K^{-beta}
print("=== Lemme de masse dyadique (beta=1) ===")
u = np.linspace(1e-4, 0.25, 400)
g = gamma_of(lam, u)
LR = np.max((1-g)/u**beta)   # constante Holder empirique de gamma
for j in range(2, 9):
    K = 2**j
    block = 2*lam[K:2*K].sum()
    borne = LR * (4*K)**(-beta) * 4**beta  # forme (3.1): 4^{-b} L R K^{-b} -> ici L R (4K)^{-b}... verifions la forme brute
    print(f"  K=2^{j}: masse bloc = {block:.5f}  vs  L_R*(4K)^-b = {LR*(4*K)**(-beta):.5f}  ratio={block/(LR*(4*K)**(-beta)):.2f} (<=1 attendu)")

# --- Estimateur : biais et echelles de variance
def simulate_subject(lam, m, rng, sigma=0.3, Kmax=None):
    Kmax = len(lam)-1
    N = rng.poisson(m)
    if N < 2: N = 2
    T = rng.uniform(0, 1, N)
    k = np.arange(1, Kmax+1)
    a = rng.normal(0,1,Kmax); b = rng.normal(0,1,Kmax)
    X = (np.sqrt(lam[0])*rng.normal() +
         np.sqrt(2*lam[1:])[None,:] @ np.zeros((Kmax,1))).item() # placeholder
    # vectorise : X(T) = sqrt(l0) g0 + sum sqrt(2 lk)(a cos + b sin)
    phase = 2*np.pi*np.outer(k, T)
    X_T = np.sqrt(lam[0])*rng.normal() + (np.sqrt(2*lam[1:])[:,None]*(a[:,None]*np.cos(phase)+b[:,None]*np.sin(phase))).sum(0)
    Y = X_T + sigma*rng.normal(0,1,N)
    return T, Y

def lambda_hat(subjects, kgrid, m):
    n = len(subjects)
    acc = np.zeros(len(kgrid))
    for T, Y in subjects:
        S = (Y[None,:]*np.exp(-2j*np.pi*np.outer(kgrid, T))).sum(1)
        acc += (np.abs(S)**2 - (Y**2).sum())
    return acc/(n*m*m)

print("\n=== Biais et variance de lambda_hat (n=400, m=8, 40 reps) ===")
n, m = 400, 8
kcheck = np.array([1, 4, 16, 64, 256])
est = []
for rep in range(40):
    subs = [simulate_subject(lam, m, rng, Kmax=1024) for _ in range(n)]
    est.append(lambda_hat(subs, kcheck, m))
est = np.array(est)
tau2 = 1/(n*m*m)
print("k     lambda_k    moy(est)    biais/se   Var_emp      borne C(l^2/n+l/nm+1/nm^2)")
for i, k in enumerate(kcheck):
    lk = lam[k]
    borne = lk*lk/n + lk/(n*m) + tau2
    se = est[:,i].std()/np.sqrt(40)
    print(f"{k:4d}  {lk:.2e}  {est[:,i].mean():.2e}  {abs(est[:,i].mean()-lk)/se:5.2f}     {est[:,i].var():.2e}   {borne:.2e}")
