import numpy as np
rng = np.random.default_rng(7)

def make_spectrum(beta, Kmax=4096):
    lam = np.zeros(Kmax+1)
    lam[1:] = np.arange(1, Kmax+1, dtype=float)**(-(beta+1))
    lam[0] = 1.0
    lam /= (lam[0] + 2*lam[1:].sum())
    return lam

def simulate(lam, n, m, rng, sigma=0.3):
    Kmax = len(lam)-1
    k = np.arange(1, Kmax+1)
    subs = []
    for i in range(n):
        N = max(2, rng.poisson(m))
        T = rng.uniform(0,1,N)
        a = rng.normal(0,1,Kmax); b = rng.normal(0,1,Kmax)
        ph = 2*np.pi*np.outer(k, T)
        X = np.sqrt(lam[0])*rng.normal() + (np.sqrt(2*lam[1:])[:,None]*(a[:,None]*np.cos(ph)+b[:,None]*np.sin(ph))).sum(0)
        subs.append((T, X + sigma*rng.normal(0,1,N)))
    return subs

def spectral_estimator(subs, m, beta, R_A, Lthr=1.0):
    n = len(subs)
    tau = 1.0/(np.sqrt(n)*m)
    Kn = int((10*tau**(-(2*beta+1)/(beta+1)))**(1/(2*beta))) + 2
    kg = np.arange(1, Kn+1)
    acc = np.zeros(Kn); acc0 = 0.0
    for T, Y in subs:
        E = np.exp(-2j*np.pi*np.outer(kg, T))
        S = E @ Y
        acc += np.abs(S)**2 - (Y**2).sum()
        acc0 += (Y.sum())**2 - (Y**2).sum()
    lam_hat = acc/(n*m*m); lam0_hat = acc0/(n*m*m)
    # seuillage par blocs
    lam_t = np.zeros(Kn); j = 0
    while 2**j <= Kn:
        lo, hi = 2**j, min(2**(j+1), Kn+1)
        d = hi-lo
        Abar = R_A * (2.0**(-j*beta))
        r = d*tau/Abar
        blk = lam_hat[lo-1:hi-1]
        if r <= 1:
            lam_t[lo-1:hi-1] = np.maximum(blk, 0)
        else:
            t = Lthr*tau*np.sqrt(np.log1p(r))
            lam_t[lo-1:hi-1] = np.maximum(blk-t, 0)
        j += 1
    return max(lam0_hat,0), lam_t

def spectral_risk(lam, lam0_t, lam_t):
    Kn = len(lam_t); Kmax = len(lam)-1
    r = (lam0_t-lam[0])**2 + 2*np.sum((lam_t-lam[1:Kn+1])**2)
    r += 2*np.sum(lam[Kn+1:]**2)
    return r

def smoother_risk(subs, lam, m, hs):
    # NW circulaire sur lags binnes ; risque L2 contre gamma vrai ; oracle sur h
    nb = 256
    edges = np.linspace(0,1,nb+1); ctr = (edges[:-1]+edges[1:])/2
    wsum = np.zeros(nb); zsum = np.zeros(nb)
    for T, Y in subs:
        U = (T[None,:]-T[:,None]).ravel(); P = np.outer(Y,Y).ravel()
        mask = ~np.eye(len(T),dtype=bool).ravel()
        U = np.mod(U[mask],1.0); P = P[mask]
        idx = np.minimum((U*nb).astype(int), nb-1)
        np.add.at(wsum, idx, 1.0); np.add.at(zsum, idx, P)
    k = np.arange(1, len(lam))
    gtrue = lam[0] + 2*(lam[1:,None]*np.cos(2*np.pi*np.outer(k,ctr))).sum(0)
    best = np.inf
    D = ctr[None,:]-ctr[:,None]; D = np.minimum(np.abs(D),1-np.abs(D))
    for h in hs:
        K = np.maximum(0, 1-(D/h)**2)
        gh = (K@zsum)/np.maximum(K@wsum, 1e-12)
        best = min(best, np.mean((gh-gtrue)**2))
    return best

hs = np.geomspace(0.008, 0.25, 8)
for beta, ns, reps in [(1.0, [500,2000,8000,32000], [6,5,4,3]),
                        (0.5, [500,2000,8000],       [6,5,4])]:
    lam = make_spectrum(beta)
    R_A = 2*sum(lam[2**0:2**1])  # calibre Abar sur le bloc j=0 (constante de classe)
    m = 8
    print(f"\n=== beta={beta} : cible spectrale {(2*beta+1)/(2*beta+2):.3f} vs lisseur {2*beta/(2*beta+1):.3f} ===")
    Ns, Rs_spec, Rs_smo = [], [], []
    for n, rep in zip(ns, reps):
        rs, rm = [], []
        for r_ in range(rep):
            subs = simulate(lam, n, m, rng)
            l0, lt = spectral_estimator(subs, m, beta, R_A)
            rs.append(spectral_risk(lam, l0, lt))
            rm.append(smoother_risk(subs, lam, m, hs))
        Ns.append(n*m*m); Rs_spec.append(np.mean(rs)); Rs_smo.append(np.mean(rm))
        print(f"  N={n*m*m:>8d}: spectral {np.mean(rs):.5f}  lisseur(oracle-h) {np.mean(rm):.5f}")
    sl_s = -np.polyfit(np.log(Ns), np.log(Rs_spec), 1)[0]
    sl_m = -np.polyfit(np.log(Ns), np.log(Rs_smo), 1)[0]
    print(f"  PENTES: spectral {sl_s:.3f} (cible {(2*beta+1)/(2*beta+2):.3f})  |  lisseur {sl_m:.3f} (cible {2*beta/(2*beta+1):.3f})")
