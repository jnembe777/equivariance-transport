import numpy as np
rng = np.random.default_rng(11)

def spectrum_flat(beta, Kmax=2048):
    lam = np.zeros(Kmax+1); lam[1:] = np.arange(1,Kmax+1,dtype=float)**(-(beta+1)); lam[0]=1.0
    lam /= (lam[0]+2*lam[1:].sum()); return lam

def spectrum_lacunary(beta, Jmax=11):
    Kmax = 2**Jmax
    lam = np.zeros(Kmax+1); lam[0]=0.3
    for j in range(Jmax):
        kj = int(1.5*2**j)          # une frequence par bloc dyadique
        lam[kj] += 0.5*2.0**(-j*beta)/2
    lam /= (lam[0]+2*lam[1:].sum()); return lam

def simulate(lam, n, m, rng, sigma=0.3):
    Kmax = len(lam)-1; k = np.arange(1,Kmax+1)
    nz = np.where(lam[1:]>0)[0]     # optimisation lacunaire : modes actifs seuls
    kk = k[nz]; ll = lam[1:][nz]
    subs=[]
    for i in range(n):
        N = max(2, rng.poisson(m)); T = rng.uniform(0,1,N)
        a = rng.normal(0,1,len(kk)); b = rng.normal(0,1,len(kk))
        ph = 2*np.pi*np.outer(kk,T)
        X = np.sqrt(lam[0])*rng.normal() + (np.sqrt(2*ll)[:,None]*(a[:,None]*np.cos(ph)+b[:,None]*np.sin(ph))).sum(0)
        subs.append((T, X+sigma*rng.normal(0,1,N)))
    return subs

def spectral(subs, lam, m, beta, R_A, L=1.0):
    n=len(subs); tau=1/(np.sqrt(n)*m)
    Kn=int((10*tau**(-(2*beta+1)/(beta+1)))**(1/(2*beta)))+2
    Kn=min(Kn, len(lam)-1)
    kg=np.arange(1,Kn+1); acc=np.zeros(Kn); a0=0.0
    for T,Y in subs:
        S=np.exp(-2j*np.pi*np.outer(kg,T))@Y
        acc+=np.abs(S)**2-(Y**2).sum(); a0+=(Y.sum())**2-(Y**2).sum()
    lh=acc/(n*m*m); l0=max(a0/(n*m*m),0)
    lt=np.zeros(Kn); j=0
    while 2**j<=Kn:
        lo,hi=2**j,min(2**(j+1),Kn+1); d=hi-lo
        r=d*tau/(R_A*2.0**(-j*beta)); blk=lh[lo-1:hi-1]
        lt[lo-1:hi-1]=np.maximum(blk,0) if r<=1 else np.maximum(blk-L*tau*np.sqrt(np.log1p(r)),0)
        j+=1
    risk=(l0-lam[0])**2+2*np.sum((lt-lam[1:Kn+1])**2)+2*np.sum(lam[Kn+1:]**2)
    return risk

def smoother(subs, lam, hs):
    nb=256; ctr=(np.arange(nb)+0.5)/nb
    ws=np.zeros(nb); zs=np.zeros(nb)
    for T,Y in subs:
        U=(T[None,:]-T[:,None]).ravel(); P=np.outer(Y,Y).ravel()
        msk=~np.eye(len(T),dtype=bool).ravel()
        U=np.mod(U[msk],1.0); P=P[msk]
        idx=np.minimum((U*nb).astype(int),nb-1)
        np.add.at(ws,idx,1.0); np.add.at(zs,idx,P)
    k=np.arange(1,len(lam)); nz=np.where(lam[1:]>0)[0]
    gt=lam[0]+2*(lam[1:][nz][:,None]*np.cos(2*np.pi*np.outer(k[nz],ctr))).sum(0)
    D=ctr[None,:]-ctr[:,None]; D=np.minimum(np.abs(D),1-np.abs(D))
    best=np.inf
    for h in hs:
        K=np.maximum(0,1-(D/h)**2)
        gh=(K@zs)/np.maximum(K@ws,1e-12)
        best=min(best,np.mean((gh-gt)**2))
    return best

hs=np.geomspace(0.008,0.25,6); m=8
for beta, ns, reps in [(1.0,[500,2000,8000],[5,4,3]), (0.5,[500,2000,8000],[5,4,3])]:
    print(f"\n=== beta={beta} ===")
    out={}
    for name, lam in [("flat",spectrum_flat(beta)), ("lacunaire",spectrum_lacunary(beta))]:
        R_A=2*lam[1:2].sum()+lam[0]*0+2*max(lam[1:4].sum(),0.05)
        Ns,Rs,Rm=[],[],[]
        for n,rep in zip(ns,reps):
            rs,rm=[],[]
            for _ in range(rep):
                subs=simulate(lam,n,m,rng)
                rs.append(spectral(subs,lam,m,beta,R_A)); rm.append(smoother(subs,lam,hs))
            Ns.append(n*m*m); Rs.append(np.mean(rs)); Rm.append(np.mean(rm))
        out[name]=(np.array(Ns),np.array(Rs),np.array(Rm))
        print(f"  membre {name:9s}: spectral {['%.5f'%x for x in Rs]}  lisseur {['%.5f'%x for x in Rm]}")
    # sup sur les deux membres, par N
    Ns=out["flat"][0]
    supS=np.maximum(out["flat"][1],out["lacunaire"][1])
    supM=np.maximum(out["flat"][2],out["lacunaire"][2])
    slS=-np.polyfit(np.log(Ns),np.log(supS),1)[0]; slM=-np.polyfit(np.log(Ns),np.log(supM),1)[0]
    print(f"  SUP(2 membres) pentes: spectral {slS:.3f} (cible {(2*beta+1)/(2*beta+2):.3f})  lisseur {slM:.3f} (cible {2*beta/(2*beta+1):.3f})")
    print(f"  ratio sup lisseur/spectral a N max: {supM[-1]/supS[-1]:.2f}x")
