"""Generate the primary network scan and figures; see README.md for usage."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eig, eigvals, eigvalsh

ROOT = Path(__file__).resolve().parents[1]
MU, G, LOW, HIGH = 2.5, .65, 1.5, 8.
SCALE = np.sqrt((MU - 2) / MU)
DELTAS = (0., 1., 2., np.inf)


def tag(delta):
    return "hard" if np.isinf(delta) else str(int(delta))


def probabilities(delta):
    A = np.array([[1 - LOW**-MU, LOW**-MU - HIGH**-MU],
                  [1 - LOW**(2-MU), LOW**(2-MU)-HIGH**(2-MU)]])
    if np.isinf(delta):
        tail = np.zeros(2)
    else:
        tail = .5 * np.array([MU/(MU+delta)*HIGH**-MU,
                              (MU-2)/(MU+delta-2)*HIGH**(2-MU)])
    body = np.linalg.solve(A, .5 - tail)
    if not np.all((body >= 0) & (body <= 1)):
        raise ValueError("Invalid probability")
    return body


def probability(r, delta):
    p0, p1 = probabilities(delta)
    tail = np.zeros_like(np.asarray(r)) if np.isinf(delta) else .5*(np.asarray(r)/HIGH)**(-delta)
    return np.where(np.asarray(r) <= LOW, p0,
                    np.where(np.asarray(r) <= HIGH, p1, tail))


def pair_tail(t, delta):
    """P(R*1[q=+1]>t), including finite-N body branches."""
    t = np.asarray(t, dtype=float)
    p0, p1 = probabilities(delta)
    tail0 = 0. if np.isinf(delta) else .5*MU/(MU+delta)*HIGH**-MU
    middle = p1*(t**-MU-HIGH**-MU)+tail0
    lower = p0*(t**-MU-LOW**-MU)+p1*(LOW**-MU-HIGH**-MU)+tail0
    upper = np.zeros_like(t) if np.isinf(delta) else .5*MU/(MU+delta)*HIGH**delta*t**(-MU-delta)
    return np.where(t < 1, .5, np.where(t <= LOW, lower,
                    np.where(t <= HIGH, middle, upper)))


def maximum_cdf(a, n, delta):
    t = np.asarray(a)*np.sqrt(n)/(G*SCALE)
    return np.exp((n*(n-1)//2)*np.log1p(-pair_tail(t, delta)))


def extreme_scale(n, delta):
    c = .5*MU/(MU+delta)*HIGH**delta
    return G*SCALE*(n*(n-1)/2*c)**(1/(MU+delta))/np.sqrt(n)


def realization(n, rep, delta):
    seed = 2026101201 + 10000*n + rep
    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(n, 1)
    R = 1+rng.pareto(MU, len(i))
    u = rng.choice([-1., 1.], len(i))
    z = rng.random(len(i))
    positive = z < probability(R, delta)
    a = G*SCALE*R/np.sqrt(n)
    W = np.zeros((n, n))
    W[i, j] = a*u
    W[j, i] = a*u*np.where(positive, 1., -1.)
    selected = np.flatnonzero(positive)
    selected = int(selected[np.argmax(a[selected])]) if len(selected) else None
    return W, (i, j, a, positive, selected, seed)


def disc_certificate(W, nodes):
    n = len(W)
    k, l = nodes
    a = float(abs(W[k, l]))
    assert W[k,l] == W[l,k]
    ids = np.delete(np.arange(n), nodes)
    U, V = W[np.ix_(nodes, ids)], W[np.ix_(ids, nodes)]
    B = W[np.ix_(ids, ids)]
    b = float(eigvalsh((B+B.T)/2, subset_by_index=[n-3, n-3], check_finite=False)[0])
    h = float(np.linalg.norm(U, 2)*np.linalg.norm(V, 2))
    gap = a-b
    r = min(gap/2, (a-1)*.999, a*.999)
    passed = bool(r > 0 and h < r*(gap-r))
    return dict(remainder_edge=b, coupling_product=h, disc_radius=float(r),
                disc_margin=float(r*(gap-r)-h), disc_pass=passed,
                disc_lower=float(a-r) if passed else None)


def run():
    path = ROOT / "results"
    path.mkdir(exist_ok=True)
    rows = []
    spectra = {}
    vectors = {}
    t0 = time.time()
    for n in [64,128,256,512,1024]:
        for rep in range(16 if n==1024 else 32):
            for delta in DELTAS:
                W, (i,j,a,positive,selected,seed) = realization(n,rep,delta)
                vals, vecs = eig(W, check_finite=False, overwrite_a=False)
                index = int(np.argmax(vals.real))
                leading = vals[index]
                vector = vecs[:,index] / np.linalg.norm(vecs[:,index])
                sedges = eigvalsh((W+W.T)/2, subset_by_index=[n-2,n-1], check_finite=False)
                nodes = [int(i[selected]),int(j[selected])] if selected is not None else [0,1]
                cert = disc_certificate(W,nodes) if selected is not None else dict(disc_pass=False)
                residual = np.linalg.norm(W@vector-leading*vector)/(np.linalg.norm(W,"fro")+abs(leading))
                if cert["disc_pass"]:
                    center = a[selected]
                    inside = np.abs(vals-center)<cert["disc_radius"]
                    assert int(inside.sum())==1
                    assert abs(vals[inside][0].imag)<1e-7
                    assert leading.real>cert["disc_lower"]-1e-7
                assert leading.real <= sedges[-1]+1e-7
                assert residual<1e-9
                key = f"n{n}_r{rep}_d{tag(delta)}"
                spectra[key] = vals
                vectors[key] = vector
                rows.append(dict(n=n,rep=rep,seed=seed,delta=tag(delta),
                  abscissa=float(leading.real), leading_imaginary=float(leading.imag),
                  symmetric_edge=float(sedges[-1]), symmetric_second=float(sedges[0]),
                  stable=bool(leading.real<1),global_certificate=bool(sedges[-1]<1),
                  max_positive_pair=float(a[selected]) if selected is not None else 0.,
                  selected_nodes=nodes, ipr=float(np.sum(abs(vector)**4)),
                  pair_mass=float(np.sum(abs(vector[nodes])**2)),
                  eigenpair_residual=float(residual), sample_positive_fraction=float(positive.mean()),
                  sample_normalized_covariance=float(np.mean(np.where(positive,1.,-1.)*a*a)*n/G**2),
                  sample_normalized_second_moment=float(np.mean(a*a)*n/G**2), **cert))
        path.joinpath("scan.json").write_text(json.dumps(dict(mu=MU,g=G,low=LOW,high=HIGH,
                 rows=rows,complete=False,seconds=time.time()-t0),indent=2))
        print("size",n,[(tag(d),sum(r["stable"] for r in rows if r["n"]==n and r["delta"]==tag(d)),
                              sum(r["global_certificate"] for r in rows if r["n"]==n and r["delta"]==tag(d)))
                         for d in DELTAS],"seconds",round(time.time()-t0,1),flush=True)
    np.savez_compressed(path/"spectra.npz",**spectra)
    np.savez_compressed(path/"leading_vectors.npz",**vectors)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [path/"spectra.npz",path/"leading_vectors.npz"]}
    path.joinpath("scan.json").write_text(json.dumps(dict(mu=MU,g=G,low=LOW,high=HIGH,
                 rows=rows,complete=True,hashes=hashes,seconds=time.time()-t0),indent=2))


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"ps.fonttype":42})
    path=ROOT/"figures"
    path.mkdir(exist_ok=True)
    data=json.loads((ROOT/"results/scan.json").read_text())
    assert data["complete"]
    rows=data["rows"]
    colors=["#bb4937","#cf931f","#286c9f","#376f52"]
    labels=[r"$\delta=0$",r"$\delta=1$",r"$\delta=2$","hard tail"]
    def save(fig,name):
        fig.savefig(path/f"{name}.pdf",bbox_inches="tight")
        fig.savefig(path/f"{name}.png",dpi=180,bbox_inches="tight")
        plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(9,3.2))
    x=np.geomspace(1,100,500)
    for d,c,l in zip(DELTAS,colors,labels):
        ax[0].plot(x,probability(x,d),color=c,label=l)
        ax[1].loglog(x,pair_tail(x,d),color=c,label=l)
    ax[0].set(xscale="log",xlabel="Pair magnitude R",ylabel="Reinforcing probability p(R)",ylim=(-.02,1.02))
    ax[1].set(xlabel="Magnitude threshold r",ylabel="P(R > r, q = +1)",ylim=(1e-8,1))
    ax[0].legend(frameon=False,fontsize=9)
    fig.tight_layout();save(fig,"allocation")
    fig,ax=plt.subplots(1,2,figsize=(9,3.2))
    for d,c,l in zip(DELTAS,colors,labels):
        ns=[];prob=[];lo=[];hi=[];med=[]
        for n in [64,128,256,512,1024]:
            rr=[r for r in rows if r["n"]==n and r["delta"]==tag(d)]
            size=len(rr);p=np.mean([r["stable"] for r in rr]);z=1.95996398454
            center=(p+z*z/(2*size))/(1+z*z/size)
            half=z*np.sqrt(p*(1-p)/size+z*z/(4*size*size))/(1+z*z/size)
            ns.append(n);prob.append(p);lo.append(p-center+half);hi.append(center+half-p)
            med.append(np.median([r["abscissa"] for r in rr]))
        ax[0].errorbar(ns,prob,yerr=[lo,hi],fmt="o-",color=c,label=l,capsize=3)
        ax[1].plot(ns,med,"o-",color=c,label=l)
    ax[0].set(xscale="log",xlabel="Network size N",ylabel="Fraction locally stable",ylim=(-.04,1.04))
    ax[1].set(xscale="log",xlabel="Network size N",ylabel="Median spectral abscissa")
    ax[1].axhline(1,color=".3",ls="--",lw=1)
    ax[0].legend(frameon=False,fontsize=9)
    fig.tight_layout();save(fig,"stability")
    fig,ax=plt.subplots(1,2,figsize=(9,3.2))
    for n,c in zip([64,256,1024],["#aeb3b7","#6592ad","#1a4f76"]):
        rr=[r for r in rows if r["n"]==n and r["delta"]=="0"]
        a=np.array([r["max_positive_pair"] for r in rr])
        edge=np.array([r["abscissa"] for r in rr])
        ax[0].scatter(a,edge,color=c,s=20,label=f"N={n}",alpha=.8)
        norm=np.sort(edge/extreme_scale(n,0))
        ax[1].step(norm,np.arange(1,len(norm)+1)/len(norm),where="post",color=c,label=f"N={n}")
    lim=max(r["abscissa"] for r in rows if r["delta"]=="0")
    ax[0].plot([0,lim],[0,lim],"--",color=".4",lw=1)
    x=np.linspace(.1,4,600)
    ax[1].plot(x,np.exp(-x**-MU),"k--",lw=1,label="Asymptotic Frechet")
    ax[0].set(xlabel="Largest reinforcing pair a",ylabel="Spectral abscissa")
    ax[1].set(xlabel="Spectral abscissa / b(N)",ylabel="Empirical CDF",xlim=(0,4),ylim=(0,1.03))
    ax[0].legend(frameon=False,fontsize=9);ax[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,"extremes")
    fig,ax=plt.subplots(1,2,figsize=(9,3.2))
    sp=np.load(ROOT/"results/spectra.npz")
    for d,c,l in zip(DELTAS,colors,labels):
        z=sp[f"n1024_r0_d{tag(d)}"]
        ax[0].scatter(z.real,z.imag,s=3,color=c,label=l,alpha=.55)
    angle=np.linspace(0,2*np.pi,300)
    ax[0].plot(G*np.cos(angle),G*np.sin(angle),"k--",lw=1)
    ax[0].axvline(1,color=".4",ls=":",lw=1)
    ax[0].set(xlabel="Re eigenvalue",ylabel="Im eigenvalue",xlim=(-2,2),ylim=(-2,2),aspect="equal")
    ax[0].legend(frameon=False,fontsize=8,markerscale=2)
    rr=[r for r in rows if r["delta"]=="0"]
    ax[1].scatter([r["max_positive_pair"] for r in rr],[r["pair_mass"] for r in rr],s=13,color=colors[0],alpha=.65)
    ax[1].set(xlabel="Largest reinforcing pair a",ylabel="Leading-vector mass on its nodes",ylim=(-.02,1.02))
    fig.tight_layout();save(fig,"bulk_localization")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["run","figures","all"])
    mode=parser.parse_args().mode
    if mode in ["run","all"]: run()
    if mode in ["figures","all"]: figures()
