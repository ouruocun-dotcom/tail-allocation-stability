"""Finite-size and amplitude-copula diagnostics. Run after reproduce.py all; see README.md.
python code/supplementary.py [--baseline ROOT] [--figures-only]
Default baseline is this reproducibility folder; ROOT contains results/scan.json.
"""
import argparse,json,time,os
for option in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[option]="1"
from pathlib import Path
import numpy as np
from scipy.linalg import eig
import reproduce as base
ROOT=Path(__file__).resolve().parents[1]
MU,G,L,H=2.5,.65,1.5,8.
S=np.sqrt((MU-2)/MU)

def z_moment_tail(power,t,kappa):
    rate=2*MU-power
    return kappa*MU/(MU-power)*t**(power-MU)+(1-kappa)*t**(-rate)*((2*MU/rate)**2+4*MU**2/rate*np.log(t))

def body_probabilities(delta,kappa):
    A=np.array([[z_moment_tail(k,1,kappa)-z_moment_tail(k,L,kappa),z_moment_tail(k,L,kappa)-z_moment_tail(k,H,kappa)] for k in [0,2]])
    target=.5*np.array([z_moment_tail(k,1,kappa) for k in [0,2]])
    tail=.5*H**delta*np.array([z_moment_tail(k-delta,H,kappa) for k in [0,2]])
    p=np.linalg.solve(A,target-tail)
    assert np.all((p>=0)&(p<=1));assert np.max(abs(A@p+tail-target))<1e-12
    return p

def allocation(z,delta,kappa):
    p0,p1=body_probabilities(delta,kappa)
    return np.where(z<=L,p0,np.where(z<=H,p1,.5*(z/H)**-delta))

def generate(n,rep,kappa,delta):
    seed=2026101291+10000*n+rep
    rng=np.random.default_rng(seed);ii,jj=np.triu_indices(n,1);m=len(ii)
    r1=1+rng.pareto(MU,m);r2=1+rng.pareto(MU,m);u=rng.choice([-1.,1.],m);z=rng.random(m);c=rng.random(m)
    rr=np.where(c<kappa,r1,r2);geom=np.sqrt(r1*rr);positive=z<allocation(geom,delta,kappa)
    W=np.zeros((n,n));W[ii,jj]=G*S*u*r1/np.sqrt(n);W[jj,ii]=G*S*u*rr*np.where(positive,1,-1)/np.sqrt(n)
    return W,dict(seed=seed,max_pair=float(np.max(np.where(positive,geom,0))*G*S/np.sqrt(n)),sample_fraction=float(positive.mean()),sample_covariance=float(S*S*np.mean(r1*rr*np.where(positive,1,-1))))

def run(baseline):
    rows=json.loads((baseline/'results/scan.json').read_text())['rows'];out=ROOT/'results';out.mkdir(exist_ok=True)
    slow=[];sample=[];original_table=[]
    for n in [64,128,256,512,1024]:
        rr=[r for r in rows if r['n']==n and r['delta']=='1'];expected=float(-np.expm1((n*(n-1)//2)*np.log1p(-base.pair_tail(np.sqrt(n)/(G*S),1))))
        slow.append(dict(n=n,blocks=len(rr),pair_threshold_probability=expected,pair_exceed_count=sum(r['max_positive_pair']>1 for r in rr),unstable_count=sum(not r['stable'] for r in rr),pair_vs_abscissa=[[r['max_positive_pair'],r['abscissa']] for r in rr]))
        for d in ['0','1','2','hard']:
            rr=[r for r in rows if r['n']==n and r['delta']==d];cov=np.array([r['sample_normalized_covariance'] for r in rr]);frac=np.array([r['sample_positive_fraction'] for r in rr])
            sample.append(dict(n=n,delta=d,blocks=len(rr),cov_quantiles=np.quantile(cov,[.1,.5,.9]).tolist(),fraction_quantiles=np.quantile(frac,[.1,.5,.9]).tolist()))
            original_table.append(dict(n=n,delta=d,stable=sum(r['stable'] for r in rr),median_abscissa=float(np.median([r['abscissa'] for r in rr])),symmetric_test_pass=sum(r['global_certificate'] for r in rr),blocks=len(rr)))
    copula=[];spectra={};start=time.time()
    for n in [128,256,512]:
        for rep in range(16):
            for kappa in [0.,.5,1.]:
                for d in [0.,2.]:
                    W,diag=generate(n,rep,kappa,d);vals,vecs=eig(W,check_finite=False);idx=int(np.argmax(vals.real));v=vecs[:,idx];res=float(np.linalg.norm(W@v-vals[idx]*v)/(np.linalg.norm(W,'fro')+abs(vals[idx]))/np.linalg.norm(v));assert res<1e-9
                    row=dict(n=n,rep=rep,kappa=kappa,delta=d,abscissa=float(vals[idx].real),stable=bool(vals[idx].real<1),residual=res,**diag);copula.append(row);spectra[f'n{n}_r{rep}_k{kappa}_d{d}']=vals
        print('copula size',n,'elapsed',round(time.time()-start,1),flush=True)
    summary=[]
    for n in [128,256,512]:
        for k in [0.,.5,1.]:
            for d in [0.,2.]:
                rr=[r for r in copula if r['n']==n and r['kappa']==k and r['delta']==d]
                summary.append(dict(n=n,kappa=k,delta=d,blocks=len(rr),stable=sum(r['stable'] for r in rr),median_abscissa=float(np.median([r['abscissa'] for r in rr])),median_max_pair=float(np.median([r['max_pair'] for r in rr]))))
    rng=np.random.default_rng(2026101299)
    paired=[]
    for n in [64,128,256,512,1024]:
        first=sorted([r for r in rows if r['n']==n and r['delta']=='0'],key=lambda r:r['rep'])
        second=sorted([r for r in rows if r['n']==n and r['delta']=='2'],key=lambda r:r['rep'])
        differences=np.array([int(a['stable'])-int(b['stable']) for a,b in zip(second,first)])
        bootstrap=np.mean(differences[rng.integers(0,len(differences),(10000,len(differences)))],axis=1)
        paired.append(dict(n=n,difference=float(differences.mean()),paired_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),resamples=10000))
    output=dict(paired_comparisons=paired,original_rows=len(rows),slow_branch=slow,sample_statistics=sample,original_table=original_table,copula_rows=copula,copula_summary=summary,body_probabilities=[dict(kappa=k,delta=d,body=body_probabilities(d,k).tolist()) for k in [0.,.5,1.] for d in [0.,2.]],independent_blocks=48,paired_matrices=288,max_residual=max(r['residual'] for r in copula),new_seed_base=2026101291)
    (out/'review_diagnostics.json').write_text(json.dumps(output,indent=2));np.savez_compressed(out/'copula_spectra.npz',**spectra);print('Completed 288 copula matrices and supplementary summaries.',flush=True)

def figures(baseline):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    data=json.loads((ROOT/'results/review_diagnostics.json').read_text());rows=json.loads((baseline/'results/scan.json').read_text())['rows'];out=ROOT/'figures';out.mkdir(exist_ok=True)
    colors=['#bb4937','#cf931f','#286c9f','#376f52'];labels=[r'$\delta=0$',r'$\delta=1$',r'$\delta=2$','hard tail'];sp=np.load(baseline/'results/spectra.npz')
    fig,ax=plt.subplots(1,3,figsize=(6.1,2.55))
    for d,c,l in zip([0.,1.,2.,np.inf],colors,labels):
        z=sp[f'n1024_r0_d{base.tag(d)}'];ax[0].scatter(z.real,z.imag,s=2,color=c,label=l,alpha=.5);ax[1].scatter(z.real,z.imag,s=4,color=c,label=l,alpha=.65)
    angle=np.linspace(0,2*np.pi,200)
    for a in ax[:2]:a.plot(G*np.cos(angle),G*np.sin(angle),'k--',lw=.8);a.axvline(1,color='.3',ls=':',lw=.8);a.set(xlabel='Re eigenvalue',ylabel='Im eigenvalue')
    ax[0].set(xlim=(-1,1),ylim=(-1,1),aspect='equal',title='Central window');ax[1].set(title='Full spectrum');ax[1].legend(frameon=False,fontsize=8)
    rr=[r for r in rows if r['delta']=='0'];ax[2].scatter([r['max_positive_pair'] for r in rr],[r['pair_mass'] for r in rr],s=10,color=colors[0],alpha=.6);ax[2].set(xlabel=r'Largest pair $a_N$',ylabel='Mass on pair nodes',ylim=(-.02,1.02),title='Pair localization')
    fig.tight_layout();fig.savefig(out/'bulk_localization.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(6.1,2.7))
    for n,c in zip([64,256,1024],['#aeb3b7','#cf931f','#8b6016']):
        rr=[r for r in rows if r['n']==n and r['delta']=='1'];ax[0].scatter([r['max_positive_pair'] for r in rr],[r['abscissa'] for r in rr],s=22,color=c,label=f'N={n}')
    lim=1.15*max(max(r['max_positive_pair'],r['abscissa']) for r in rows if r['delta']=='1');ax[0].plot([0,lim],[0,lim],'k--',lw=.8);ax[0].set(xlabel=r'Largest pair $a_N$',ylabel='Spectral abscissa',title=r'Slow branch ($\delta=1$)');ax[0].legend(frameon=False,fontsize=8)
    ss=data['slow_branch'];ns=[r['n'] for r in ss];ax[1].plot(ns,[r['pair_threshold_probability'] for r in ss],'k-',label=r'Exact $P(a_N>1)$');ax[1].plot(ns,[r['pair_exceed_count']/r['blocks'] for r in ss],'o--',label=r'Observed $a_N>1$');ax[1].plot(ns,[r['unstable_count']/r['blocks'] for r in ss],'s:',label=r'Observed $\max\,\mathrm{Re}\,\lambda>1$');ax[1].set(xscale='log',xlabel='Network size N',ylabel='Probability / fraction',ylim=(0,.25));ax[1].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(out/'slow_branch.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--baseline',type=Path,default=ROOT);pa.add_argument('--figures-only',action='store_true');args=pa.parse_args()
    if not args.figures_only:run(args.baseline)
    figures(args.baseline)
