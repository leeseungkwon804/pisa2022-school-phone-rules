import pandas as pd, numpy as np, pyreadstat, json, time
from scipy import stats
t0=time.time()
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
SUBJ={"MATH":[f"PV{i}MATH" for i in range(1,11)],
      "READ":[f"PV{i}READ" for i in range(1,11)],
      "SCIE":[f"PV{i}SCIE" for i in range(1,11)]}
ALLPV=[p for v in SUBJ.values() for p in v]
BASE=["CNTSCHID","CNT","W_FSTUWT","ESCS","DISCLIM","IC170Q02JA"]
d,_=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+ALLPV+RW)
SC=["CNTSCHID","SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA",
    "SC013Q01TA","SC002Q01TA","SC002Q02TA"]
s,_=pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV", usecols=SC)
d=d.merge(s,on="CNTSCHID",how="inner")
d=d[d.IC170Q02JA.notna() & (d.IC170Q02JA<=5)]
d["private"]=(d.SC013Q01TA==2).astype(float)
d["schsize"]=np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1,min_count=1))
d["sch_escs"]=d.groupby("CNTSCHID").ESCS.transform("mean")
CTRL=["ESCS","DISCLIM","sch_escs","private","schsize"]
print(f"loaded {len(d):,} | {time.time()-t0:.0f}s",flush=True)

def fit(sub, treat, pvs, label):
    sub=sub.dropna(subset=[treat]+CTRL+pvs).copy()
    n=len(sub)
    if n<5000: return None
    codes,_=pd.factorize(sub.CNT.values)
    G=codes.max()+1
    M=np.column_stack([sub[treat].values.astype(float)]+[sub[c].values.astype(float) for c in CTRL])
    Y=sub[pvs].values.astype(float)
    W0=sub.W_FSTUWT.values.astype(float)
    RWv=sub[RW].values.astype(float)
    def demean(A,w):
        den=np.bincount(codes,weights=w,minlength=G)
        out=np.empty_like(A)
        for j in range(A.shape[1]):
            num=np.bincount(codes,weights=w*A[:,j],minlength=G)
            out[:,j]=A[:,j]-(num/den)[codes]
        return out
    def beta(w, ycol):
        Xc=demean(M,w); yc=demean(ycol[:,None],w)[:,0]
        sw=np.sqrt(w)
        b,*_=np.linalg.lstsq(Xc*sw[:,None], yc*sw, rcond=None)
        return b[0]
    th=[];samp=[]
    for p_i in range(len(pvs)):
        y=Y[:,p_i]
        b0=beta(W0,y); th.append(b0)
        samp.append(sum((beta(RWv[:,k],y)-b0)**2 for k in range(80))/20.0)
    th=np.array(th); U=np.mean(samp); B=th.var(ddof=1); se=np.sqrt(U+1.1*B)
    p=2*(1-stats.norm.cdf(abs(th.mean()/se)))
    print(f"  {label:<44} {th.mean():+7.2f}  SE {se:5.2f}  p {p:.4g}  n={n:,}  [{time.time()-t0:.0f}s]",flush=True)
    return dict(label=label,beta=float(th.mean()),se=float(se),p=float(p),n=n)

res=[]
d["ban"]=(d.SC190Q02JA==1).astype(float)
print("\n=== 1) by domain (ban, full controls, within country) ===",flush=True)
for sub,pvs in SUBJ.items():
    r=fit(d,"ban",pvs,f"phone ban -> {sub}")
    if r: res.append(r)
print("\n=== 2) by policy type (mathematics) ===",flush=True)
NAMES={"SC190Q01JA":"written statement on device use","SC190Q05JA":"formal guidelines for teaching",
       "SC190Q06JA":"teachers set class rules","SC190Q07JA":"rules set with students",
       "SC190Q08JA":"responsible use programme"}
for v,nm in NAMES.items():
    d[v+"_b"]=(d[v]==1).astype(float)
    r=fit(d,v+"_b",SUBJ["MATH"],nm)
    if r: res.append(r)
json.dump(res,open("out/supp.json","w"),ensure_ascii=False,indent=1)
print(f"\ntotal {time.time()-t0:.0f}s",flush=True)
