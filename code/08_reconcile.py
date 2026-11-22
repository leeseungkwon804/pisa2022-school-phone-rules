import pandas as pd, numpy as np, pyreadstat, time
from scipy import stats
t0=time.time()
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
PVM=[f"PV{i}MATH" for i in range(1,11)]
PVR=[f"PV{i}READ" for i in range(1,11)]; PVS=[f"PV{i}SCIE" for i in range(1,11)]
BASE=["CNTSCHID","CNT","W_FSTUWT","ESCS","DISCLIM","IC170Q02JA"]
d,_=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+PVM+PVR+PVS+RW)
s,_=pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV",
    usecols=["CNTSCHID","SC190Q02JA","SC013Q01TA","SC002Q01TA","SC002Q02TA"])
d=d.merge(s,on="CNTSCHID",how="inner")
d=d[d.IC170Q02JA.notna() & (d.IC170Q02JA<=5)]
d["ban"]=(d.SC190Q02JA==1).astype(float)
d["private"]=(d.SC013Q01TA==2).astype(float)
d["schsize"]=np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1,min_count=1))
d["sch_escs"]=d.groupby("CNTSCHID").ESCS.transform("mean")
CTRL=["ESCS","DISCLIM","sch_escs","private","schsize"]
def run(sub,label,pvs=PVM):
    sub=sub.dropna(subset=["ban"]+CTRL+pvs).copy()
    n=len(sub); codes,_=pd.factorize(sub.CNT.values); G=codes.max()+1
    M=np.column_stack([sub.ban.values]+[sub[c].values.astype(float) for c in CTRL])
    Y=sub[pvs].values.astype(float); W0=sub.W_FSTUWT.values.astype(float); RWv=sub[RW].values.astype(float)
    def dm(A,w):
        den=np.bincount(codes,weights=w,minlength=G); out=np.empty_like(A)
        for j in range(A.shape[1]):
            out[:,j]=A[:,j]-(np.bincount(codes,weights=w*A[:,j],minlength=G)/den)[codes]
        return out
    def beta(w,y):
        Xc=dm(M,w); yc=dm(y[:,None],w)[:,0]; sw=np.sqrt(w)
        b,*_=np.linalg.lstsq(Xc*sw[:,None], yc*sw, rcond=None); return b[0]
    th=[];samp=[]
    for i in range(len(pvs)):
        y=Y[:,i]; b0=beta(W0,y); th.append(b0)
        samp.append(sum((beta(RWv[:,k],y)-b0)**2 for k in range(80))/20.0)
    th=np.array(th); se=np.sqrt(np.mean(samp)+1.1*th.var(ddof=1))
    print(f"  {label:<46} {th.mean():+7.2f}  SE {se:5.2f}  p {2*(1-stats.norm.cdf(abs(th.mean()/se))):.4g}  n={n:,} countries={sub.CNT.nunique()}",flush=True)
print("=== where the two estimates differ ===",flush=True)
run(d,"A) drop on mathematics PVs only")
dd=d.dropna(subset=PVR+PVS)
run(dd,"B) drop on reading and science PVs too")
print(f"\n  A sample {len(d.dropna(subset=['ban']+CTRL+PVM)):,} vs B sample {len(dd.dropna(subset=['ban']+CTRL+PVM)):,}",flush=True)
print(f"  {time.time()-t0:.0f}s",flush=True)
