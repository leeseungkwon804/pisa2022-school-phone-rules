import pandas as pd, numpy as np, pyreadstat, json, time
from scipy import stats
t0=time.time()
PV=[f"PV{i}MATH" for i in range(1,11)]
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
BASE=["CNTSCHID","CNT","W_FSTUWT","ESCS","DISCLIM","IC170Q02JA"]
d,_=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+PV+RW)
s,_=pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV",
      usecols=["CNTSCHID","SC190Q02JA","SC013Q01TA","SC002Q01TA","SC002Q02TA"])
d=d.merge(s,on="CNTSCHID",how="inner")
d=d[d.SC190Q02JA.notna() & d.IC170Q02JA.notna() & (d.IC170Q02JA<=5)]
d["ban"]=(d.SC190Q02JA==1).astype(float)
d["private"]=(d.SC013Q01TA==2).astype(float)
d["schsize"]=np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1,min_count=1))
d["sch_escs"]=d.groupby("CNTSCHID").ESCS.transform("mean")
d["heavy"]=(d.IC170Q02JA>=5).astype(float)
d=d.dropna(subset=["ESCS","DISCLIM","sch_escs","schsize","private"]+PV)
print(f"sample {len(d):,} · countries {d.CNT.nunique()} | {time.time()-t0:.0f}s",flush=True)
CTRL=["ESCS","DISCLIM","sch_escs","private","schsize"]
def beta(X,y,w):
    sw=np.sqrt(w); b,*_=np.linalg.lstsq(X*sw[:,None], y*sw, rcond=None); return b[1]
rows=[]
for cnt,g in d.groupby("CNT"):
    if g.ban.nunique()<2: continue
    na,nc=int((g.ban==1).sum()),int((g.ban==0).sum())
    if na<80 or nc<80: continue
    n=len(g)
    X=np.column_stack([np.ones(n), g.ban.values]+[g[c].values.astype(float) for c in CTRL])
    if np.linalg.matrix_rank(X)<X.shape[1]: continue
    w0=g.W_FSTUWT.values; RWv=g[RW].values
    th=[]; samp=[]
    for p in PV:
        y=g[p].values; b0=beta(X,y,w0); th.append(b0)
        samp.append(sum((beta(X,y,RWv[:,k])-b0)**2 for k in range(80))/20.0)
    th=np.array(th); U=np.mean(samp); B=th.var(ddof=1)
    se=np.sqrt(U+(1+1/10)*B)
    # compliance gap: ban minus no-ban heavy use, student weighted
    ga,gc=g[g.ban==1],g[g.ban==0]
    gap=(np.average(ga.heavy,weights=ga.W_FSTUWT)-np.average(gc.heavy,weights=gc.W_FSTUWT))*100
    rows.append(dict(CNT=cnt,n=n,n_ban=na,beta=float(th.mean()),se=float(se),gap=float(gap)))
    print(f"  {cnt} n={n:>6,} beta {th.mean():+7.2f} SE {se:5.2f} gap {gap:+6.1f}",flush=True)
r=pd.DataFrame(rows); r.to_csv("out/country_brr.csv",index=False)
print(f"\ncountries {len(r)} | {time.time()-t0:.0f}s",flush=True)
