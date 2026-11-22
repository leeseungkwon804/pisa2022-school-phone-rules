import pandas as pd, numpy as np, pyreadstat, json, time
from scipy import stats
t0=time.time()
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
BASE=["CNTSCHID","CNT","W_FSTUWT","ESCS","ST004D01T","DISCLIM",
      "IC170Q02JA","IC179Q01JA","IC179Q02JA","IC179Q03JA"]
d,_=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+RW)
SC=["CNTSCHID","SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA",
    "SC013Q01TA","SC002Q01TA","SC002Q02TA"]
s,_=pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV", usecols=SC)
d=d.merge(s,on="CNTSCHID",how="inner")
d=d[d.IC170Q02JA.notna() & (d.IC170Q02JA<=5) & d.SC190Q02JA.notna()]
d["ban"]=(d.SC190Q02JA==1).astype(float)
d["private"]=(d.SC013Q01TA==2).astype(float)
d["schsize"]=np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1,min_count=1))
d["sch_escs"]=d.groupby("CNTSCHID").ESCS.transform("mean")
d["female"]=(d.ST004D01T==1).astype(float)
d["heavy"]=(d.IC170Q02JA>=5).astype(float)
# regulation intensity: ban plus five other policies, summed (0-6)
REG=["SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA"]
d["reg_n"]=sum((d[c]==1).astype(float) for c in REG)
CTRL=["ESCS","DISCLIM","sch_escs","private","schsize","female"]
print(f"sample {len(d):,} · countries {d.CNT.nunique()} | {time.time()-t0:.0f}s",flush=True)
def fit(sub, treat, y, label, extra=()):
    cols=[treat,y]+CTRL+list(extra)
    sub=sub.dropna(subset=cols).copy()
    n=len(sub); codes,_=pd.factorize(sub.CNT.values); G=codes.max()+1
    M=np.column_stack([sub[treat].values.astype(float)]
        +[sub[c].values.astype(float) for c in CTRL+list(extra)])
    yv=sub[y].values.astype(float); W0=sub.W_FSTUWT.values.astype(float); RWv=sub[RW].values.astype(float)
    def dm(A,w):
        den=np.bincount(codes,weights=w,minlength=G); out=np.empty_like(A)
        for j in range(A.shape[1]):
            out[:,j]=A[:,j]-(np.bincount(codes,weights=w*A[:,j],minlength=G)/den)[codes]
        return out
    def beta(w):
        Xc=dm(M,w); yc=dm(yv[:,None],w)[:,0]; sw=np.sqrt(w)
        b,*_=np.linalg.lstsq(Xc*sw[:,None], yc*sw, rcond=None); return b[0]
    b0=beta(W0); U=sum((beta(RWv[:,k])-b0)**2 for k in range(80))/20.0
    se=np.sqrt(U); p=2*(1-stats.norm.cdf(abs(b0/se)))
    print(f"  {label:<46} {b0:+8.4f}  SE {se:.4f}  p {p:.3g}  n={n:,}  [{time.time()-t0:.0f}s]",flush=True)
    return dict(label=label,beta=float(b0),se=float(se),p=float(p),n=n)
res=[]
print("\n=== 1) dose response: number of rules (0-6) -> endorsement ===",flush=True)
res.append(fit(d,"reg_n","IC179Q01JA","rule count -> endorsement"))
res.append(fit(d,"reg_n","IC179Q01JA","rule count -> endorsement, own use held constant",extra=("heavy",)))
print("\n=== 2) other attitude items (convergent check) ===",flush=True)
res.append(fit(d,"ban","IC179Q02JA","ban -> laptops should not be brought"))
res.append(fit(d,"ban","IC179Q03JA","ban -> students should help set rules"))
print("\n=== 3) between-country heterogeneity ===",flush=True)
rows=[]
for cnt,g in d.groupby("CNT"):
    g2=g.dropna(subset=["ban","IC179Q01JA"]+CTRL)
    if g2.ban.nunique()<2 or len(g2)<800: continue
    if (g2.ban==1).sum()<150 or (g2.ban==0).sum()<150: continue
    X=np.column_stack([np.ones(len(g2)), g2.ban.values]+[g2[c].values.astype(float) for c in CTRL])
    sw=np.sqrt(g2.W_FSTUWT.values)
    b,*_=np.linalg.lstsq(X*sw[:,None], g2.IC179Q01JA.values.astype(float)*sw, rcond=None)
    rows.append((cnt,len(g2),b[1]))
r=pd.DataFrame(rows,columns=["CNT","n","beta"])
print(f"  countries {len(r)} · median {r.beta.median():+.4f} · positive {int((r.beta>0).sum())}/{len(r)}",flush=True)
print(f"  IQR {r.beta.quantile(.25):+.4f} ~ {r.beta.quantile(.75):+.4f}",flush=True)
r.to_csv("out/paperB_country.csv",index=False)
json.dump(res,open("out/paperB2.json","w"),ensure_ascii=False,indent=1)
print(f"\n{time.time()-t0:.0f}s",flush=True)
